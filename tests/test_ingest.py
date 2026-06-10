"""Sprint 1 acceptance: canonical /ingest pipeline (DB-backed).

These tests exercise persistence/upsert mechanics, so the agent turn is
stubbed (the real orchestrator is covered in test_orchestrator.py).
"""

import base64
import uuid

import pytest
from sqlmodel import select

from app.core import storage
from app.models import Contact, Conversation, Message
from app.schemas.ingest import OutboundMessage
from app.services import orchestrator

BSUID = "bsuid-3EB0C431D26A1916E55F"


@pytest.fixture(autouse=True)
def stub_agent_turn(monkeypatch):
    """Replace the LangGraph turn with a canned echo (no LLM in these tests)."""

    async def fake_run_turn(session, conversation, inbound, **kwargs):
        text = f"Echo: {inbound.text}" if inbound.text else f"Recibí tu {inbound.type}."
        return [OutboundMessage(type="text", text=text)]

    monkeypatch.setattr(orchestrator, "run_turn", fake_run_turn)


def canonical_payload(**overrides) -> dict:
    payload = {
        "channel": "whatsapp",
        "contact": {
            "external_id": BSUID,
            "wa_id": None,
            "display_name": "Juan Pérez",
            "metadata": {},
        },
        "message": {
            "type": "text",
            "text": "¿Tienen tienda en Miraflores?",
            "media_url": None,
            "raw": {"wamid": "wamid.test123"},
        },
        "received_at": "2026-06-09T15:30:00Z",
    }
    payload.update(overrides)
    return payload


async def test_ingest_creates_contact_conversation_and_messages(client, session):
    resp = await client.post("/ingest", json=canonical_payload())
    assert resp.status_code == 200

    body = resp.json()
    # Canonical response shape (§5)
    assert uuid.UUID(body["conversation_id"])
    assert len(body["messages"]) >= 1
    assert body["messages"][0]["type"] == "text"
    assert body["messages"][0]["text"]

    # Rows landed
    contact = (await session.exec(select(Contact))).one()
    assert contact.channel == "whatsapp"
    assert contact.display_name == "Juan Pérez"

    conversation = (await session.exec(select(Conversation))).one()
    assert conversation.contact_id == contact.id
    assert str(conversation.id) == body["conversation_id"]

    messages = (await session.exec(select(Message).order_by(Message.created_at))).all()
    directions = [m.direction for m in messages]
    assert "in" in directions and "out" in directions
    inbound = next(m for m in messages if m.direction == "in")
    assert inbound.text == "¿Tienen tienda en Miraflores?"
    assert inbound.meta == {"wamid": "wamid.test123"}


async def test_ingest_contact_upsert_is_idempotent(client, session):
    r1 = await client.post("/ingest", json=canonical_payload())
    r2 = await client.post("/ingest", json=canonical_payload())
    assert r1.status_code == r2.status_code == 200

    # Same contact + same single conversation thread reused
    assert r1.json()["conversation_id"] == r2.json()["conversation_id"]

    contacts = (await session.exec(select(Contact))).all()
    conversations = (await session.exec(select(Conversation))).all()
    messages = (await session.exec(select(Message))).all()
    assert len(contacts) == 1
    assert len(conversations) == 1
    assert len(messages) == 4  # 2 turns × (in + out)


async def test_bsuid_stored_as_identity_wa_id_nullable(client, session):
    resp = await client.post("/ingest", json=canonical_payload())
    assert resp.status_code == 200

    contact = (await session.exec(select(Contact))).one()
    # BSUID-first (§10): external_id holds the BSUID, wa_id may be null
    assert contact.external_id == BSUID
    assert contact.wa_id is None


async def test_upsert_refreshes_wa_id_and_display_name(client, session):
    await client.post("/ingest", json=canonical_payload())

    updated = canonical_payload(
        contact={
            "external_id": BSUID,
            "wa_id": "51999888777",
            "display_name": "Juan P. Quispe",
            "metadata": {"lang": "es"},
        }
    )
    await client.post("/ingest", json=updated)

    contact = (await session.exec(select(Contact))).one()
    assert contact.wa_id == "51999888777"
    assert contact.display_name == "Juan P. Quispe"
    assert contact.meta == {"lang": "es"}


async def test_different_channel_same_external_id_is_a_new_contact(client, session):
    await client.post("/ingest", json=canonical_payload())
    await client.post("/ingest", json=canonical_payload(channel="web"))

    contacts = (await session.exec(select(Contact))).all()
    assert len(contacts) == 2
    assert {c.channel for c in contacts} == {"whatsapp", "web"}


# --- Sprint 6: media persistence (ADR-003) -------------------------------

JPEG_DATA_URI = "data:image/jpeg;base64," + base64.b64encode(b"fake-jpeg").decode()


def media_payload(media_url: str = JPEG_DATA_URI) -> dict:
    return canonical_payload(
        message={
            "type": "image",
            "text": None,
            "media_url": media_url,
            "raw": {"media_id": "mid.test"},
        }
    )


async def test_media_not_persisted_when_storage_unconfigured(client, session):
    # Default test settings have no STORAGE_* — exactly today's behavior.
    resp = await client.post("/ingest", json=media_payload())
    assert resp.status_code == 200

    inbound = (
        await session.exec(select(Message).where(Message.direction == "in"))
    ).one()
    assert inbound.media_url is None


async def test_media_uploaded_and_key_persisted(client, session, monkeypatch):
    uploads: list[tuple[str, bytes, str]] = []

    async def fake_put_media(key, data, content_type):
        uploads.append((key, data, content_type))

    monkeypatch.setattr(storage, "is_configured", lambda: True)
    monkeypatch.setattr(storage, "put_media", fake_put_media)

    resp = await client.post("/ingest", json=media_payload())
    assert resp.status_code == 200

    contact = (await session.exec(select(Contact))).one()
    inbound = (
        await session.exec(select(Message).where(Message.direction == "in"))
    ).one()
    expected_key = f"media/{contact.id}/{inbound.id}.jpg"
    assert inbound.media_url == expected_key
    assert uploads == [(expected_key, b"fake-jpeg", "image/jpeg")]


async def test_media_upload_failure_never_breaks_the_turn(
    client, session, monkeypatch
):
    async def broken_put_media(key, data, content_type):
        raise RuntimeError("bucket down")

    monkeypatch.setattr(storage, "is_configured", lambda: True)
    monkeypatch.setattr(storage, "put_media", broken_put_media)

    resp = await client.post("/ingest", json=media_payload())

    assert resp.status_code == 200  # the turn answered anyway
    inbound = (
        await session.exec(select(Message).where(Message.direction == "in"))
    ).one()
    assert inbound.media_url is None  # log + NULL, the embeddings pattern


async def test_malformed_data_uri_is_treated_as_upload_failure(
    client, session, monkeypatch
):
    monkeypatch.setattr(storage, "is_configured", lambda: True)

    resp = await client.post("/ingest", json=media_payload(media_url="data:image/jpeg"))

    assert resp.status_code == 200
    inbound = (
        await session.exec(select(Message).where(Message.direction == "in"))
    ).one()
    assert inbound.media_url is None


async def test_non_data_media_url_persists_verbatim(client, session):
    resp = await client.post(
        "/ingest", json=media_payload(media_url="https://cdn.example.com/x.jpg")
    )
    assert resp.status_code == 200

    inbound = (
        await session.exec(select(Message).where(Message.direction == "in"))
    ).one()
    assert inbound.media_url == "https://cdn.example.com/x.jpg"
