"""Sprint 6: GET /admin/media/{message_id} — presigned URL as JSON (ADR-003)."""

import uuid

import pytest

from app.core import storage
from app.models import Contact, Conversation, Message
from app.services.admin_service import create_admin_access_token


@pytest.fixture
def admin_headers() -> dict:
    token = create_admin_access_token(uuid.uuid4(), "admin@test.local", "super_admin")
    return {"Authorization": f"Bearer {token}"}


async def make_message(session, media_url: str | None) -> Message:
    contact = Contact(channel="whatsapp", external_id=f"PE.{uuid.uuid4().hex[:12]}")
    session.add(contact)
    await session.flush()
    conversation = Conversation(contact_id=contact.id)
    session.add(conversation)
    await session.flush()
    message = Message(
        conversation_id=conversation.id,
        direction="in",
        type="image",
        media_url=media_url,
    )
    session.add(message)
    await session.flush()
    return message


async def test_media_requires_auth(client):
    assert (await client.get(f"/admin/media/{uuid.uuid4()}")).status_code == 401


async def test_media_404_when_message_missing(client, admin_headers):
    response = await client.get(f"/admin/media/{uuid.uuid4()}", headers=admin_headers)
    assert response.status_code == 404


async def test_media_404_when_no_stored_object(client, session, admin_headers):
    # NULL media_url (text or pre-storage media) and foreign URLs are both
    # "nothing we can serve" — only media/ keys are ours (ADR-003).
    no_media = await make_message(session, media_url=None)
    foreign = await make_message(session, media_url="https://example.com/x.jpg")
    await session.commit()

    for message in (no_media, foreign):
        response = await client.get(
            f"/admin/media/{message.id}", headers=admin_headers
        )
        assert response.status_code == 404


async def test_media_503_when_storage_unconfigured(
    client, session, admin_headers, monkeypatch
):
    message = await make_message(session, media_url="media/c/m.jpg")
    await session.commit()
    monkeypatch.setattr(storage, "is_configured", lambda: False)

    response = await client.get(f"/admin/media/{message.id}", headers=admin_headers)

    assert response.status_code == 503


async def test_media_returns_presigned_url(client, session, admin_headers, monkeypatch):
    message = await make_message(session, media_url="media/c/m.jpg")
    await session.commit()
    monkeypatch.setattr(storage, "is_configured", lambda: True)
    monkeypatch.setattr(
        storage, "presigned_get", lambda key, expires=300: f"https://signed.test/{key}"
    )

    response = await client.get(f"/admin/media/{message.id}", headers=admin_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["url"] == "https://signed.test/media/c/m.jpg"
    assert body["expires_in"] == storage.PRESIGN_EXPIRES_SECONDS


async def test_messages_listing_exposes_has_media_flag(
    client, session, admin_headers
):
    message = await make_message(session, media_url="media/c/m.jpg")
    await session.commit()

    conversation = await session.get(Conversation, message.conversation_id)
    response = await client.get(
        f"/admin/contacts/{conversation.contact_id}/messages", headers=admin_headers
    )

    item = response.json()["items"][0]
    assert item["has_media"] is True
    assert "media_url" not in item  # the key never leaves the core
