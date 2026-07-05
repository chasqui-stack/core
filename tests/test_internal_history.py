"""Internal conversation read (ADR-011): GET /conversations/{channel}/{external_id}/messages.

Gateway-facing, INTERNAL_API_KEY-protected, generic (channel param), read-only.
Mirrors the admin timeline serialization (has_media boolean, no media payloads).
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import func, select

from app.core.config import settings
from app.models import Contact, Conversation, Message


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


async def make_web_contact(session, external_id: str) -> Contact:
    contact = Contact(channel="web", external_id=external_id, display_name=None)
    session.add(contact)
    await session.flush()
    return contact


async def make_thread(session, contact: Contact, texts: list[str]) -> Conversation:
    conversation = Conversation(contact_id=contact.id)
    session.add(conversation)
    await session.flush()
    base = _now()
    for i, text in enumerate(texts):
        session.add(
            Message(
                conversation_id=conversation.id,
                direction="in" if i % 2 == 0 else "out",
                type="text",
                text=text,
                created_at=base + timedelta(seconds=i),
            )
        )
    await session.flush()
    return conversation


@pytest.fixture
def internal_key(monkeypatch):
    monkeypatch.setattr(settings, "internal_api_key", "test-secret")
    return "test-secret"


def _url(external_id: str, channel: str = "web") -> str:
    return f"/conversations/{channel}/{external_id}/messages"


async def test_rejects_missing_or_wrong_key(client, session, internal_key):
    vid = uuid.uuid4().hex
    await make_thread(session, await make_web_contact(session, vid), ["hi"])
    await session.commit()

    assert (await client.get(_url(vid))).status_code == 401
    bad = await client.get(_url(vid), headers={"X-Internal-API-Key": "nope"})
    assert bad.status_code == 401


async def test_open_when_key_unset(client, session):
    # _no_dev_env sets internal_api_key=None → the seam is open in local dev.
    vid = uuid.uuid4().hex
    await make_thread(session, await make_web_contact(session, vid), ["hi"])
    await session.commit()

    resp = await client.get(_url(vid))
    assert resp.status_code == 200


async def test_returns_recent_newest_first_scoped(client, session, internal_key):
    vid = uuid.uuid4().hex
    await make_thread(
        session, await make_web_contact(session, vid), ["one", "two", "three"]
    )
    # A different visitor — must NOT leak into the first one's history.
    other = uuid.uuid4().hex
    await make_thread(session, await make_web_contact(session, other), ["secret"])
    await session.commit()

    resp = await client.get(
        _url(vid), params={"limit": 2}, headers={"X-Internal-API-Key": internal_key}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    assert [m["text"] for m in body["items"]] == ["three", "two"]


async def test_channel_is_generic_not_hardcoded(client, session, internal_key):
    # The same external_id under a different channel is a different contact.
    vid = uuid.uuid4().hex
    tg = Contact(channel="telegram", external_id=vid)
    session.add(tg)
    await session.flush()
    await make_thread(session, tg, ["telegram msg"])
    await session.commit()

    resp = await client.get(
        _url(vid, channel="telegram"), headers={"X-Internal-API-Key": internal_key}
    )
    assert resp.status_code == 200
    assert [m["text"] for m in resp.json()["items"]] == ["telegram msg"]


async def test_404_when_contact_absent_and_read_only(client, session, internal_key):
    vid = uuid.uuid4().hex
    resp = await client.get(_url(vid), headers={"X-Internal-API-Key": internal_key})
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "NO_CONTACT"

    # Read-only: the 404 must NOT have created a contact.
    count = (
        await session.exec(
            select(func.count()).select_from(Contact).where(Contact.external_id == vid)
        )
    ).one()
    assert count == 0


async def test_contact_without_conversation_is_empty(client, session, internal_key):
    vid = uuid.uuid4().hex
    await make_web_contact(session, vid)
    await session.commit()

    resp = await client.get(_url(vid), headers={"X-Internal-API-Key": internal_key})
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "total": 0}


async def test_never_serializes_media_payload(client, session, internal_key):
    vid = uuid.uuid4().hex
    await make_thread(session, await make_web_contact(session, vid), ["one"])
    await session.commit()

    resp = await client.get(_url(vid), headers={"X-Internal-API-Key": internal_key})
    item = resp.json()["items"][0]
    assert "has_media" in item and isinstance(item["has_media"], bool)
    # No raw payloads / embeddings leak through the gateway read.
    assert "media_url" not in item
    assert "embedding" not in item
