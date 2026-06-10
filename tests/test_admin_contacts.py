"""Sprint 5: /admin/contacts — read-only conversation inspection."""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.models import Contact, Conversation, Memory, Message
from app.services.admin_service import create_admin_access_token


@pytest.fixture
def admin_headers() -> dict:
    token = create_admin_access_token(uuid.uuid4(), "admin@test.local", "super_admin")
    return {"Authorization": f"Bearer {token}"}


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


async def make_contact(
    session, *, display_name: str = "Ada", external_id: str | None = None
) -> Contact:
    contact = Contact(
        channel="whatsapp",
        external_id=external_id or f"PE.{uuid.uuid4().hex[:12]}",
        display_name=display_name,
    )
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


async def test_contacts_require_auth(client):
    assert (await client.get("/admin/contacts")).status_code == 401


async def test_list_contacts_empty(client, admin_headers):
    response = await client.get("/admin/contacts", headers=admin_headers)

    assert response.status_code == 200
    assert response.json() == {"items": [], "total": 0}


async def test_list_contacts_with_preview_and_counts(client, session, admin_headers):
    ada = await make_contact(session, display_name="Ada")
    await make_thread(session, ada, ["hi", "hello!", "thanks"])
    await make_contact(session, display_name="NoThread")
    await session.commit()

    response = await client.get("/admin/contacts", headers=admin_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    by_name = {c["display_name"]: c for c in body["items"]}

    assert by_name["Ada"]["message_count"] == 3
    assert by_name["Ada"]["last_message"]["text"] == "thanks"
    assert by_name["Ada"]["last_message"]["direction"] == "in"

    assert by_name["NoThread"]["message_count"] == 0
    assert by_name["NoThread"]["last_message"] is None


async def test_list_contacts_search_and_pagination(client, session, admin_headers):
    await make_contact(session, display_name="Ada Lovelace")
    await make_contact(session, display_name="Grace Hopper")
    await session.commit()

    found = await client.get(
        "/admin/contacts", params={"search": "lovel"}, headers=admin_headers
    )
    assert found.json()["total"] == 1
    assert found.json()["items"][0]["display_name"] == "Ada Lovelace"

    page = await client.get(
        "/admin/contacts", params={"limit": 1, "offset": 1}, headers=admin_headers
    )
    assert page.json()["total"] == 2
    assert len(page.json()["items"]) == 1


async def test_contact_detail_and_404(client, session, admin_headers):
    ada = await make_contact(session, display_name="Ada")
    await session.commit()

    ok = await client.get(f"/admin/contacts/{ada.id}", headers=admin_headers)
    assert ok.status_code == 200
    assert ok.json()["display_name"] == "Ada"

    missing = await client.get(f"/admin/contacts/{uuid.uuid4()}", headers=admin_headers)
    assert missing.status_code == 404


async def test_messages_newest_first_with_total(client, session, admin_headers):
    ada = await make_contact(session)
    await make_thread(session, ada, ["one", "two", "three"])
    await session.commit()

    response = await client.get(
        f"/admin/contacts/{ada.id}/messages",
        params={"limit": 2},
        headers=admin_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert [m["text"] for m in body["items"]] == ["three", "two"]
    # Admin responses never carry media payloads
    assert "media_url" not in body["items"][0]


async def test_messages_for_contact_without_conversation(
    client, session, admin_headers
):
    ada = await make_contact(session)
    await session.commit()

    response = await client.get(
        f"/admin/contacts/{ada.id}/messages", headers=admin_headers
    )

    assert response.json() == {"items": [], "total": 0}


async def test_memories_without_embeddings(client, session, admin_headers):
    ada = await make_contact(session)
    session.add(Memory(contact_id=ada.id, content="Prefers afternoons"))
    await session.flush()
    await session.commit()

    response = await client.get(
        f"/admin/contacts/{ada.id}/memories", headers=admin_headers
    )

    assert response.status_code == 200
    items = response.json()["items"]
    assert [m["content"] for m in items] == ["Prefers afternoons"]
    assert items[0]["has_embedding"] is False
    assert "embedding" not in items[0]  # never the vector
