"""Sprint 7 (ADR-004): conversation mode + the human-handoff inbox.

Mode short-circuit on ingest, the human_handoff tool flipping the mode,
take-over/resume endpoints, and operator messages through the (stubbed)
channel send seam.
"""

import uuid
from types import SimpleNamespace

import pytest
from sqlmodel import select

from app.models import AgentConfig, Contact, Conversation, Message
from app.modules.handoff import human_handoff
from app.services import channel_send, notify_service, orchestrator
from app.services.admin_service import create_admin_access_token
from app.services.agent_context import TurnContext

BSUID = "bsuid-HANDOFF-TEST"


@pytest.fixture
def admin_headers() -> dict:
    token = create_admin_access_token(uuid.uuid4(), "admin@test.local", "super_admin")
    return {"Authorization": f"Bearer {token}"}


def canonical_payload(text: str = "hola") -> dict:
    return {
        "channel": "whatsapp",
        "contact": {"external_id": BSUID, "wa_id": "51999000111", "metadata": {}},
        "message": {"type": "text", "text": text, "media_url": None, "raw": {}},
    }


async def make_contact(session, *, wa_id: str | None = "51999000111") -> Contact:
    contact = Contact(channel="whatsapp", external_id=BSUID, wa_id=wa_id)
    session.add(contact)
    await session.flush()
    return contact


async def make_conversation(session, contact: Contact, mode: str = "agent") -> Conversation:
    conversation = Conversation(contact_id=contact.id, mode=mode)
    session.add(conversation)
    await session.flush()
    return conversation


# --- ingest short-circuit --------------------------------------------------


async def test_human_mode_persists_inbound_and_runs_no_turn(
    client, session, monkeypatch
):
    contact = await make_contact(session)
    await make_conversation(session, contact, mode="human")
    await session.commit()

    async def must_not_run(*args, **kwargs):
        raise AssertionError("agent turn must not run in human mode")

    monkeypatch.setattr(orchestrator, "run_turn", must_not_run)

    resp = await client.post("/ingest", json=canonical_payload("estoy esperando"))

    assert resp.status_code == 200
    assert resp.json()["messages"] == []  # silence — gateways render nothing

    messages = (await session.exec(select(Message))).all()
    assert [m.direction for m in messages] == ["in"]
    assert messages[0].text == "estoy esperando"


async def test_agent_mode_still_runs_the_turn(client, session, monkeypatch):
    contact = await make_contact(session)
    await make_conversation(session, contact, mode="agent")
    await session.commit()

    async def fake_run_turn(session_, conversation, inbound, **kwargs):
        from app.schemas.ingest import OutboundMessage

        return [OutboundMessage(type="text", text="echo")]

    monkeypatch.setattr(orchestrator, "run_turn", fake_run_turn)

    resp = await client.post("/ingest", json=canonical_payload())

    assert resp.status_code == 200
    assert resp.json()["messages"][0]["text"] == "echo"


# --- the human_handoff tool ------------------------------------------------


async def test_human_handoff_flips_mode_and_dispatches_notification(
    session, monkeypatch
):
    contact = await make_contact(session)
    conversation = await make_conversation(session, contact)
    runtime = SimpleNamespace(
        context=TurnContext(
            session=session,
            contact_id=contact.id,
            conversation_id=conversation.id,
            config=AgentConfig(),
        )
    )

    dispatched: list[tuple[Contact, str]] = []
    monkeypatch.setattr(
        notify_service, "dispatch_handoff", lambda c, r: dispatched.append((c, r))
    )

    result = await human_handoff.coroutine(reason="asks for a person", runtime=runtime)

    assert "go silent" in result
    assert conversation.mode == "human"
    handoff = conversation.conversation_state["handoff"]
    assert handoff["requested"] is True
    assert handoff["reason"] == "asks for a person"
    assert dispatched and dispatched[0][0].id == contact.id


# --- take over / resume ----------------------------------------------------


async def test_take_over_and_resume(client, session, admin_headers):
    contact = await make_contact(session)
    conversation = await make_conversation(session, contact)
    conversation.conversation_state = {
        "handoff": {"requested": True, "reason": "upset", "at": "2026-06-10T00:00:00Z"}
    }
    conversation.mode = "human"
    session.add(conversation)
    await session.commit()

    resumed = await client.put(
        f"/admin/contacts/{contact.id}/mode",
        json={"mode": "agent"},
        headers=admin_headers,
    )
    assert resumed.status_code == 200
    assert resumed.json() == {"mode": "agent"}

    await session.refresh(conversation)
    assert conversation.mode == "agent"
    handoff = conversation.conversation_state["handoff"]
    assert handoff["requested"] is False
    assert handoff["reason"] == "upset"  # audit kept
    assert handoff["resolved_at"]

    taken = await client.put(
        f"/admin/contacts/{contact.id}/mode",
        json={"mode": "human"},
        headers=admin_headers,
    )
    assert taken.json() == {"mode": "human"}


async def test_take_over_creates_missing_conversation(client, session, admin_headers):
    contact = await make_contact(session)
    await session.commit()

    resp = await client.put(
        f"/admin/contacts/{contact.id}/mode",
        json={"mode": "human"},
        headers=admin_headers,
    )

    assert resp.status_code == 200
    conversation = (await session.exec(select(Conversation))).one()
    assert conversation.contact_id == contact.id
    assert conversation.mode == "human"


async def test_mode_requires_auth_and_valid_value(client, session, admin_headers):
    contact = await make_contact(session)
    await session.commit()

    assert (
        await client.put(f"/admin/contacts/{contact.id}/mode", json={"mode": "human"})
    ).status_code == 401
    assert (
        await client.put(
            f"/admin/contacts/{contact.id}/mode",
            json={"mode": "robot"},
            headers=admin_headers,
        )
    ).status_code == 422


# --- operator messages -----------------------------------------------------


async def test_operator_message_409_in_agent_mode(client, session, admin_headers):
    contact = await make_contact(session)
    await make_conversation(session, contact, mode="agent")
    await session.commit()

    resp = await client.post(
        f"/admin/contacts/{contact.id}/messages",
        json={"text": "hola"},
        headers=admin_headers,
    )
    assert resp.status_code == 409


async def test_operator_message_sends_then_persists(
    client, session, admin_headers, monkeypatch
):
    contact = await make_contact(session)
    await make_conversation(session, contact, mode="human")
    await session.commit()

    sent: list[dict] = []

    async def fake_send_message(contact_, **kwargs):
        sent.append({"contact": contact_, **kwargs})
        return {"status": "sent", "message_id": "wamid.X"}

    monkeypatch.setattr(channel_send, "send_message", fake_send_message)

    resp = await client.post(
        f"/admin/contacts/{contact.id}/messages",
        json={"text": "Hola, soy Ana del equipo"},
        headers=admin_headers,
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["direction"] == "out"
    assert body["text"] == "Hola, soy Ana del equipo"
    assert body["meta"]["sent_by_email"] == "admin@test.local"

    assert sent and sent[0]["text"] == "Hola, soy Ana del equipo"
    assert sent[0]["type"] == "text"
    message = (await session.exec(select(Message))).one()
    assert message.direction == "out"
    assert message.meta["sent_by"]  # the admin's id


async def test_operator_message_send_failure_persists_nothing(
    client, session, admin_headers, monkeypatch
):
    contact = await make_contact(session)
    await make_conversation(session, contact, mode="human")
    await session.commit()

    async def failing_send_message(contact_, **kwargs):
        raise channel_send.ChannelSendError("WINDOW_EXPIRED", "outside the 24h window")

    monkeypatch.setattr(channel_send, "send_message", failing_send_message)

    resp = await client.post(
        f"/admin/contacts/{contact.id}/messages",
        json={"text": "llego tarde"},
        headers=admin_headers,
    )

    assert resp.status_code == 502
    assert resp.json()["detail"]["code"] == "WINDOW_EXPIRED"
    assert (await session.exec(select(Message))).all() == []  # thread never lies


async def test_operator_media_message_sends_and_persists_key(
    client, session, admin_headers, monkeypatch
):
    import base64

    from app.core import storage

    contact = await make_contact(session)
    await make_conversation(session, contact, mode="human")
    await session.commit()

    sent: list[dict] = []

    async def fake_send_message(contact_, **kwargs):
        sent.append(kwargs)
        return {"status": "sent", "message_id": "wamid.M"}

    uploads: list[tuple[str, bytes, str]] = []

    async def fake_put_media(key, data, content_type):
        uploads.append((key, data, content_type))

    monkeypatch.setattr(channel_send, "send_message", fake_send_message)
    monkeypatch.setattr(storage, "is_configured", lambda: True)
    monkeypatch.setattr(storage, "put_media", fake_put_media)

    data_uri = "data:image/jpeg;base64," + base64.b64encode(b"fake-jpeg").decode()
    resp = await client.post(
        f"/admin/contacts/{contact.id}/messages",
        json={"type": "image", "text": "mira", "media_data_uri": data_uri},
        headers=admin_headers,
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["type"] == "image"
    assert body["has_media"] is True
    assert "media_url" not in body  # keys never serialized (ADR-003)

    assert sent[0]["type"] == "image"
    assert sent[0]["media_url"] == data_uri
    message = (await session.exec(select(Message))).one()
    assert message.media_url == f"media/{contact.id}/{message.id}.jpg"
    assert uploads and uploads[0][1] == b"fake-jpeg"


async def test_operator_media_message_validation(client, session, admin_headers):
    contact = await make_contact(session)
    await make_conversation(session, contact, mode="human")
    await session.commit()

    # media type without a data URI
    no_media = await client.post(
        f"/admin/contacts/{contact.id}/messages",
        json={"type": "image", "text": "sin adjunto"},
        headers=admin_headers,
    )
    assert no_media.status_code == 422

    # text type without text
    no_text = await client.post(
        f"/admin/contacts/{contact.id}/messages",
        json={"type": "text", "text": "  "},
        headers=admin_headers,
    )
    assert no_text.status_code == 422


# --- inbox listing ---------------------------------------------------------


async def test_list_exposes_mode_filter_and_attention_sort(
    client, session, admin_headers
):
    quiet = Contact(channel="whatsapp", external_id="bsuid-quiet")
    waiting = Contact(channel="whatsapp", external_id="bsuid-waiting")
    session.add(quiet)
    session.add(waiting)
    await session.flush()
    session.add(Conversation(contact_id=quiet.id, mode="agent"))
    conv = Conversation(
        contact_id=waiting.id,
        mode="human",
        conversation_state={
            "handoff": {"requested": True, "reason": "sales", "at": "2026-06-10T01:00:00Z"}
        },
    )
    session.add(conv)
    await session.flush()
    session.add(
        Message(conversation_id=conv.id, direction="in", type="text", text="ayuda")
    )
    # quiet is more recent — attention sort must still put `waiting` first
    quiet.updated_at = quiet.updated_at.replace(year=2030)
    session.add(quiet)
    await session.commit()

    resp = await client.get("/admin/contacts", headers=admin_headers)
    items = resp.json()["items"]
    assert [i["external_id"] for i in items] == ["bsuid-waiting", "bsuid-quiet"]
    top = items[0]
    assert top["mode"] == "human"
    assert top["handoff_reason"] == "sales"
    assert top["handoff_at"] == "2026-06-10T01:00:00Z"
    assert top["last_inbound_at"] is not None
    assert items[1]["mode"] == "agent"
    assert items[1]["last_inbound_at"] is None

    filtered = await client.get(
        "/admin/contacts", params={"mode": "human"}, headers=admin_headers
    )
    assert filtered.json()["total"] == 1
    assert filtered.json()["items"][0]["external_id"] == "bsuid-waiting"


async def test_detail_exposes_mode_and_window_anchor(client, session, admin_headers):
    contact = await make_contact(session)
    conversation = await make_conversation(session, contact, mode="human")
    session.add(
        Message(
            conversation_id=conversation.id, direction="in", type="text", text="hola"
        )
    )
    await session.commit()

    resp = await client.get(f"/admin/contacts/{contact.id}", headers=admin_headers)
    body = resp.json()
    assert body["mode"] == "human"
    assert body["last_inbound_at"] is not None


# --- async delivery status (gateway → core) ---------------------------------


async def test_channel_status_marks_message_failed(client, session):
    contact = await make_contact(session)
    conversation = await make_conversation(session, contact, mode="human")
    message = Message(
        conversation_id=conversation.id,
        direction="out",
        type="audio",
        meta={"sent_by": "x", "wamid": "wamid.FAIL1"},
    )
    session.add(message)
    await session.commit()

    resp = await client.post(
        "/channel/status",
        json={
            "message_id": "wamid.FAIL1",
            "status": "failed",
            "code": "131053",
            "detail": "Media upload error",
        },
    )

    assert resp.status_code == 204
    await session.refresh(message)
    assert message.meta["delivery_status"] == "failed"
    assert message.meta["delivery_code"] == "131053"
    assert message.meta["delivery_detail"] == "Media upload error"


async def test_channel_status_unknown_wamid_is_acknowledged(client, session):
    resp = await client.post(
        "/channel/status",
        json={"message_id": "wamid.UNKNOWN", "status": "failed"},
    )
    assert resp.status_code == 204
