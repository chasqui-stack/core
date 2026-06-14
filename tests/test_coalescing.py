"""core#6 Etapa 2 — inbound debounce + coalescing (ADR-008).

Two halves: the ingest SCHEDULE path (persist pending + arm the window, ack
empty) and the WORKER (claim due → one coalesced turn → dispatch via the send
seam). The agent turn and the channel send are stubbed — orchestration and the
HTTP send are covered elsewhere; here we assert the coalescing mechanics.
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import select

from app.core.config import settings
from app.models import Contact, Conversation, Message
from app.schemas.ingest import OutboundMessage
from app.services import channel_send, coalesce_worker, orchestrator

BSUID = "bsuid-COALESCE-0001"


def _naive_utc(offset_seconds: float = 0.0) -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(
        seconds=offset_seconds
    )


def canonical_payload(text: str = "Hola", **overrides) -> dict:
    payload = {
        "channel": "telegram",
        "contact": {"external_id": BSUID, "wa_id": None, "display_name": "Ana", "metadata": {}},
        "message": {"type": "text", "text": text, "media_url": None, "raw": {}},
        "received_at": "2026-06-14T10:00:00Z",
    }
    payload.update(overrides)
    return payload


async def _seed_conversation(session, *, mode: str = "agent") -> Conversation:
    contact = Contact(channel="telegram", external_id=BSUID, display_name="Ana")
    session.add(contact)
    await session.flush()
    conversation = Conversation(contact_id=contact.id, mode=mode)
    session.add(conversation)
    await session.flush()
    return conversation


def _add_pending_inbound(session, conversation, text, *, seconds: float) -> Message:
    message = Message(
        conversation_id=conversation.id,
        direction="in",
        type="text",
        text=text,
        created_at=_naive_utc(seconds),  # ordering within the burst
    )
    session.add(message)
    return message


# ---------------------------------------------------------------------------
# Ingest schedule path
# ---------------------------------------------------------------------------


async def test_ingest_schedules_instead_of_replying(client, session, monkeypatch):
    monkeypatch.setattr(settings, "inbound_debounce_seconds", 5)

    resp = await client.post("/ingest", json=canonical_payload("Hola"))
    assert resp.status_code == 200
    # Deferred: the reply does NOT come back in the response body.
    assert resp.json()["messages"] == []

    conversation = (await session.exec(select(Conversation))).one()
    assert conversation.debounce_due_at is not None  # window armed
    inbound = (await session.exec(select(Message))).one()
    assert inbound.direction == "in"
    assert inbound.processed_at is None  # pending — the worker will fold it in


async def test_rapid_messages_extend_the_same_window(client, session, monkeypatch):
    monkeypatch.setattr(settings, "inbound_debounce_seconds", 5)

    await client.post("/ingest", json=canonical_payload("Hola"))
    first_due = (await session.exec(select(Conversation))).one().debounce_due_at
    session.expire_all()
    await client.post("/ingest", json=canonical_payload("una consulta"))

    conv = (await session.exec(select(Conversation))).one()
    assert conv.debounce_due_at >= first_due  # pushed out, not a second window
    pending = (
        await session.exec(select(Message).where(Message.processed_at.is_(None)))
    ).all()
    assert len(pending) == 2  # both still pending in one batch


async def test_human_mode_under_debounce_schedules_nothing(client, session, monkeypatch):
    monkeypatch.setattr(settings, "inbound_debounce_seconds", 5)
    await _seed_conversation(session, mode="human")
    await session.commit()

    resp = await client.post("/ingest", json=canonical_payload("Hola"))
    assert resp.json()["messages"] == []

    conv = (await session.exec(select(Conversation))).one()
    assert conv.debounce_due_at is None  # a human owns the reply — never queued
    inbound = (await session.exec(select(Message))).one()
    assert inbound.processed_at is not None  # marked handled, not pending


async def test_synchronous_path_when_disabled(client, session, monkeypatch):
    # debounce defaults to 0 in tests (conftest) → legacy synchronous reply.
    async def fake_run_turn(session, conversation, inbound, **kwargs):
        return [OutboundMessage(type="text", text=f"Echo: {inbound.text}")]

    monkeypatch.setattr(orchestrator, "run_turn", fake_run_turn)
    resp = await client.post("/ingest", json=canonical_payload("Hola"))

    assert resp.json()["messages"][0]["text"] == "Echo: Hola"
    conv = (await session.exec(select(Conversation))).one()
    assert conv.debounce_due_at is None  # synchronous path never arms a window


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


async def test_worker_coalesces_burst_into_one_turn(
    session, session_factory, monkeypatch
):
    conv = await _seed_conversation(session)
    _add_pending_inbound(session, conv, "Hola", seconds=0)
    _add_pending_inbound(session, conv, "una consulta", seconds=1)
    _add_pending_inbound(session, conv, "sobre mi pedido", seconds=2)
    conv.debounce_due_at = _naive_utc(-1)  # window already elapsed
    await session.commit()

    seen_batches: list[list[str]] = []

    async def fake_coalesced(session, conversation, batch, **kwargs):
        seen_batches.append([m.text for m in batch])
        return [OutboundMessage(type="text", text="Una sola respuesta")]

    sent: list[str] = []

    async def fake_send(contact, **kwargs):
        sent.append(kwargs.get("text"))
        return {"ok": True}

    monkeypatch.setattr(orchestrator, "run_coalesced_turn", fake_coalesced)
    monkeypatch.setattr(channel_send, "send_message", fake_send)

    owned = await coalesce_worker.process_conversation(session_factory, conv.id)
    assert owned is True

    # Exactly ONE turn, over the ordered batch.
    assert seen_batches == [["Hola", "una consulta", "sobre mi pedido"]]
    # One reply dispatched via the send seam (not the ingest body).
    assert sent == ["Una sola respuesta"]

    session.expire_all()
    refreshed = (await session.exec(select(Conversation))).one()
    assert refreshed.debounce_due_at is None  # window cleared
    inbound = (
        await session.exec(select(Message).where(Message.direction == "in"))
    ).all()
    assert all(m.processed_at is not None for m in inbound)  # batch consumed
    outbound = (
        await session.exec(select(Message).where(Message.direction == "out"))
    ).all()
    assert len(outbound) == 1 and outbound[0].text == "Una sola respuesta"


async def test_worker_only_marks_the_gathered_batch(
    session, session_factory, monkeypatch
):
    """A message arriving after the batch was gathered stays pending for the
    NEXT window — no double-processing."""
    conv = await _seed_conversation(session)
    _add_pending_inbound(session, conv, "A", seconds=0)
    _add_pending_inbound(session, conv, "B", seconds=1)
    conv.debounce_due_at = _naive_utc(-1)
    await session.commit()

    async def fake_coalesced(session, conversation, batch, **kwargs):
        return [OutboundMessage(type="text", text="ok")]

    monkeypatch.setattr(orchestrator, "run_coalesced_turn", fake_coalesced)
    monkeypatch.setattr(channel_send, "send_message", lambda *a, **k: _noop())

    await coalesce_worker.process_conversation(session_factory, conv.id)

    # A late straggler arrives + re-arms the window.
    late = _add_pending_inbound(session, conv, "C", seconds=5)
    conv.debounce_due_at = _naive_utc(-1)
    await session.commit()
    session.expire_all()

    pending = (
        await session.exec(
            select(Message).where(
                Message.direction == "in", Message.processed_at.is_(None)
            )
        )
    ).all()
    assert [m.text for m in pending] == ["C"]  # only the straggler awaits a turn


async def test_dispatch_failure_does_not_unwind_the_turn(
    session, session_factory, monkeypatch
):
    conv = await _seed_conversation(session)
    _add_pending_inbound(session, conv, "Hola", seconds=0)
    conv.debounce_due_at = _naive_utc(-1)
    await session.commit()

    async def fake_coalesced(session, conversation, batch, **kwargs):
        return [OutboundMessage(type="text", text="reply")]

    async def boom(contact, **kwargs):
        raise channel_send.ChannelSendError("GATEWAY_UNREACHABLE", "down")

    monkeypatch.setattr(orchestrator, "run_coalesced_turn", fake_coalesced)
    monkeypatch.setattr(channel_send, "send_message", boom)

    owned = await coalesce_worker.process_conversation(session_factory, conv.id)
    assert owned is True  # best-effort dispatch failure is swallowed

    session.expire_all()
    # Turn is committed despite the send failure: batch marked, reply persisted.
    inbound = (
        await session.exec(select(Message).where(Message.direction == "in"))
    ).one()
    assert inbound.processed_at is not None
    outbound = (
        await session.exec(select(Message).where(Message.direction == "out"))
    ).all()
    assert len(outbound) == 1


async def test_run_once_skips_not_yet_due_windows(session, session_factory, monkeypatch):
    conv = await _seed_conversation(session)
    _add_pending_inbound(session, conv, "todavía no", seconds=0)
    conv.debounce_due_at = _naive_utc(60)  # window 60s in the FUTURE
    await session.commit()

    called = False

    async def fake_coalesced(session, conversation, batch, **kwargs):
        nonlocal called
        called = True
        return [OutboundMessage(type="text", text="x")]

    monkeypatch.setattr(orchestrator, "run_coalesced_turn", fake_coalesced)
    monkeypatch.setattr(channel_send, "send_message", lambda *a, **k: _noop())

    processed = await coalesce_worker.run_once(session_factory)
    assert processed == 0 and called is False  # not due yet


# ---------------------------------------------------------------------------
# History exclusion (orchestrator seam used by the coalesced turn)
# ---------------------------------------------------------------------------


async def test_history_excludes_pending_inbound(session):
    conv = await _seed_conversation(session)
    processed = Message(
        conversation_id=conv.id, direction="in", type="text", text="vieja",
        created_at=_naive_utc(0), processed_at=_naive_utc(0),
    )
    out = Message(
        conversation_id=conv.id, direction="out", type="text", text="respondida",
        created_at=_naive_utc(1),
    )
    pending = Message(
        conversation_id=conv.id, direction="in", type="text", text="ACTUAL",
        created_at=_naive_utc(2),
    )
    session.add_all([processed, out, pending])
    await session.commit()

    history = await orchestrator._history_messages(
        session, conv.id, 50, exclude_pending=True
    )
    texts = [m.content for m in history]
    assert "vieja" in texts and "respondida" in texts
    assert "ACTUAL" not in texts  # the pending batch is the input, not history


async def _noop():
    return {"ok": True}
