"""Inbound coalesce worker (ADR-008) — the deferred half of core#6 Etapa 2.

When INBOUND_DEBOUNCE_SECONDS > 0, /ingest stops replying inline: it persists
the inbound as pending and arms `conversations.debounce_due_at`. This worker
claims conversations whose silence window has elapsed, folds the whole pending
burst into ONE agent turn, persists the reply, and dispatches it through the
canonical send seam (ADR-004) — the same `/send` the human-handoff outbound
already uses, so gateways need no new contract.

Concurrency model:
- One claim+turn per conversation runs in a SINGLE transaction that locks the
  conversation row with `FOR UPDATE SKIP LOCKED`. Holding the lock across the
  turn keeps it simple and correct under N replicas (others skip the locked
  row); at Chasqui's scale a per-conversation row lock for a few seconds is
  fine. The per-identity advisory lock (Etapa 1) is also taken, serializing
  against a late straggler hitting /ingest mid-turn.
- Dispatch happens AFTER commit, best-effort (the notify_service posture): a
  gateway failure is logged, never rolls back the already-persisted turn.

The session factory is injectable so tests can bind the worker to their
transactional-rollback connection; production uses the real pooled factory.
"""

import asyncio
import logging
from datetime import datetime, timezone

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import settings
from app.db.session import async_session_factory
from app.models import Contact, Conversation, Message
from app.services import channel_send, memory_service, orchestrator
from app.services.ingest_service import acquire_turn_lock

logger = logging.getLogger(__name__)

# How many due conversations to look at per tick (a soft cap, not a hard limit:
# the next tick picks up the rest).
CLAIM_BATCH = 20


def _now() -> datetime:
    """Naive UTC — matches how debounce_due_at is written by the ingest path."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


async def _due_conversation_ids(session: AsyncSession, limit: int) -> list:
    """Conversations whose debounce window has elapsed (cheap read, no lock)."""
    result = await session.exec(
        select(Conversation.id)
        .where(
            Conversation.debounce_due_at.is_not(None),
            Conversation.debounce_due_at <= _now(),
        )
        .order_by(Conversation.debounce_due_at)
        .limit(limit)
    )
    return list(result.all())


async def process_conversation(session_factory, conversation_id) -> bool:
    """Run one coalesced turn for a conversation, then dispatch. Returns True
    if this worker owned and processed the conversation."""
    dispatch: list[tuple[Contact, object]] = []
    async with session_factory() as session:
        try:
            # Claim the row — SKIP LOCKED yields nothing if another worker holds
            # it, or if its window was cleared/re-armed since discovery.
            conv = (
                await session.exec(
                    select(Conversation)
                    .where(
                        Conversation.id == conversation_id,
                        Conversation.debounce_due_at.is_not(None),
                        Conversation.debounce_due_at <= _now(),
                    )
                    .with_for_update(skip_locked=True)
                )
            ).first()
            if conv is None:
                await session.rollback()
                return False

            contact = (
                await session.exec(
                    select(Contact).where(Contact.id == conv.contact_id)
                )
            ).first()
            # Serialize against a straggler /ingest for the same identity (Etapa 1)
            await acquire_turn_lock(session, contact.channel, contact.external_id)

            batch = list(
                (
                    await session.exec(
                        select(Message)
                        .where(
                            Message.conversation_id == conv.id,
                            Message.direction == "in",
                            Message.processed_at.is_(None),
                        )
                        .order_by(Message.created_at)
                    )
                ).all()
            )

            now = _now()
            if batch:
                replies = await orchestrator.run_coalesced_turn(session, conv, batch)
                for reply in replies:
                    session.add(
                        Message(
                            conversation_id=conv.id,
                            direction="out",
                            type=reply.type,
                            text=reply.text,
                            media_url=reply.media_url,
                            meta=reply.metadata,
                        )
                    )
                for message in batch:
                    message.processed_at = now
                    session.add(message)
                await memory_service.extract_after_turn(
                    session, conv.contact_id, conv.id
                )
                dispatch = [(contact, reply) for reply in replies]

            conv.debounce_due_at = None  # disarm; a new inbound re-arms it
            conv.updated_at = now
            session.add(conv)
            await session.commit()
        except Exception:
            await session.rollback()
            logger.exception(
                "Coalesce processing failed for conversation %s", conversation_id
            )
            return False

    # Post-commit dispatch — best-effort, never unwinds the persisted turn.
    for contact, reply in dispatch:
        try:
            await channel_send.send_message(
                contact, type=reply.type, text=reply.text, media_url=reply.media_url
            )
        except channel_send.ChannelSendError as exc:
            logger.error(
                "Deferred dispatch failed for %s:%s (%s): %s",
                contact.channel,
                contact.external_id,
                exc.code,
                exc.message,
            )
    return True


async def run_once(session_factory=async_session_factory) -> int:
    """One worker tick: process every currently-due conversation. Returns count."""
    async with session_factory() as session:
        due_ids = await _due_conversation_ids(session, CLAIM_BATCH)
    processed = 0
    for conversation_id in due_ids:
        if await process_conversation(session_factory, conversation_id):
            processed += 1
    return processed


async def run_loop(stop: asyncio.Event, session_factory=async_session_factory) -> None:
    """Poll for due conversations until `stop` is set (LISTEN/NOTIFY is a
    follow-up). Drains greedily when work is found; sleeps otherwise."""
    poll = settings.inbound_debounce_poll_seconds
    logger.info(
        "Coalesce worker started (window=%ss, poll=%ss)",
        settings.inbound_debounce_seconds,
        poll,
    )
    while not stop.is_set():
        try:
            count = await run_once(session_factory)
        except Exception:
            logger.exception("Coalesce worker tick failed")
            count = 0
        if count == 0:
            try:  # interruptible sleep — shutdown wakes us immediately
                await asyncio.wait_for(stop.wait(), timeout=poll)
            except asyncio.TimeoutError:
                pass
    logger.info("Coalesce worker stopped")
