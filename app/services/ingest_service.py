"""Canonical ingest pipeline (ARCHITECTURE §5, §6).

upsert contact → get-or-create the single conversation → reply.

Two reply modes, gated by INBOUND_DEBOUNCE_SECONDS (ADR-008):

- **Synchronous (= 0):** run the agent turn inline and return the reply in the
  /ingest response. The turn runs BEFORE the inbound row is persisted on
  purpose — the orchestrator's history query must see only *prior* messages
  (the current one travels as the turn's input). One transaction.

- **Deferred / coalesced (> 0, the default):** persist the inbound as *pending*
  (processed_at NULL), arm the conversation's debounce window, and ack with an
  empty reply. The coalesce worker (app/services/coalesce_worker.py) later folds
  the whole burst into ONE turn and dispatches via the channel send seam.

Both paths first take the per-identity advisory lock (Etapa 1) so a rapid burst
— e.g. a gateway webhook-retry storm — can't race on contact/conversation
creation.
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import text
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core import storage
from app.core.config import settings
from app.models import Contact, Conversation, Message
from app.schemas.ingest import ContactPayload, IngestRequest, IngestResponse
from app.services import memory_service, orchestrator

logger = logging.getLogger(__name__)


async def acquire_turn_lock(session: AsyncSession, channel: str, external_id: str) -> None:
    """Serialize concurrent turns for one identity (Etapa 1, core#6).

    Transaction-scoped advisory lock: auto-released on commit/rollback, held
    only for the txn this session already spans, so it adds no connection beyond
    today's. Shared by the ingest pipeline and the coalesce worker.
    """
    await session.execute(
        text("SELECT pg_advisory_xact_lock(hashtext(:key))"),
        {"key": f"{channel}:{external_id}"},
    )


def _to_naive_utc(dt: datetime | None) -> datetime:
    """Normalize to naive UTC (asyncpg + TIMESTAMP WITHOUT TIME ZONE)."""
    if dt is None:
        return datetime.now(timezone.utc).replace(tzinfo=None)
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


async def find_contact(
    session: AsyncSession, channel: str, external_id: str
) -> Contact | None:
    """Read-only lookup by (channel, external_id) — no create.

    The mirror of `upsert_contact`'s find step, for gateway-facing reads (the
    internal history endpoint, ADR-011) that must not create a contact.
    """
    result = await session.exec(
        select(Contact).where(
            Contact.channel == channel,
            Contact.external_id == external_id,
        )
    )
    return result.first()


async def upsert_contact(
    session: AsyncSession, channel: str, payload: ContactPayload
) -> Contact:
    """Find-or-create by (channel, external_id); refresh mutable fields on hit.

    external_id is the identity (BSUID on WhatsApp); wa_id/display_name/metadata
    are informational and updated when the gateway sends fresher values.
    """
    result = await session.exec(
        select(Contact).where(
            Contact.channel == channel,
            Contact.external_id == payload.external_id,
        )
    )
    contact = result.first()

    if contact is None:
        contact = Contact(
            channel=channel,
            external_id=payload.external_id,
            wa_id=payload.wa_id,
            display_name=payload.display_name,
            meta=payload.metadata,
        )
        session.add(contact)
        await session.flush()
        return contact

    changed = False
    if payload.wa_id and payload.wa_id != contact.wa_id:
        contact.wa_id = payload.wa_id
        changed = True
    if payload.display_name and payload.display_name != contact.display_name:
        contact.display_name = payload.display_name
        changed = True
    if payload.metadata:
        contact.meta = {**contact.meta, **payload.metadata}
        changed = True
    if changed:
        contact.updated_at = _to_naive_utc(None)
        session.add(contact)

    return contact


async def get_or_create_conversation(
    session: AsyncSession, contact_id: uuid.UUID
) -> Conversation:
    """The single thread per contact (unique contact_id enforces it)."""
    result = await session.exec(
        select(Conversation).where(Conversation.contact_id == contact_id)
    )
    conversation = result.first()

    if conversation is None:
        conversation = Conversation(contact_id=contact_id)
        session.add(conversation)
        await session.flush()

    return conversation


async def _store_inbound_media(
    contact_id: uuid.UUID, message_id: uuid.UUID, data_uri: str
) -> str | None:
    """Upload an inline `data:` URI to the bucket; return the object key.

    ADR-003: failures NEVER break the turn (the embeddings pattern) — any
    problem logs and returns None, leaving media_url NULL like before the
    storage layer existed.
    """
    try:
        return await storage.put_data_uri(contact_id, message_id, data_uri)
    except Exception:
        logger.exception("Media upload failed (message %s) — persisting NULL", message_id)
        return None


async def _persist_inbound(
    session: AsyncSession, contact: Contact, conversation: Conversation,
    request: IngestRequest,
) -> Message:
    """Persist the inbound message (timestamped by the gateway when provided).

    Inline media (base64 data URIs) feeds the current turn; with storage
    configured (ADR-003) it is also uploaded and the OBJECT KEY persisted
    in media_url — otherwise history stays text-only as before (the
    media_id in `raw` allows re-download if needed).
    """
    inbound = Message(
        conversation_id=conversation.id,
        direction="in",
        type=request.message.type,
        text=request.message.text,
        media_url=None,
        meta=request.message.raw,
        created_at=_to_naive_utc(request.received_at),
    )
    media_url = request.message.media_url
    if media_url and media_url.startswith("data:"):
        if storage.is_configured():
            inbound.media_url = await _store_inbound_media(
                contact.id, inbound.id, media_url
            )
    else:
        inbound.media_url = media_url
    session.add(inbound)
    return inbound


async def handle_ingest(session: AsyncSession, request: IngestRequest) -> IngestResponse:
    """Accept one canonical inbound; reply synchronously or defer to the worker."""
    await acquire_turn_lock(session, request.channel, request.contact.external_id)

    contact = await upsert_contact(session, request.channel, request.contact)
    conversation = await get_or_create_conversation(session, contact.id)

    if settings.inbound_debounce_seconds > 0:
        return await _schedule_coalesced(session, contact, conversation, request)
    return await _run_synchronous(session, contact, conversation, request)


async def _schedule_coalesced(
    session: AsyncSession, contact: Contact, conversation: Conversation,
    request: IngestRequest,
) -> IngestResponse:
    """Deferred path (ADR-008): persist as pending, arm the window, ack empty.

    No turn runs here. Human mode (ADR-004) persists the inbound marked
    processed (a human owns the reply) and arms NOTHING — the coalesce worker
    never sees human-owned threads.
    """
    inbound = await _persist_inbound(session, contact, conversation, request)
    if conversation.mode == "human":
        inbound.processed_at = _to_naive_utc(None)  # handled by a human, not pending
    else:
        # (Re)arm the silence window — every fresh message pushes it out, so the
        # turn fires only after the burst settles. Pending (processed_at NULL).
        conversation.debounce_due_at = _to_naive_utc(None) + timedelta(
            seconds=settings.inbound_debounce_seconds
        )
    conversation.updated_at = _to_naive_utc(None)
    session.add(conversation)
    await session.flush()
    return IngestResponse(messages=[], conversation_id=conversation.id)


async def _run_synchronous(
    session: AsyncSession, contact: Contact, conversation: Conversation,
    request: IngestRequest,
) -> IngestResponse:
    """Synchronous path (INBOUND_DEBOUNCE_SECONDS=0): run the turn inline."""
    # Human mode (ADR-004) — checked FIRST: a human owns this thread, so the
    # inbound is persisted but NO agent turn runs. The empty reply list is
    # silence on every channel (gateways render 0..N messages), which is how
    # channels inherit human mode with zero changes.
    if conversation.mode == "human":
        inbound = await _persist_inbound(session, contact, conversation, request)
        inbound.processed_at = _to_naive_utc(None)
        conversation.updated_at = _to_naive_utc(None)
        session.add(conversation)
        await session.flush()
        return IngestResponse(messages=[], conversation_id=conversation.id)

    # Agent turn (LangGraph — history query must not see the current message yet)
    replies = await orchestrator.run_turn(session, conversation, request.message)

    inbound = await _persist_inbound(session, contact, conversation, request)
    inbound.processed_at = _to_naive_utc(None)  # handled inline, never pending

    # Persist outbound message(s)
    for reply in replies:
        session.add(
            Message(
                conversation_id=conversation.id,
                direction="out",
                type=reply.type,
                text=reply.text,
                media_url=reply.media_url,
                meta=reply.metadata,
            )
        )

    conversation.updated_at = _to_naive_utc(None)
    session.add(conversation)

    # Memory extraction seam — no-op until Sprint 3/4
    await memory_service.extract_after_turn(session, contact.id, conversation.id)

    await session.flush()
    return IngestResponse(messages=replies, conversation_id=conversation.id)
