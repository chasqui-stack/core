"""Conversation inspection + the human-handoff inbox (Sprint 5 / Sprint 7).

Contact-centric: the schema enforces one conversation per contact
(`conversations.contact_id` UNIQUE), so the admin never handles a
conversation id — messages and memories hang off the contact. Sprint 7
(ADR-004) added the two writes the inbox needs: flipping the conversation
mode and sending an operator message through the channel's `/send`.
"""

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import case, func
from sqlmodel import col, or_, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core import storage
from app.core.dependencies import CurrentAdmin
from app.db.session import get_session
from app.models import Contact, Conversation, Memory, Message
from app.schemas.admin_contacts import (
    ContactDetail,
    ContactListItem,
    ContactListResponse,
    LastMessagePreview,
    MemoryItem,
    MemoryListResponse,
    MessageItem,
    MessageListResponse,
    ModeResponse,
    ModeUpdateRequest,
    OperatorMessageCreate,
)
from app.services import channel_send
from app.services.ingest_service import get_or_create_conversation

logger = logging.getLogger(__name__)

router = APIRouter()


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _handoff_fields(conversation: Conversation | None) -> tuple[str | None, str | None]:
    """(reason, at) from the audit JSONB — None when never handed off."""
    if conversation is None:
        return None, None
    handoff = conversation.conversation_state.get("handoff") or {}
    return handoff.get("reason"), handoff.get("at")


# Meta's media size limits (decoded bytes) — validated before bothering the
# gateway; base64 inflates ~4/3, so the JSON body stays bounded too.
_MEDIA_MAX_BYTES = {"image": 5 * 1024 * 1024, "audio": 16 * 1024 * 1024,
                    "document": 25 * 1024 * 1024}


def _validate_operator_message(payload: OperatorMessageCreate) -> None:
    if payload.type == "text":
        if not (payload.text or "").strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Text messages need a non-empty text",
            )
        return
    if not payload.media_data_uri or not payload.media_data_uri.startswith("data:"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{payload.type} messages need media_data_uri as a base64 data: URI",
        )
    # 3/4 of the base64 length ≈ decoded size (cheap, no decode)
    approx_bytes = (len(payload.media_data_uri) * 3) // 4
    if approx_bytes > _MEDIA_MAX_BYTES[payload.type]:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"{payload.type} exceeds the "
            f"{_MEDIA_MAX_BYTES[payload.type] // (1024 * 1024)}MB channel limit",
        )


async def _last_inbound_by_conv(
    session: AsyncSession, conv_ids: list[uuid.UUID]
) -> dict[uuid.UUID, datetime]:
    """Newest inbound timestamp per conversation — the 24h-window anchor."""
    if not conv_ids:
        return {}
    rows = await session.exec(
        select(Message.conversation_id, func.max(Message.created_at))
        .where(
            col(Message.conversation_id).in_(conv_ids),
            Message.direction == "in",
        )
        .group_by(col(Message.conversation_id))
    )
    return dict(rows.all())


async def _get_contact_or_404(session: AsyncSession, contact_id: uuid.UUID) -> Contact:
    contact = await session.get(Contact, contact_id)
    if contact is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Contact not found"
        )
    return contact


def _search_filter(search: str):
    pattern = f"%{search}%"
    return or_(
        col(Contact.display_name).ilike(pattern),
        col(Contact.external_id).ilike(pattern),
        col(Contact.wa_id).ilike(pattern),
    )


@router.get("", response_model=ContactListResponse)
async def list_contacts(
    session: AsyncSession = Depends(get_session),
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    search: str | None = Query(default=None),
    mode: str | None = Query(default=None, pattern="^(agent|human)$"),
):
    # Outer join: a contact may not have a conversation yet (mode = "agent")
    base = select(Contact, Conversation).join(
        Conversation, col(Conversation.contact_id) == col(Contact.id), isouter=True
    )
    count = select(func.count()).select_from(Contact).join(
        Conversation, col(Conversation.contact_id) == col(Contact.id), isouter=True
    )
    if search:
        base = base.where(_search_filter(search))
        count = count.where(_search_filter(search))
    if mode:
        base = base.where(Conversation.mode == mode)
        count = count.where(Conversation.mode == mode)

    total = (await session.exec(count)).one()
    # Attention-first: human-mode conversations on top, then recency
    needs_human = case((Conversation.mode == "human", 0), else_=1)
    rows = (
        await session.exec(
            base.order_by(needs_human, col(Contact.updated_at).desc())
            .limit(limit)
            .offset(offset)
        )
    ).all()
    contacts = [contact for contact, _ in rows]
    conv_by_contact = {
        contact.id: conversation for contact, conversation in rows if conversation
    }

    # Message counts + last message + last inbound (three queries, no N+1)
    counts: dict[uuid.UUID, int] = {}
    last_by_conv: dict[uuid.UUID, Message] = {}
    conv_ids = [c.id for c in conv_by_contact.values()]
    if conv_ids:
        count_rows = await session.exec(
            select(Message.conversation_id, func.count())
            .where(col(Message.conversation_id).in_(conv_ids))
            .group_by(col(Message.conversation_id))
        )
        counts = dict(count_rows.all())

        # Postgres DISTINCT ON: newest message per conversation
        last_rows = await session.exec(
            select(Message)
            .where(col(Message.conversation_id).in_(conv_ids))
            .order_by(col(Message.conversation_id), col(Message.created_at).desc())
            .distinct(col(Message.conversation_id))
        )
        last_by_conv = {m.conversation_id: m for m in last_rows.all()}
    last_inbound = await _last_inbound_by_conv(session, conv_ids)

    items = []
    for contact in contacts:
        conversation = conv_by_contact.get(contact.id)
        conv_id = conversation.id if conversation else None
        last = last_by_conv.get(conv_id) if conv_id else None
        handoff_reason, handoff_at = _handoff_fields(conversation)
        items.append(
            ContactListItem(
                id=contact.id,
                channel=contact.channel,
                external_id=contact.external_id,
                wa_id=contact.wa_id,
                display_name=contact.display_name,
                created_at=contact.created_at,
                updated_at=contact.updated_at,
                message_count=counts.get(conv_id, 0) if conv_id else 0,
                last_message=(
                    LastMessagePreview(
                        direction=last.direction,
                        type=last.type,
                        text=last.text,
                        created_at=last.created_at,
                    )
                    if last
                    else None
                ),
                mode=conversation.mode if conversation else "agent",
                handoff_reason=handoff_reason,
                handoff_at=handoff_at,
                last_inbound_at=last_inbound.get(conv_id) if conv_id else None,
            )
        )

    return ContactListResponse(items=items, total=total)


@router.get("/{contact_id}", response_model=ContactDetail)
async def get_contact(
    contact_id: uuid.UUID, session: AsyncSession = Depends(get_session)
):
    contact = await _get_contact_or_404(session, contact_id)
    conversation = (
        await session.exec(
            select(Conversation).where(Conversation.contact_id == contact_id)
        )
    ).first()
    handoff_reason, handoff_at = _handoff_fields(conversation)
    last_inbound = (
        await _last_inbound_by_conv(session, [conversation.id]) if conversation else {}
    )
    return ContactDetail(
        id=contact.id,
        channel=contact.channel,
        external_id=contact.external_id,
        wa_id=contact.wa_id,
        display_name=contact.display_name,
        meta=contact.meta,
        created_at=contact.created_at,
        updated_at=contact.updated_at,
        mode=conversation.mode if conversation else "agent",
        handoff_reason=handoff_reason,
        handoff_at=handoff_at,
        last_inbound_at=last_inbound.get(conversation.id) if conversation else None,
    )


@router.put("/{contact_id}/mode", response_model=ModeResponse)
async def set_mode(
    contact_id: uuid.UUID,
    payload: ModeUpdateRequest,
    session: AsyncSession = Depends(get_session),
):
    """Take over (human) / resume the bot (agent) — ADR-004.

    Get-or-create: taking over a contact whose conversation doesn't exist
    yet must not 500. Resuming keeps the handoff audit but marks it
    resolved — mode is the single source of truth, the JSONB is history.
    """
    await _get_contact_or_404(session, contact_id)
    conversation = await get_or_create_conversation(session, contact_id)
    conversation.mode = payload.mode
    if payload.mode == "agent" and "handoff" in conversation.conversation_state:
        handoff = {
            **conversation.conversation_state["handoff"],
            "requested": False,
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        }
        # Reassign (don't mutate) so SQLAlchemy detects the JSONB change
        conversation.conversation_state = {
            **conversation.conversation_state,
            "handoff": handoff,
        }
    conversation.updated_at = _utcnow_naive()
    session.add(conversation)
    await session.commit()
    return ModeResponse(mode=conversation.mode)


@router.post(
    "/{contact_id}/messages",
    response_model=MessageItem,
    status_code=status.HTTP_201_CREATED,
)
async def send_operator_message(
    contact_id: uuid.UUID,
    payload: OperatorMessageCreate,
    admin: CurrentAdmin,
    session: AsyncSession = Depends(get_session),
):
    """Operator reply (text or media), pushed through the channel's `/send`.

    Only in human mode (409 otherwise — the agent owns agent-mode replies).
    Send-then-persist: a failed send persists nothing, so the thread never
    shows a message the user didn't get; the gateway's error code
    (WINDOW_EXPIRED, NO_WA_ID, ...) flows through for the panel to explain.
    Media travels as a base64 data URI (mirror of inbound, ADR-004) and is
    also uploaded to the bucket so the timeline can render it (ADR-003 —
    upload failures log + NULL, never undo a delivered message).
    """
    _validate_operator_message(payload)
    contact = await _get_contact_or_404(session, contact_id)
    conversation = (
        await session.exec(
            select(Conversation).where(Conversation.contact_id == contact_id)
        )
    ).first()
    if conversation is None or conversation.mode != "human":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Conversation is in agent mode — take it over first",
        )

    try:
        sent = await channel_send.send_message(
            contact,
            type=payload.type,
            text=payload.text,
            media_url=payload.media_data_uri,
            filename=payload.filename,
        )
    except channel_send.ChannelSendError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"code": exc.code, "message": exc.message},
        )

    message = Message(
        conversation_id=conversation.id,
        direction="out",
        type=payload.type,
        text=payload.text,
        # wamid correlates async delivery statuses (POST /channel/status)
        meta={
            "sent_by": str(admin.admin_id),
            "sent_by_email": admin.email,
            "wamid": sent.get("message_id"),
        },
    )
    if payload.filename:
        message.meta = {**message.meta, "filename": payload.filename}
    if payload.media_data_uri and storage.is_configured():
        try:
            message.media_url = await storage.put_data_uri(
                contact.id, message.id, payload.media_data_uri
            )
        except Exception:
            logger.exception(
                "Outbound media upload failed (message %s) — persisting NULL",
                message.id,
            )
    session.add(message)
    conversation.updated_at = _utcnow_naive()
    session.add(conversation)
    await session.commit()

    return MessageItem(
        id=message.id,
        direction=message.direction,
        type=message.type,
        text=message.text,
        meta=message.meta,
        created_at=message.created_at,
        has_media=storage.is_media_key(message.media_url),
    )


@router.get("/{contact_id}/messages", response_model=MessageListResponse)
async def list_messages(
    contact_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """Newest first (the UI renders pages bottom-up like a chat)."""
    await _get_contact_or_404(session, contact_id)

    conversation = (
        await session.exec(
            select(Conversation).where(Conversation.contact_id == contact_id)
        )
    ).first()
    if conversation is None:
        return MessageListResponse(items=[], total=0)

    total = (
        await session.exec(
            select(func.count())
            .select_from(Message)
            .where(Message.conversation_id == conversation.id)
        )
    ).one()

    # Uses ix_messages_conversation_created
    messages = (
        await session.exec(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(col(Message.created_at).desc())
            .limit(limit)
            .offset(offset)
        )
    ).all()

    items = [
        MessageItem(
            id=m.id,
            direction=m.direction,
            type=m.type,
            text=m.text,
            meta=m.meta,
            created_at=m.created_at,
            has_media=storage.is_media_key(m.media_url),
        )
        for m in messages
    ]
    return MessageListResponse(items=items, total=total)


@router.get("/{contact_id}/memories", response_model=MemoryListResponse)
async def list_memories(
    contact_id: uuid.UUID, session: AsyncSession = Depends(get_session)
):
    await _get_contact_or_404(session, contact_id)
    memories = (
        await session.exec(
            select(Memory)
            .where(Memory.contact_id == contact_id)
            .order_by(col(Memory.created_at).desc())
        )
    ).all()
    items = [
        MemoryItem(
            id=m.id,
            content=m.content,
            has_embedding=m.embedding is not None,
            created_at=m.created_at,
        )
        for m in memories
    ]
    return MemoryListResponse(items=items)
