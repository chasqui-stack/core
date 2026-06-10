"""Read-only conversation inspection (Sprint 5).

Contact-centric: the schema enforces one conversation per contact
(`conversations.contact_id` UNIQUE), so the admin never handles a
conversation id — messages and memories hang off the contact. Strictly
read-only: the panel observes, the agent acts.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlmodel import col, or_, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core import storage
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
)

router = APIRouter()


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
):
    base = select(Contact)
    count = select(func.count()).select_from(Contact)
    if search:
        base = base.where(_search_filter(search))
        count = count.where(_search_filter(search))

    total = (await session.exec(count)).one()
    contacts = (
        await session.exec(
            base.order_by(col(Contact.updated_at).desc()).limit(limit).offset(offset)
        )
    ).all()

    # Conversation per contact (one query for the page)
    contact_ids = [c.id for c in contacts]
    conv_by_contact: dict[uuid.UUID, uuid.UUID] = {}
    if contact_ids:
        rows = await session.exec(
            select(Conversation.contact_id, Conversation.id).where(
                col(Conversation.contact_id).in_(contact_ids)
            )
        )
        conv_by_contact = dict(rows.all())

    # Message counts + last message per conversation (two queries, no N+1)
    counts: dict[uuid.UUID, int] = {}
    last_by_conv: dict[uuid.UUID, Message] = {}
    conv_ids = list(conv_by_contact.values())
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

    items = []
    for contact in contacts:
        conv_id = conv_by_contact.get(contact.id)
        last = last_by_conv.get(conv_id) if conv_id else None
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
            )
        )

    return ContactListResponse(items=items, total=total)


@router.get("/{contact_id}", response_model=ContactDetail)
async def get_contact(
    contact_id: uuid.UUID, session: AsyncSession = Depends(get_session)
):
    contact = await _get_contact_or_404(session, contact_id)
    return ContactDetail(
        id=contact.id,
        channel=contact.channel,
        external_id=contact.external_id,
        wa_id=contact.wa_id,
        display_name=contact.display_name,
        meta=contact.meta,
        created_at=contact.created_at,
        updated_at=contact.updated_at,
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
