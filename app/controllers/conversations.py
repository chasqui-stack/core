"""Internal conversation read (ADR-011, ARCHITECTURE §5).

A gateway-facing, `INTERNAL_API_KEY`-protected read of a thread's recent
messages, scoped by `(channel, external_id)` — the read-mirror of `/ingest`.
The web channel uses it to rehydrate a visitor's chat on (re)open; it is
**generic** (channel is a path param), reusable by any channel.

Like the admin timeline, it NEVER serializes embeddings or media payloads:
`has_media` is a boolean and the bytes are fetched separately. It is read-only —
a contact that doesn't exist yet is a 404 (no create).
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.controllers.ingest import verify_internal_key
from app.core import storage
from app.db.session import get_session
from app.models import Conversation, Message
from app.schemas.admin_contacts import MessageItem, MessageListResponse
from app.services.ingest_service import find_contact

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(verify_internal_key)])


@router.get(
    "/conversations/{channel}/{external_id}/messages",
    response_model=MessageListResponse,
)
async def read_history(
    channel: str,
    external_id: str,
    session: AsyncSession = Depends(get_session),
    limit: int = Query(default=50, ge=1, le=200),
):
    """Recent messages for `(channel, external_id)`, newest first.

    Read-only: 404 when the contact doesn't exist yet. Mirrors the admin
    timeline serialization (`has_media` boolean, no payloads).
    """
    contact = await find_contact(session, channel, external_id)
    if contact is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NO_CONTACT", "message": "No such contact"},
        )

    conversation = (
        await session.exec(
            select(Conversation).where(Conversation.contact_id == contact.id)
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

    messages = (
        await session.exec(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(col(Message.created_at).desc())
            .limit(limit)
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
