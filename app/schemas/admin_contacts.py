"""Schemas for the read-only conversation-inspection endpoints (/admin/contacts)."""

import uuid
from datetime import datetime

from pydantic import BaseModel


class LastMessagePreview(BaseModel):
    direction: str
    type: str
    text: str | None
    created_at: datetime


class ContactListItem(BaseModel):
    id: uuid.UUID
    channel: str
    external_id: str
    wa_id: str | None
    display_name: str | None
    created_at: datetime
    updated_at: datetime
    message_count: int
    last_message: LastMessagePreview | None


class ContactListResponse(BaseModel):
    items: list[ContactListItem]
    total: int


class ContactDetail(BaseModel):
    id: uuid.UUID
    channel: str
    external_id: str
    wa_id: str | None
    display_name: str | None
    meta: dict
    created_at: datetime
    updated_at: datetime


class MessageItem(BaseModel):
    id: uuid.UUID
    direction: str
    type: str
    text: str | None
    meta: dict
    created_at: datetime
    # media_url is intentionally absent: history never stores media (the
    # gateway inlines data: URIs for the current turn only) and the admin
    # must never receive blobs.


class MessageListResponse(BaseModel):
    items: list[MessageItem]
    total: int


class MemoryItem(BaseModel):
    id: uuid.UUID
    content: str
    has_embedding: bool  # never the vector itself
    created_at: datetime


class MemoryListResponse(BaseModel):
    items: list[MemoryItem]
