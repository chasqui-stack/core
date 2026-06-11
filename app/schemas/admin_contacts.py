"""Schemas for the conversation endpoints (/admin/contacts).

Read-only inspection (Sprint 5) + the human-handoff inbox (Sprint 7,
ADR-004): conversation mode, operator messages, attention metadata.
"""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


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
    # Inbox fields (ADR-004) — "agent" unless a conversation says otherwise
    mode: str = "agent"
    handoff_reason: str | None = None
    handoff_at: str | None = None  # ISO string straight from the audit JSONB
    # The 24h-window anchor (channel-specific advisory, computed client-side)
    last_inbound_at: datetime | None = None


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
    mode: str = "agent"
    handoff_reason: str | None = None
    handoff_at: str | None = None
    last_inbound_at: datetime | None = None


class ModeUpdateRequest(BaseModel):
    mode: Literal["agent", "human"]


class ModeResponse(BaseModel):
    mode: str


class OperatorMessageCreate(BaseModel):
    text: str = Field(min_length=1, max_length=4096)


class MessageItem(BaseModel):
    id: uuid.UUID
    direction: str
    type: str
    text: str | None
    meta: dict
    created_at: datetime
    # media_url (the object key) is intentionally absent — the admin never
    # receives keys or blobs. has_media=True means GET /admin/media/{id}
    # will return a presigned URL for it (ADR-003).
    has_media: bool = False


class MediaUrlResponse(BaseModel):
    url: str
    expires_in: int


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
