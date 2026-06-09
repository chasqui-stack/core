"""Canonical message contract — the gateway ↔ core seam (ARCHITECTURE §5).

The core only speaks this contract; it never knows which channel is on the
other side. Any new channel (web, telegram, ...) is "just another adapter"
that produces an IngestRequest and consumes an IngestResponse.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class ContactPayload(BaseModel):
    """Who sent the message. `external_id` is the BSUID on WhatsApp (§10)."""

    external_id: str = Field(min_length=1, max_length=255)
    wa_id: str | None = None  # optional, may be null under BSUID
    display_name: str | None = None
    metadata: dict = Field(default_factory=dict)


class InboundMessage(BaseModel):
    """The message as normalized by the gateway."""

    type: str = Field(default="text", description="text | audio | image | button | ...")
    text: str | None = None
    media_url: str | None = None
    raw: dict = Field(default_factory=dict, description="Original channel payload if needed")


class IngestRequest(BaseModel):
    """Inbound: gateway → core, POST /ingest."""

    channel: str = Field(min_length=1, max_length=32, description='"whatsapp" | "web" | ...')
    contact: ContactPayload
    message: InboundMessage
    received_at: datetime | None = None


class OutboundMessage(BaseModel):
    """One reply message for the gateway to render on its channel."""

    type: str = "text"
    text: str | None = None
    media_url: str | None = None
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    """Outbound: core → gateway (agent reply; may be 1..N messages)."""

    messages: list[OutboundMessage]
    conversation_id: uuid.UUID
