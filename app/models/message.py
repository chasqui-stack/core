"""Messages — full inbound/outbound history; this is the LLM context (§6)."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, Index, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Message(SQLModel, table=True):
    """One inbound ("in") or outbound ("out") message in a conversation."""

    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_conversation_created", "conversation_id", "created_at"),
        # Per-conversation pending-batch gather (ADR-008) — partial.
        Index(
            "ix_messages_pending_inbound",
            "conversation_id",
            "created_at",
            postgresql_where=text("processed_at IS NULL AND direction = 'in'"),
        ),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    conversation_id: uuid.UUID = Field(foreign_key="conversations.id", nullable=False)

    direction: str = Field(max_length=3, nullable=False, description='"in" | "out"')
    type: str = Field(
        max_length=20, nullable=False, description="text | audio | image | button | ..."
    )

    text: str | None = Field(default=None)
    media_url: str | None = Field(default=None, max_length=1024)

    # Channel payload / extra info ("metadata" is reserved in SQLModel)
    meta: dict = Field(
        default_factory=dict,
        sa_column=Column("metadata", JSONB, nullable=False, server_default="{}"),
    )

    # For inbound messages this is the gateway's `received_at`; for outbound,
    # the moment the core produced the reply. Naive UTC.
    created_at: datetime = Field(default_factory=_utcnow_naive, nullable=False)

    # Inbound coalescing (ADR-008): NULL = pending (not yet folded into a turn).
    # The coalesce worker gathers pending inbound (processed_at IS NULL,
    # direction='in') for a conversation, runs ONE turn, then stamps them. Marks
    # pending state per-row instead of watermarking on created_at (the gateway
    # clock is not monotonic). Outbound rows leave this NULL (filtered by direction).
    processed_at: datetime | None = Field(default=None, nullable=True)
