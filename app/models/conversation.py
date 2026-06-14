"""Conversations — a single thread per contact (ARCHITECTURE §6)."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, Index, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Conversation(SQLModel, table=True):
    """The one conversation thread for a contact.

    `contact_id` is unique — the schema itself enforces the
    single-thread-per-contact rule.
    """

    __tablename__ = "conversations"
    __table_args__ = (
        # The coalesce worker only ever scans armed rows (ADR-008) — partial.
        Index(
            "ix_conversations_debounce_due",
            "debounce_due_at",
            postgresql_where=text("debounce_due_at IS NOT NULL"),
        ),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    contact_id: uuid.UUID = Field(
        foreign_key="contacts.id", nullable=False, unique=True, index=True
    )

    # Who owns the replies (ADR-004): "agent" (default) or "human". The ingest
    # pipeline checks this FIRST — human mode persists the inbound and runs no
    # agent turn. A real indexed column, never a JSONB flag.
    mode: str = Field(default="agent", max_length=8, nullable=False, index=True)

    # Orchestrator state (LangGraph) — opaque to everything but the orchestrator
    conversation_state: dict = Field(
        default_factory=dict,
        sa_column=Column("conversation_state", JSONB, nullable=False, server_default="{}"),
    )

    # Inbound coalescing (ADR-008): when the current debounce window elapses.
    # NULL = nothing pending. The coalesce worker claims rows whose window has
    # passed (FOR UPDATE SKIP LOCKED) and runs ONE turn over the burst. Only
    # armed when INBOUND_DEBOUNCE_SECONDS > 0; the synchronous path never sets it.
    debounce_due_at: datetime | None = Field(default=None, nullable=True)

    created_at: datetime = Field(default_factory=_utcnow_naive, nullable=False)
    updated_at: datetime = Field(default_factory=_utcnow_naive, nullable=False)
