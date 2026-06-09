"""Conversations — a single thread per contact (ARCHITECTURE §6)."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column
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

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    contact_id: uuid.UUID = Field(
        foreign_key="contacts.id", nullable=False, unique=True, index=True
    )

    # Orchestrator state (LangGraph) — opaque to everything but the orchestrator
    conversation_state: dict = Field(
        default_factory=dict,
        sa_column=Column("conversation_state", JSONB, nullable=False, server_default="{}"),
    )

    created_at: datetime = Field(default_factory=_utcnow_naive, nullable=False)
    updated_at: datetime = Field(default_factory=_utcnow_naive, nullable=False)
