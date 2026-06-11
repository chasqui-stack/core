"""Leads — module-contributed table (ARCHITECTURE §8, like faq_entries).

Sprint 7 graduates leads from conversation_state JSONB (audit-only) to a
first-class table the admin panel lists. The `extra` JSONB holds answers to
the operator-configured `extra_fields` (lead_capture config schema).
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Lead(SQLModel, table=True):
    """One captured lead, tied to the contact who produced it."""

    __tablename__ = "leads"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    contact_id: uuid.UUID = Field(foreign_key="contacts.id", nullable=False, index=True)

    name: str = Field(max_length=255, nullable=False)
    interest: str | None = Field(default=None, max_length=500)
    email: str | None = Field(default=None, max_length=255)
    phone: str | None = Field(default=None, max_length=32)
    notes: str | None = Field(default=None)

    # Answers to the operator-configured extra_fields (e.g. company, city)
    extra: dict = Field(
        default_factory=dict,
        sa_column=Column("extra", JSONB, nullable=False, server_default="{}"),
    )

    created_at: datetime = Field(default_factory=_utcnow_naive, nullable=False)
