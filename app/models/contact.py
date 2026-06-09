"""Contacts — the person on the other side of a channel (ARCHITECTURE §6, §10).

BSUID-first: `external_id` holds the WhatsApp BSUID (or the channel-scoped id
for other channels). `wa_id` is stored when available but is NOT the identity.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


def _utcnow_naive() -> datetime:
    """Naive UTC now (asyncpg requires naive datetimes for TIMESTAMP columns)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Contact(SQLModel, table=True):
    """End user reached through a channel. Never authenticates (§4)."""

    __tablename__ = "contacts"
    __table_args__ = (
        UniqueConstraint("channel", "external_id", name="uq_contacts_channel_external_id"),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    # Identity — unique per (channel, external_id); external_id = BSUID on WhatsApp
    channel: str = Field(max_length=32, nullable=False, index=True)
    external_id: str = Field(max_length=255, nullable=False)

    # WhatsApp phone-derived id — optional under BSUID, may be null
    wa_id: str | None = Field(default=None, max_length=32, index=True)

    display_name: str | None = Field(default=None, max_length=255)

    # Free-form channel/contact metadata ("metadata" is reserved in SQLModel,
    # so the Python attribute is `meta` mapped to the "metadata" column)
    meta: dict = Field(
        default_factory=dict,
        sa_column=Column("metadata", JSONB, nullable=False, server_default="{}"),
    )

    created_at: datetime = Field(default_factory=_utcnow_naive, nullable=False)
    updated_at: datetime = Field(default_factory=_utcnow_naive, nullable=False)
