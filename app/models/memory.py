"""Memories — long-term facts/summaries about a contact, pgvector-backed (§6).

Sprint 1 ships the table; extraction + retrieval wiring lands in Sprint 3/4.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column
from sqlmodel import Field, SQLModel

# Gemini text-embedding-004 dimension; revisit when the embedder is wired (Sprint 3/4)
EMBEDDING_DIM = 768


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Memory(SQLModel, table=True):
    """A long-term memory (fact/summary) extracted from conversations."""

    __tablename__ = "memories"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    contact_id: uuid.UUID = Field(foreign_key="contacts.id", nullable=False, index=True)

    content: str = Field(nullable=False, description="The memory text (fact/summary)")

    # Nullable until the embedder runs (extraction is async to the turn)
    embedding: Any = Field(
        default=None,
        sa_column=Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
    )

    created_at: datetime = Field(default_factory=_utcnow_naive, nullable=False)
