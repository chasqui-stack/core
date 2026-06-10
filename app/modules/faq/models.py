"""FAQ entries — the Q&A knowledge base, pgvector-backed (ARCHITECTURE §8).

Module-contributed table: it lives with the module, not in app/models/.
The vector width comes from EMBEDDING_DIM (.env, provision-time — ADR-001).
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel

from app.core.config import settings


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class FaqEntry(SQLModel, table=True):
    """One Q&A pair of the knowledge base (atomic — no chunking)."""

    __tablename__ = "faq_entries"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    question: str = Field(nullable=False, description="The question (also what gets embedded)")
    answer: str = Field(nullable=False, description="The grounded answer the agent relays")

    tags: list[str] = Field(
        default_factory=list,
        sa_column=Column("tags", JSONB, nullable=False, server_default="[]"),
    )

    # Nullable: an embeddings outage must never block the CRUD — the
    # "re-embed all" admin action backfills.
    embedding: Any = Field(
        default=None,
        sa_column=Column("embedding", Vector(settings.embedding_dim), nullable=True),
    )

    created_at: datetime = Field(default_factory=_utcnow_naive, nullable=False)
    updated_at: datetime = Field(default_factory=_utcnow_naive, nullable=False)
