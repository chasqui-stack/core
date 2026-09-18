"""Knowledge documents — uploaded files, chunked and pgvector-backed.

Module-contributed tables: they live with the module, not in app/models/.
The vector width comes from EMBEDDING_DIM (.env, provision-time — ADR-001).

The original file is never stored: `content_text` keeps the extracted text,
which is all `reprocess` needs to re-chunk and re-embed.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, ForeignKey, Text, Uuid
from sqlmodel import Field, SQLModel

from app.core.config import settings

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_READY = "ready"
STATUS_ERROR = "error"


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Document(SQLModel, table=True):
    """One uploaded file and its processing state."""

    __tablename__ = "documents"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    filename: str = Field(max_length=255, nullable=False)
    mime_type: str = Field(max_length=255, nullable=False)
    size_bytes: int = Field(nullable=False)
    content_sha256: str = Field(
        max_length=64,
        nullable=False,
        unique=True,
        description="sha256 of the raw upload — dedupe key",
    )

    # Extracted text — the source for reprocess. NULL until extraction succeeds.
    content_text: str | None = Field(
        default=None, sa_column=Column("content_text", Text, nullable=True)
    )

    status: str = Field(default=STATUS_PENDING, max_length=20, nullable=False, index=True)
    error_detail: str | None = Field(default=None, max_length=500)
    chunk_count: int = Field(default=0, nullable=False)

    created_at: datetime = Field(default_factory=_utcnow_naive, nullable=False)
    updated_at: datetime = Field(default_factory=_utcnow_naive, nullable=False)


class DocumentChunk(SQLModel, table=True):
    """One overlapping slice of a document's text, with its embedding."""

    __tablename__ = "document_chunks"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    document_id: uuid.UUID = Field(
        sa_column=Column(
            "document_id",
            Uuid,
            ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    seq: int = Field(nullable=False, description="0-based order within the document")
    content: str = Field(sa_column=Column("content", Text, nullable=False))

    embedding: Any = Field(
        default=None,
        sa_column=Column("embedding", Vector(settings.embedding_dim), nullable=True),
    )

    created_at: datetime = Field(default_factory=_utcnow_naive, nullable=False)
