"""knowledge module: documents + document_chunks (Document-RAG, pgvector)

Revision ID: 008_knowledge_documents
Revises: 007_inbound_coalescing
Create Date: 2026-09-18

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from app.core.vector_search import hnsw_index_ddl

# revision identifiers, used by Alembic.
revision: str = "008_knowledge_documents"
down_revision: Union[str, Sequence[str], None] = "007_inbound_coalescing"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Provision-time: the vector width comes from .env (EMBEDDING_DIM) at the
# moment this migration first runs (same as 002/004). See ADR-001.
from app.core.config import settings

EMBEDDING_DIM = settings.embedding_dim


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("mime_type", sa.String(length=255), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("content_text", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("error_detail", sa.String(length=500), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("content_sha256"),
    )
    op.create_index("ix_documents_status", "documents", ["status"])

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
    )
    op.create_index(
        "ix_document_chunks_document_id", "document_chunks", ["document_id"]
    )

    # ANN index, auto-selected by EMBEDDING_DIM (None when dim > 4000)
    ddl = hnsw_index_ddl("document_chunks")
    if ddl is not None:
        op.execute(ddl)


def downgrade() -> None:
    op.drop_table("document_chunks")
    op.drop_table("documents")
