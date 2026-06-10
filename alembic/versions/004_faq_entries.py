"""faq module: faq_entries Q&A knowledge base (pgvector)

Revision ID: 004_faq_entries
Revises: 003_agent_config
Create Date: 2026-06-10

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = "004_faq_entries"
down_revision: Union[str, Sequence[str], None] = "003_agent_config"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Provision-time: the vector width comes from .env (EMBEDDING_DIM) at the
# moment this migration first runs (same as 002_domain). See ADR-001.
from app.core.config import settings

EMBEDDING_DIM = settings.embedding_dim


def upgrade() -> None:
    op.create_table(
        "faq_entries",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("tags", JSONB(), nullable=False, server_default="[]"),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    # ANN index lands in 005_vector_indexes (auto-selected by EMBEDDING_DIM)


def downgrade() -> None:
    op.drop_table("faq_entries")
