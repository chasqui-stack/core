"""ANN indexes on memories + faq_entries, auto-selected by EMBEDDING_DIM

Revision ID: 005_vector_indexes
Revises: 004_faq_entries
Create Date: 2026-06-10

Strategy (ADR-001, parent repo): EMBEDDING_DIM <= 2000 -> HNSW on `vector`;
2001-4000 -> HNSW on a halfvec cast (pgvector >= 0.7); > 4000 -> no ANN index
(exact scan — hnsw_index_ddl logs the warning). The query side mirrors this
in app/core/vector_search.py: index and query expressions can't drift.
"""
from typing import Sequence, Union

from alembic import op

from app.core.vector_search import hnsw_index_ddl

# revision identifiers, used by Alembic.
revision: str = "005_vector_indexes"
down_revision: Union[str, Sequence[str], None] = "004_faq_entries"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLES = ("memories", "faq_entries")


def upgrade() -> None:
    for table in TABLES:
        ddl = hnsw_index_ddl(table)  # None when dim > 4000 (exact scan)
        if ddl is not None:
            op.execute(ddl)


def downgrade() -> None:
    for table in TABLES:
        op.execute(f"DROP INDEX IF EXISTS ix_{table}_embedding_hnsw")
