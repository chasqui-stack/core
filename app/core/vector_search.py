"""Dim-aware pgvector search & indexing strategy (ADR-001, parent repo).

EMBEDDING_DIM decides how we index and how we query — and both MUST agree,
because Postgres only uses an expression index when the query expression
matches it textually:

- dim <= 2000          -> HNSW on the `vector` column; plain cosine_distance.
- 2000 < dim <= 4000   -> HNSW on a `halfvec` cast expression (pgvector >= 0.7);
                          queries cast BOTH sides to halfvec(dim).
- dim > 4000           -> no ANN index possible; exact scan + startup warning.

Centralizing the expression here is what keeps the index and the query from
drifting apart. Used by memory_service and the faq module.
"""

import logging
from typing import Any

from sqlalchemy import cast
from pgvector.sqlalchemy import HALFVEC

from app.core.config import settings

logger = logging.getLogger(__name__)

MAX_VECTOR_INDEX_DIM = 2000  # HNSW limit for `vector` columns
MAX_HALFVEC_INDEX_DIM = 4000  # HNSW limit for `halfvec` (pgvector >= 0.7)


def index_strategy(dim: int | None = None) -> str:
    """'vector' | 'halfvec' | 'exact' for the given (or configured) dim."""
    dim = dim or settings.embedding_dim
    if dim <= MAX_VECTOR_INDEX_DIM:
        return "vector"
    if dim <= MAX_HALFVEC_INDEX_DIM:
        return "halfvec"
    return "exact"


def cosine_distance(column: Any, vector: list[float]) -> Any:
    """ORDER BY-ready cosine distance expression matching the index strategy."""
    dim = settings.embedding_dim
    if index_strategy(dim) == "halfvec":
        return cast(column, HALFVEC(dim)).cosine_distance(
            cast(vector, HALFVEC(dim))
        )
    return column.cosine_distance(vector)


def hnsw_index_ddl(table: str, column: str = "embedding", dim: int | None = None) -> str | None:
    """CREATE INDEX statement for the strategy, or None when ANN isn't possible.

    Used by Alembic migrations — the strategy is provision-time, like the dim
    itself (changing EMBEDDING_DIM later = column migration + re-embed).
    """
    dim = dim or settings.embedding_dim
    strategy = index_strategy(dim)
    name = f"ix_{table}_{column}_hnsw"
    if strategy == "vector":
        return (
            f"CREATE INDEX IF NOT EXISTS {name} ON {table} "
            f"USING hnsw ({column} vector_cosine_ops)"
        )
    if strategy == "halfvec":
        return (
            f"CREATE INDEX IF NOT EXISTS {name} ON {table} "
            f"USING hnsw (({column}::halfvec({dim})) halfvec_cosine_ops)"
        )
    logger.warning(
        "EMBEDDING_DIM=%s exceeds every ANN index limit (vector<=%s, halfvec<=%s): "
        "%s.%s will use exact scans — fine for small tables, slow at scale.",
        dim,
        MAX_VECTOR_INDEX_DIM,
        MAX_HALFVEC_INDEX_DIM,
        table,
        column,
    )
    return None
