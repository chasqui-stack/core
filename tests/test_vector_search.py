"""Unit tests for the dim-aware index/query strategy (ADR-001, no DB)."""

from app.core.config import settings
from app.core.vector_search import cosine_distance, hnsw_index_ddl, index_strategy
from app.models import Memory


def test_strategy_selection_by_dim(monkeypatch):
    assert index_strategy(768) == "vector"
    assert index_strategy(2000) == "vector"
    assert index_strategy(2001) == "halfvec"
    assert index_strategy(3072) == "halfvec"  # gemini native
    assert index_strategy(4000) == "halfvec"
    assert index_strategy(4001) == "exact"

    monkeypatch.setattr(settings, "embedding_dim", 3072)
    assert index_strategy() == "halfvec"  # defaults to the configured dim


def test_query_expression_matches_strategy(monkeypatch):
    vector = [0.1, 0.2]

    monkeypatch.setattr(settings, "embedding_dim", 768)
    plain = str(cosine_distance(Memory.embedding, vector))
    assert "<=>" in plain and "HALFVEC" not in plain

    monkeypatch.setattr(settings, "embedding_dim", 3072)
    casted = str(cosine_distance(Memory.embedding, vector))
    # Both sides cast, so Postgres can use the halfvec expression index
    assert casted.count("HALFVEC(3072)") == 2 and "<=>" in casted


def test_index_ddl_per_strategy(monkeypatch):
    monkeypatch.setattr(settings, "embedding_dim", 768)
    ddl = hnsw_index_ddl("memories")
    assert "USING hnsw (embedding vector_cosine_ops)" in ddl

    monkeypatch.setattr(settings, "embedding_dim", 3072)
    ddl = hnsw_index_ddl("faq_entries")
    assert "((embedding::halfvec(3072)) halfvec_cosine_ops)" in ddl

    monkeypatch.setattr(settings, "embedding_dim", 5000)
    assert hnsw_index_ddl("memories") is None  # exact scan — no ANN index
