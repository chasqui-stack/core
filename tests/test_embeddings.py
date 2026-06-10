"""Unit tests for the provider-swappable embeddings factory (no network)."""

import pytest

import app.core.embeddings as embeddings_mod
from app.core.config import settings
from app.core.embeddings import EMBEDDING_DIM, get_embeddings


@pytest.fixture(autouse=True)
def fresh_factory(monkeypatch):
    """Clear the lru_cache and capture init_embeddings calls."""
    get_embeddings.cache_clear()
    calls = []

    def fake_init(model, **kwargs):
        calls.append((model, kwargs))
        return object()

    monkeypatch.setattr(embeddings_mod, "init_embeddings", fake_init)
    yield calls
    get_embeddings.cache_clear()


def test_google_maps_to_google_genai_and_requests_768(fresh_factory, monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "google")
    monkeypatch.setattr(settings, "embedding_model", "gemini-embedding-001")

    get_embeddings()

    model, kwargs = fresh_factory[0]
    assert model == "google_genai:gemini-embedding-001"
    assert kwargs["output_dimensionality"] == EMBEDDING_DIM


def test_openai_requests_dimensions_768(fresh_factory, monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "openai")
    monkeypatch.setattr(settings, "embedding_model", "text-embedding-3-small")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")

    get_embeddings()

    model, kwargs = fresh_factory[0]
    assert model == "openai:text-embedding-3-small"
    assert kwargs["dimensions"] == EMBEDDING_DIM
    assert kwargs["api_key"] == "sk-test"


def test_ollama_passes_through_without_dim_kwarg(fresh_factory, monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "ollama")
    monkeypatch.setattr(settings, "embedding_model", "nomic-embed-text")
    monkeypatch.setattr(settings, "ollama_base_url", "http://localhost:11434")

    get_embeddings()

    model, kwargs = fresh_factory[0]
    assert model == "ollama:nomic-embed-text"
    assert "dimensions" not in kwargs and "output_dimensionality" not in kwargs
    assert kwargs["base_url"] == "http://localhost:11434"


def test_factory_result_is_cached(fresh_factory, monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "google")
    assert get_embeddings() is get_embeddings()
    assert len(fresh_factory) == 1
