"""Unit tests for the provider-swappable embeddings factory (no network)."""

import pytest

import app.core.embeddings as embeddings_mod
from app.core.config import settings
from app.core.embeddings import get_embeddings


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


def test_google_maps_to_google_genai_and_requests_configured_dim(fresh_factory, monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "google")
    monkeypatch.setattr(settings, "embedding_model", "gemini-embedding-001")

    get_embeddings()

    model, kwargs = fresh_factory[0]
    assert model == "google_genai:gemini-embedding-001"
    assert kwargs["output_dimensionality"] == settings.embedding_dim


def test_openai_requests_configured_dimensions(fresh_factory, monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "openai")
    monkeypatch.setattr(settings, "embedding_model", "text-embedding-3-small")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")

    get_embeddings()

    model, kwargs = fresh_factory[0]
    assert model == "openai:text-embedding-3-small"
    assert kwargs["dimensions"] == settings.embedding_dim
    assert kwargs["api_key"] == "sk-test"


def test_embedding_dim_is_env_driven(fresh_factory, monkeypatch):
    """A dev choosing gemini at full 3072 dims only edits .env (provision-time)."""
    monkeypatch.setattr(settings, "embedding_provider", "google")
    monkeypatch.setattr(settings, "embedding_model", "gemini-embedding-001")
    monkeypatch.setattr(settings, "embedding_dim", 3072)

    get_embeddings()

    _, kwargs = fresh_factory[0]
    assert kwargs["output_dimensionality"] == 3072


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
