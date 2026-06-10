"""Embeddings factory — provider-swappable, like the chat model (app/core/llm.py).

Built on LangChain's `init_embeddings()` (the `init_chat_model()` twin):

    EMBEDDING_PROVIDER=google  EMBEDDING_MODEL=gemini-embedding-001   # default
    EMBEDDING_PROVIDER=openai  EMBEDDING_MODEL=text-embedding-3-small
    EMBEDDING_PROVIDER=ollama  EMBEDDING_MODEL=nomic-embed-text       # 768 native

**EMBEDDING_DIM (.env, default 768) is provision-time config** — the vector
column width is created from it on the first `alembic upgrade`, so changing
it afterwards requires a column migration + re-embedding every row.
Matryoshka providers (Google, OpenAI) are asked for it explicitly;
fixed-dim models (e.g. nomic-embed-text = 768) must match it natively.
Switching providers also requires re-embedding: vectors from different
models live in different spaces and are not comparable.

Why 768 as default (vs gemini-embedding-001's native 3072): MRL keeps ~all
the quality, storage is 4x cheaper, and pgvector HNSW-indexes `vector`
columns only up to 2,000 dims — 3072 needs a `halfvec` index (auto-selected
in Sprint 4). Full rationale: parent repo docs/design/adr-001.
"""

from functools import lru_cache

from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings

from app.core.config import settings

# settings.embedding_provider → init_embeddings' provider string
_PROVIDER_MAP = {"google": "google_genai"}

# Matryoshka providers accept a requested output dimension; anything not
# listed here must produce settings.embedding_dim natively.
_DIM_KWARG = {"google_genai": "output_dimensionality", "openai": "dimensions"}


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """The configured embeddings model (cached — stateless client)."""
    provider = _PROVIDER_MAP.get(settings.embedding_provider, settings.embedding_provider)

    kwargs: dict = {}
    if dim_kwarg := _DIM_KWARG.get(provider):
        kwargs[dim_kwarg] = settings.embedding_dim

    if provider == "google_genai" and settings.google_api_key:
        kwargs["google_api_key"] = settings.google_api_key
    elif provider == "openai" and settings.openai_api_key:
        kwargs["api_key"] = settings.openai_api_key
    elif provider == "ollama" and settings.ollama_base_url:
        kwargs["base_url"] = settings.ollama_base_url

    return init_embeddings(f"{provider}:{settings.embedding_model}", **kwargs)
