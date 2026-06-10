"""Embeddings factory — provider-swappable, like the chat model (app/core/llm.py).

Built on LangChain's `init_embeddings()` (the `init_chat_model()` twin):

    EMBEDDING_PROVIDER=google  EMBEDDING_MODEL=gemini-embedding-001   # default
    EMBEDDING_PROVIDER=openai  EMBEDDING_MODEL=text-embedding-3-small
    EMBEDDING_PROVIDER=ollama  EMBEDDING_MODEL=nomic-embed-text       # 768 native

**EMBEDDING_DIM = 768 is the project constant** — it must match `Vector(768)`
in app/models/memory.py (migration 002). Matryoshka providers (Google,
OpenAI) are asked for 768 explicitly; fixed-dim models must be 768 native.
Switching providers requires re-embedding existing rows: vectors from
different models live in different spaces and are not comparable.

Why 768 and not 3072 (gemini-embedding-001's default): MRL keeps ~all the
quality at 768, storage is 4x cheaper, and pgvector indexes `vector` columns
only up to 2,000 dims (3072 would force `halfvec`, see the embeddings ADR in
the parent repo's docs/design/).
"""

from functools import lru_cache

from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings

from app.core.config import settings

EMBEDDING_DIM = 768

# settings.embedding_provider → init_embeddings' provider string
_PROVIDER_MAP = {"google": "google_genai"}

# Matryoshka providers accept a requested output dimension; anything not
# listed here must produce EMBEDDING_DIM natively (e.g. nomic-embed-text).
_DIM_KWARG = {"google_genai": "output_dimensionality", "openai": "dimensions"}


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """The configured embeddings model (cached — stateless client)."""
    provider = _PROVIDER_MAP.get(settings.embedding_provider, settings.embedding_provider)

    kwargs: dict = {}
    if dim_kwarg := _DIM_KWARG.get(provider):
        kwargs[dim_kwarg] = EMBEDDING_DIM

    if provider == "google_genai" and settings.google_api_key:
        kwargs["google_api_key"] = settings.google_api_key
    elif provider == "openai" and settings.openai_api_key:
        kwargs["api_key"] = settings.openai_api_key
    elif provider == "ollama" and settings.ollama_base_url:
        kwargs["base_url"] = settings.ollama_base_url

    return init_embeddings(f"{provider}:{settings.embedding_model}", **kwargs)
