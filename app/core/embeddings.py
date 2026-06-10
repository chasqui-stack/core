"""Embeddings factory — used by long-term memory (§6) and FAQ-RAG (Sprint 4).

gemini-embedding-001 defaults to 3072 dims; we request 768 to match
EMBEDDING_DIM in app/models/memory.py and migration 002. If you swap the
embedding model, keep the dimension in sync (new migration). Cosine
distance is scale-invariant, so the non-normalized truncated vectors are
fine for pgvector's `<=>`.
"""

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import settings

EMBEDDING_DIM = 768


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """The configured embeddings model (cached — stateless client)."""
    return GoogleGenerativeAIEmbeddings(
        model=f"models/{settings.embedding_model}",
        google_api_key=settings.google_api_key,
        output_dimensionality=EMBEDDING_DIM,
    )
