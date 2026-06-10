"""FAQ knowledge-base service — embed on save, threshold search, re-embed all.

Embedding failures never break a request: entries are saved with
embedding=None (invisible to search) and the admin "re-embed all" action
backfills them later.
"""

import logging
from datetime import datetime, timezone

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.vector_search import cosine_distance
from app.modules.faq.models import FaqEntry

logger = logging.getLogger(__name__)


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _embeddable_text(question: str, answer: str) -> str:
    """What we vectorize: question + answer, so either side can match a query."""
    return f"{question}\n{answer}"


async def _try_embed(text: str) -> list[float] | None:
    try:
        from app.core.embeddings import get_embeddings

        return await get_embeddings().aembed_query(text)
    except Exception:  # embeddings down ≠ broken CRUD — backfill via /reembed
        logger.warning("Embedding failed; saving FAQ entry without vector", exc_info=True)
        return None


async def create_entry(
    session: AsyncSession,
    *,
    question: str,
    answer: str,
    tags: list[str] | None = None,
) -> FaqEntry:
    entry = FaqEntry(
        question=question,
        answer=answer,
        tags=tags or [],
        embedding=await _try_embed(_embeddable_text(question, answer)),
    )
    session.add(entry)
    await session.flush()
    return entry


async def update_entry(
    session: AsyncSession,
    entry: FaqEntry,
    *,
    question: str | None = None,
    answer: str | None = None,
    tags: list[str] | None = None,
) -> FaqEntry:
    content_changed = False
    if question is not None and question != entry.question:
        entry.question = question
        content_changed = True
    if answer is not None and answer != entry.answer:
        entry.answer = answer
        content_changed = True
    if tags is not None:
        entry.tags = tags

    if content_changed:  # edit → re-embed (stale vectors lie)
        entry.embedding = await _try_embed(_embeddable_text(entry.question, entry.answer))

    entry.updated_at = _utcnow_naive()
    session.add(entry)
    await session.flush()
    return entry


async def search(
    session: AsyncSession,
    query: str,
    *,
    top_k: int = 4,
    min_similarity: float = 0.5,
) -> list[tuple[FaqEntry, float]]:
    """Top-k entries with cosine similarity >= min_similarity, best first."""
    vector = await _try_embed(query)
    if vector is None:
        return []

    distance = cosine_distance(FaqEntry.embedding, vector)
    max_distance = 1.0 - min_similarity
    result = await session.exec(
        select(FaqEntry, distance.label("distance"))
        .where(FaqEntry.embedding.is_not(None), distance <= max_distance)
        .order_by(distance)
        .limit(top_k)
    )
    return [(entry, 1.0 - dist) for entry, dist in result.all()]


async def reembed_all(session: AsyncSession) -> int:
    """Re-embed every entry (one batched call). Returns how many were updated.

    For provider/model swaps at the SAME dim; a dim change is a migration
    (ADR-001), not a re-embed.
    """
    result = await session.exec(select(FaqEntry))
    entries = list(result.all())
    if not entries:
        return 0

    from app.core.embeddings import get_embeddings

    texts = [_embeddable_text(e.question, e.answer) for e in entries]
    vectors = await get_embeddings().aembed_documents(texts)

    now = _utcnow_naive()
    for entry, vector in zip(entries, vectors):
        entry.embedding = vector
        entry.updated_at = now
        session.add(entry)
    await session.flush()
    return len(entries)
