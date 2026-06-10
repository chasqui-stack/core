"""Long-term memory service (ARCHITECTURE §6).

Retrieval: embed the query and run a pgvector cosine search over
`memories.embedding`, scoped to the contact. The orchestrator injects the
hits into the system prompt before every turn.

Writing: the agent itself saves facts via the `save_memory` tool
(app/modules/memory) — extraction-by-tool means one LLM call per turn.
`extract_after_turn` remains as a seam for batch/offline extraction.
"""

import logging
import uuid

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import settings
from app.core.vector_search import cosine_distance
from app.models import Memory

logger = logging.getLogger(__name__)


async def retrieve_relevant(
    session: AsyncSession,
    contact_id: uuid.UUID,
    query: str,
    limit: int = 5,
) -> list[Memory]:
    """Return the memories most relevant to `query` for this contact."""
    if not query or not settings.google_api_key:
        return []

    try:
        from app.core.embeddings import get_embeddings

        vector = await get_embeddings().aembed_query(query)
    except Exception:  # embeddings down ≠ broken turn — just no memories
        logger.warning("Embedding failed; skipping memory retrieval", exc_info=True)
        return []

    result = await session.exec(
        select(Memory)
        .where(Memory.contact_id == contact_id, Memory.embedding.is_not(None))
        .order_by(cosine_distance(Memory.embedding, vector))
        .limit(limit)
    )
    return list(result.all())


async def find_nearest(
    session: AsyncSession,
    contact_id: uuid.UUID,
    vector: list[float],
    max_distance: float,
) -> Memory | None:
    """The contact's closest memory to `vector`, or None beyond `max_distance`.

    Backs dedup-on-save and the update/forget tools (app/modules/memory).
    """
    distance = cosine_distance(Memory.embedding, vector)
    result = await session.exec(
        select(Memory, distance.label("distance"))
        .where(Memory.contact_id == contact_id, Memory.embedding.is_not(None))
        .order_by(distance)
        .limit(1)
    )
    row = result.first()
    if row is None:
        return None
    memory, dist = row
    return memory if dist <= max_distance else None


async def extract_after_turn(
    session: AsyncSession,
    contact_id: uuid.UUID,
    conversation_id: uuid.UUID,
) -> None:
    """Seam for batch/offline memory extraction.

    Inline extraction is covered by the `save_memory` tool (the agent saves
    facts during the turn). Keep this hook for future summarization jobs.
    """
    return None
