"""Long-term memory service — SKELETON (ARCHITECTURE §6).

Sprint 1 defines the two interfaces the orchestrator will consume; the real
implementation (embeddings + pgvector retrieval + LLM extraction) lands in
Sprint 3/4. Keeping the seam here means the orchestrator never changes when
memory becomes real.
"""

import uuid

from sqlmodel.ext.asyncio.session import AsyncSession

from app.models import Memory


async def retrieve_relevant(
    session: AsyncSession,
    contact_id: uuid.UUID,
    query: str,
    limit: int = 5,
) -> list[Memory]:
    """Return the memories most relevant to `query` for this contact.

    Sprint 3/4: embed `query` and run a pgvector similarity search over
    `memories.embedding`. For now: nothing is relevant.
    """
    return []


async def extract_after_turn(
    session: AsyncSession,
    contact_id: uuid.UUID,
    conversation_id: uuid.UUID,
) -> None:
    """Extract durable facts/summaries from the latest turn into `memories`.

    Sprint 3/4: LLM extraction + embedding, possibly async to the request.
    For now: no-op.
    """
    return None
