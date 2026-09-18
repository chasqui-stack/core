"""Knowledge service — extract, chunk, embed, search.

Processing runs as a FastAPI background task, so `process_document` opens
its OWN session (the request's is gone by then) and never raises: every
failure lands in the document row (`status="error"` + `error_detail`), and
`reprocess` recovers it from the stored text.

A document is `ready` only with chunks AND vectors — never half-indexed.
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import PurePath

from sqlmodel import delete, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.vector_search import cosine_distance
from app.db import session as db_session
from app.modules.knowledge.extract import extension_of, extract_text
from app.modules.knowledge.models import (
    STATUS_ERROR,
    STATUS_PENDING,
    STATUS_PROCESSING,
    STATUS_READY,
    Document,
    DocumentChunk,
)

logger = logging.getLogger(__name__)

# Omakase constants, not operator knobs — changing them = reprocess.
CHUNK_SIZE = 1200  # characters
CHUNK_OVERLAP = 180  # 15%

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# A crash mid-job leaves the row in pending/processing forever (there is no
# worker to resume it). Past this age the job is presumed dead and
# `reprocess` may take over.
STALE_AFTER = timedelta(minutes=10)


class DuplicateDocument(Exception):
    """A document with the same sha256 already exists."""


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def is_busy(document: Document) -> bool:
    """True while a (presumably live) background job owns the document."""
    if document.status not in (STATUS_PENDING, STATUS_PROCESSING):
        return False
    return _utcnow_naive() - document.updated_at < STALE_AFTER


def split_text(filename: str, text: str) -> list[str]:
    """Overlapping chunks; markdown splits on headings/fences first."""
    from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

    if extension_of(filename) == ".md":
        splitter = RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
    return [chunk for chunk in splitter.split_text(text) if chunk.strip()]


async def create_document(
    session: AsyncSession, *, filename: str, mime_type: str | None, data: bytes
) -> Document:
    """Insert the `pending` row. Raises DuplicateDocument on a known sha256."""
    sha256 = hashlib.sha256(data).hexdigest()
    existing = await session.exec(
        select(Document.id).where(Document.content_sha256 == sha256)
    )
    if existing.first() is not None:
        raise DuplicateDocument()

    document = Document(
        filename=PurePath(filename).name[:255],
        mime_type=(mime_type or "application/octet-stream")[:255],
        size_bytes=len(data),
        content_sha256=sha256,
    )
    session.add(document)
    await session.flush()
    return document


async def _mark(session: AsyncSession, document: Document, status: str) -> None:
    document.status = status
    document.updated_at = _utcnow_naive()
    session.add(document)
    await session.commit()


async def process_document(
    document_id: uuid.UUID, data: bytes | None = None, session_factory=None
) -> None:
    """Background job: (extract →) chunk → embed → `ready`, or `error`.

    `data` is the raw upload on the first run; None on reprocess, which
    starts from the stored `content_text`. Never raises.
    """
    factory = session_factory or db_session.async_session_factory
    try:
        async with factory() as session:
            await _process(session, document_id, data)
    except Exception:  # pragma: no cover - last resort, e.g. the DB is down
        logger.exception("Knowledge processing crashed for document %s", document_id)


async def _process(
    session: AsyncSession, document_id: uuid.UUID, data: bytes | None
) -> None:
    document = await session.get(Document, document_id)
    if document is None:  # deleted between the upload and the job
        return
    await _mark(session, document, STATUS_PROCESSING)

    text = document.content_text
    try:
        if data is not None:
            text = await asyncio.to_thread(extract_text, document.filename, data)
            document.content_text = text
        if not text:
            raise ValueError("no stored text — delete the document and upload it again")

        chunks = split_text(document.filename, text)
        if not chunks:
            raise ValueError("no extractable text")

        from app.core.embeddings import get_embeddings

        # ONE batched call per document — per-chunk calls burn rate limits
        vectors = await get_embeddings().aembed_documents(chunks)
        if len(vectors) != len(chunks):
            raise RuntimeError("the embeddings provider returned a partial batch")

        await session.exec(
            delete(DocumentChunk).where(DocumentChunk.document_id == document.id)
        )
        session.add_all(
            DocumentChunk(document_id=document.id, seq=seq, content=chunk, embedding=vector)
            for seq, (chunk, vector) in enumerate(zip(chunks, vectors))
        )
        document.chunk_count = len(chunks)
        document.error_detail = None
        await _mark(session, document, STATUS_READY)
    except Exception as exc:
        logger.warning(
            "Knowledge processing failed for '%s'", document.filename, exc_info=True
        )
        await session.rollback()
        document = await session.get(Document, document_id)
        if document is None:
            return
        await session.exec(
            delete(DocumentChunk).where(DocumentChunk.document_id == document.id)
        )
        if text:  # extraction succeeded — keep it so reprocess can recover
            document.content_text = text
        document.chunk_count = 0
        document.error_detail = (str(exc) or type(exc).__name__)[:500]
        await _mark(session, document, STATUS_ERROR)


async def mark_for_reprocess(session: AsyncSession, document: Document) -> None:
    """Back to `pending`; the caller schedules `process_document(id)`."""
    document.status = STATUS_PENDING
    document.error_detail = None
    document.updated_at = _utcnow_naive()
    session.add(document)
    await session.flush()


async def list_ready_filenames(session: AsyncSession) -> list[str]:
    """Filenames of searchable documents, oldest first (prompt document index)."""
    result = await session.exec(
        select(Document.filename)
        .where(Document.status == STATUS_READY)
        .order_by(Document.created_at)
    )
    return list(result.all())


async def search(
    session: AsyncSession,
    query: str,
    *,
    top_k: int = 4,
    min_similarity: float = 0.5,
) -> list[tuple[DocumentChunk, Document, float]]:
    """Top-k chunks of `ready` documents with similarity >= floor, best first."""
    try:
        from app.core.embeddings import get_embeddings

        vector = await get_embeddings().aembed_query(query)
    except Exception:  # embeddings down ≠ broken panel/turn — degrade to no hits
        logger.warning("Embedding failed for knowledge search", exc_info=True)
        return []

    distance = cosine_distance(DocumentChunk.embedding, vector)
    max_distance = 1.0 - min_similarity
    result = await session.exec(
        select(DocumentChunk, Document, distance.label("distance"))
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            Document.status == STATUS_READY,
            DocumentChunk.embedding.is_not(None),
            distance <= max_distance,
        )
        .order_by(distance)
        .limit(top_k)
    )
    return [(chunk, document, 1.0 - dist) for chunk, document, dist in result.all()]
