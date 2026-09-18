"""Knowledge admin endpoints — mounted under /admin/modules/knowledge (JWT-protected).

The parent router in app/main.py enforces admin auth for every module route;
this file only declares upload/list/delete/reprocess/search. Schemas live
here too: the module stays self-contained (nothing added to app/schemas/).
"""

import uuid
from datetime import datetime

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    UploadFile,
    status,
)
from pydantic import BaseModel
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.session import get_session
from app.modules.knowledge import service
from app.modules.knowledge.extract import UnsupportedType, validate_type
from app.modules.knowledge.models import Document

SNIPPET_CHARS = 400


class DocumentResponse(BaseModel):
    id: uuid.UUID
    filename: str
    mime_type: str
    size_bytes: int
    status: str
    error_detail: str | None
    chunk_count: int
    # False when extraction itself failed (e.g. a scanned PDF): no stored text
    can_reprocess: bool
    created_at: datetime
    updated_at: datetime


class SearchHit(BaseModel):
    document_id: uuid.UUID
    filename: str
    seq: int
    content: str  # snippet — the full chunk is noise for the panel
    similarity: float


def _to_response(document: Document) -> DocumentResponse:
    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        mime_type=document.mime_type,
        size_bytes=document.size_bytes,
        status=document.status,
        error_detail=document.error_detail,
        chunk_count=document.chunk_count,
        can_reprocess=bool(document.content_text),
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


async def _get_or_404(session: AsyncSession, document_id: uuid.UUID) -> Document:
    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )
    return document


def register(router: APIRouter) -> None:
    """Module hook target — declares the routes on the module's sub-router."""

    @router.get("/search", response_model=list[SearchHit])
    async def search_preview(
        q: str,
        top_k: int = 8,
        min_similarity: float = 0.0,
        session: AsyncSession = Depends(get_session),
    ):
        """Retrieval preview for operators — what would the agent see?

        No similarity floor by default, so the panel can show the scores
        that tune the tool's threshold. Embeddings outage degrades to [].
        """
        hits = await service.search(
            session, q, top_k=top_k, min_similarity=min_similarity
        )
        return [
            SearchHit(
                document_id=document.id,
                filename=document.filename,
                seq=chunk.seq,
                content=chunk.content[:SNIPPET_CHARS],
                similarity=round(similarity, 4),
            )
            for chunk, document, similarity in hits
        ]

    @router.get("/documents", response_model=list[DocumentResponse])
    async def list_documents(session: AsyncSession = Depends(get_session)):
        result = await session.exec(
            select(Document).order_by(Document.created_at.desc())
        )
        return [_to_response(d) for d in result.all()]

    @router.post(
        "/documents",
        response_model=DocumentResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def upload_document(
        file: UploadFile,
        background_tasks: BackgroundTasks,
        session: AsyncSession = Depends(get_session),
    ):
        """Accept the file and process it in the background (poll the list)."""
        filename = file.filename or ""
        try:
            validate_type(filename, file.content_type)
        except UnsupportedType as exc:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)
            )

        # Content-Length can lie — cap what we actually read instead
        data = await file.read(service.MAX_UPLOAD_BYTES + 1)
        if len(data) > service.MAX_UPLOAD_BYTES:
            limit_mb = service.MAX_UPLOAD_BYTES // (1024 * 1024)
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"File exceeds the {limit_mb} MB limit",
            )
        if not data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="The file is empty"
            )

        try:
            document = await service.create_document(
                session, filename=filename, mime_type=file.content_type, data=data
            )
        except service.DuplicateDocument:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This document was already uploaded",
            )
        # Commit BEFORE scheduling: the job opens its own session and must
        # find the row.
        await session.commit()
        background_tasks.add_task(service.process_document, document.id, data)
        return _to_response(document)

    @router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_document(
        document_id: uuid.UUID, session: AsyncSession = Depends(get_session)
    ):
        document = await _get_or_404(session, document_id)
        await session.delete(document)  # chunks go with it (FK cascade)
        await session.commit()

    @router.post(
        "/documents/{document_id}/reprocess",
        response_model=DocumentResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def reprocess_document(
        document_id: uuid.UUID,
        background_tasks: BackgroundTasks,
        session: AsyncSession = Depends(get_session),
    ):
        """Re-chunk + re-embed from the stored text — also recovers `error` docs."""
        document = await _get_or_404(session, document_id)
        if service.is_busy(document):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The document is still being processed",
            )
        if not document.content_text:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="No stored text to reprocess — delete the document and upload it again",
            )
        await service.mark_for_reprocess(session, document)
        await session.commit()
        background_tasks.add_task(service.process_document, document.id)
        return _to_response(document)
