"""FAQ admin endpoints — mounted under /admin/modules/faq (JWT-protected).

The parent router in app/main.py enforces admin auth for every module route;
this file only declares the CRUD. Schemas live here too: the module stays
self-contained (nothing added to app/schemas/).
"""

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.session import get_session
from app.modules.faq import service
from app.modules.faq.models import FaqEntry


class FaqEntryCreate(BaseModel):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    tags: list[str] = []


class FaqEntryUpdate(BaseModel):
    question: str | None = Field(default=None, min_length=1)
    answer: str | None = Field(default=None, min_length=1)
    tags: list[str] | None = None


class FaqEntryResponse(BaseModel):
    id: uuid.UUID
    question: str
    answer: str
    tags: list[str]
    has_embedding: bool  # the vector itself is noise for the panel
    created_at: datetime
    updated_at: datetime


class ReembedResponse(BaseModel):
    reembedded: int


def _to_response(entry: FaqEntry) -> FaqEntryResponse:
    return FaqEntryResponse(
        id=entry.id,
        question=entry.question,
        answer=entry.answer,
        tags=entry.tags,
        has_embedding=entry.embedding is not None,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
    )


async def _get_or_404(session: AsyncSession, entry_id: uuid.UUID) -> FaqEntry:
    entry = await session.get(FaqEntry, entry_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="FAQ entry not found"
        )
    return entry


def register(router: APIRouter) -> None:
    """Module hook target — declares the CRUD on the module's sub-router."""

    @router.get("/entries", response_model=list[FaqEntryResponse])
    async def list_entries(session: AsyncSession = Depends(get_session)):
        result = await session.exec(
            select(FaqEntry).order_by(FaqEntry.created_at.desc())
        )
        return [_to_response(e) for e in result.all()]

    @router.post(
        "/entries",
        response_model=FaqEntryResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_entry(
        payload: FaqEntryCreate, session: AsyncSession = Depends(get_session)
    ):
        entry = await service.create_entry(
            session,
            question=payload.question,
            answer=payload.answer,
            tags=payload.tags,
        )
        await session.commit()
        return _to_response(entry)

    @router.get("/entries/{entry_id}", response_model=FaqEntryResponse)
    async def get_entry(
        entry_id: uuid.UUID, session: AsyncSession = Depends(get_session)
    ):
        return _to_response(await _get_or_404(session, entry_id))

    @router.put("/entries/{entry_id}", response_model=FaqEntryResponse)
    async def update_entry(
        entry_id: uuid.UUID,
        payload: FaqEntryUpdate,
        session: AsyncSession = Depends(get_session),
    ):
        entry = await _get_or_404(session, entry_id)
        entry = await service.update_entry(
            session,
            entry,
            question=payload.question,
            answer=payload.answer,
            tags=payload.tags,
        )
        await session.commit()
        return _to_response(entry)

    @router.delete("/entries/{entry_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_entry(
        entry_id: uuid.UUID, session: AsyncSession = Depends(get_session)
    ):
        entry = await _get_or_404(session, entry_id)
        await session.delete(entry)
        await session.commit()

    @router.post("/reembed", response_model=ReembedResponse)
    async def reembed(session: AsyncSession = Depends(get_session)):
        """Re-embed every entry — for provider/model swaps at the same dim."""
        try:
            count = await service.reembed_all(session)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The embeddings provider failed — try again",
            )
        await session.commit()
        return ReembedResponse(reembedded=count)
