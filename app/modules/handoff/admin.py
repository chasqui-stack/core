"""Handoff admin endpoints — mounted under /admin/modules/handoff (JWT).

Leads listing for the panel's /leads page. Read-only: leads are created by
the agent (lead_capture tool); operators consume them.
"""

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.session import get_session
from app.models import Contact
from app.modules.handoff.models import Lead


class LeadItem(BaseModel):
    id: uuid.UUID
    contact_id: uuid.UUID
    contact_display_name: str | None
    name: str
    interest: str | None
    email: str | None
    phone: str | None
    notes: str | None
    extra: dict
    created_at: datetime


class LeadListResponse(BaseModel):
    items: list[LeadItem]
    total: int


def register(router: APIRouter) -> None:
    """Module hook target — declares the endpoints on the module's sub-router."""

    @router.get("/leads", response_model=LeadListResponse)
    async def list_leads(
        session: AsyncSession = Depends(get_session),
        limit: int = Query(default=25, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
        contact_id: uuid.UUID | None = Query(default=None),
    ):
        base = select(Lead, Contact.display_name).join(
            Contact, col(Lead.contact_id) == col(Contact.id)
        )
        count = select(func.count()).select_from(Lead)
        if contact_id is not None:
            base = base.where(Lead.contact_id == contact_id)
            count = count.where(Lead.contact_id == contact_id)

        total = (await session.exec(count)).one()
        rows = (
            await session.exec(
                base.order_by(col(Lead.created_at).desc()).limit(limit).offset(offset)
            )
        ).all()

        items = [
            LeadItem(
                id=lead.id,
                contact_id=lead.contact_id,
                contact_display_name=display_name,
                name=lead.name,
                interest=lead.interest,
                email=lead.email,
                phone=lead.phone,
                notes=lead.notes,
                extra=lead.extra,
                created_at=lead.created_at,
            )
            for lead, display_name in rows
        ]
        return LeadListResponse(items=items, total=total)
