"""Canonical entry point: POST /ingest (ARCHITECTURE §5).

The only door into the agent. Gateways (whatsapp today, web/telegram tomorrow)
POST the canonical payload here and render the canonical reply on their channel.

NOTE: gateway↔core auth (shared secret header) lands with the e2e wiring in
Sprint 2 — both services are private/localhost until then.
"""

from fastapi import APIRouter, Depends
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.session import get_session
from app.schemas.ingest import IngestRequest, IngestResponse
from app.services.ingest_service import handle_ingest

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    session: AsyncSession = Depends(get_session),
) -> IngestResponse:
    """Receive one canonical inbound message and return the agent's reply."""
    return await handle_ingest(session, request)
