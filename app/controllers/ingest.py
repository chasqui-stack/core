"""Canonical entry point: POST /ingest (ARCHITECTURE §5).

The only door into the agent. Gateways (whatsapp today, web/telegram tomorrow)
POST the canonical payload here and render the canonical reply on their channel.

Auth: a shared secret (`INTERNAL_API_KEY`, sent as `X-Internal-API-Key`).
When unset (local dev), the check is skipped — set it in any deployment.
"""

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import settings
from app.db.session import get_session
from app.schemas.ingest import IngestRequest, IngestResponse
from app.services.ingest_service import handle_ingest


async def verify_internal_key(
    x_internal_api_key: str | None = Header(default=None),
) -> None:
    """Gateway↔core shared secret. No-op when INTERNAL_API_KEY is unset."""
    if settings.internal_api_key and x_internal_api_key != settings.internal_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal API key",
        )


router = APIRouter(dependencies=[Depends(verify_internal_key)])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    session: AsyncSession = Depends(get_session),
) -> IngestResponse:
    """Receive one canonical inbound message and return the agent's reply."""
    return await handle_ingest(session, request)
