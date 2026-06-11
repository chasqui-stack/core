"""Canonical entry point: POST /ingest (ARCHITECTURE §5).

The only door into the agent. Gateways (whatsapp today, web/telegram tomorrow)
POST the canonical payload here and render the canonical reply on their channel.

Auth: a shared secret (`INTERNAL_API_KEY`, sent as `X-Internal-API-Key`).
When unset (local dev), the check is skipped — set it in any deployment.
"""

import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import settings
from app.db.session import get_session
from app.models import Message
from app.schemas.ingest import ChannelStatusUpdate, IngestRequest, IngestResponse
from app.services.ingest_service import handle_ingest

logger = logging.getLogger(__name__)


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


@router.post("/channel/status", status_code=status.HTTP_204_NO_CONTENT)
async def channel_status(
    update: ChannelStatusUpdate,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Async delivery status from a gateway (ADR-004) — e.g. Meta rejecting
    an accepted send minutes later. Operator messages persist the channel id
    in meta.wamid; statuses for ids we never stored (agent replies sent via
    update.reply) are acknowledged and dropped."""
    result = await session.exec(
        select(Message).where(col(Message.meta)["wamid"].astext == update.message_id)
    )
    message = result.first()
    if message is None:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # Reassign (don't mutate) so SQLAlchemy detects the JSONB change
    message.meta = {
        **message.meta,
        "delivery_status": update.status,
        "delivery_code": update.code,
        "delivery_detail": update.detail,
    }
    session.add(message)
    await session.commit()
    logger.warning(
        "Outbound message %s marked %s by the gateway: %s %s",
        message.id, update.status, update.code, update.detail,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
