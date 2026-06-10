"""Serve stored media to the admin panel (Sprint 6, ADR-003).

Returns a short-lived presigned URL as JSON — never a redirect: `<img src>`
cannot send the JWT header, so the SPA fetches `{url}` with axios and feeds
the presigned URL (straight to the bucket) to `<img>`/`<audio>`. The bucket
stays private and the panel never holds storage credentials.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core import storage
from app.db.session import get_session
from app.models import Message
from app.schemas.admin_contacts import MediaUrlResponse

router = APIRouter()


@router.get("/{message_id}", response_model=MediaUrlResponse)
async def get_media_url(
    message_id: uuid.UUID, session: AsyncSession = Depends(get_session)
):
    message = await session.get(Message, message_id)
    if message is None or not storage.is_media_key(message.media_url):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No stored media"
        )
    if not storage.is_configured():
        # The key exists but this deployment lost its storage config.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Storage not configured",
        )
    url = storage.presigned_get(message.media_url)
    return MediaUrlResponse(url=url, expires_in=storage.PRESIGN_EXPIRES_SECONDS)
