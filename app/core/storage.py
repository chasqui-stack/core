"""Media storage — S3-compatible API via boto3 (ADR-003).

One client covers AWS S3, Cloudflare R2, Spaces, B2 and MinIO; configured
entirely by the STORAGE_* env vars. Storage is OPTIONAL: when unconfigured
every function degrades gracefully (no-op / None) and the stack behaves
exactly as before the storage layer existed.

boto3 is sync — uploads go through `asyncio.to_thread` so the event loop
stays free. `generate_presigned_url` is pure local computation (no network)
and is called synchronously. The client is a lazy module-level singleton so
unconfigured deployments and tests never pay the boto3 import/setup cost.
"""

import asyncio
import logging
import mimetypes
import uuid

from app.core.config import settings

logger = logging.getLogger(__name__)

PRESIGN_EXPIRES_SECONDS = 300

# Object keys for media stored by the ingest pipeline. Anything else in
# messages.media_url (e.g. a tool-produced absolute URL) is not ours to serve.
MEDIA_KEY_PREFIX = "media/"

# Explicit map for the mimes WhatsApp actually sends; mimetypes covers the
# long tail (its jpeg pick varies by platform, hence the override).
_EXT_BY_MIME = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "audio/ogg": "ogg",
    "audio/mpeg": "mp3",
    "audio/mp4": "m4a",
    "video/mp4": "mp4",
    "application/pdf": "pdf",
}

_client = None
_presign_client = None


def is_configured() -> bool:
    return settings.storage_configured


def _build_client(endpoint_url: str | None):
    """Import boto3 only when storage is actually used (lazy singletons)."""
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=settings.storage_access_key,
        aws_secret_access_key=settings.storage_secret_key,
        region_name=settings.storage_region,
        # Path-style keeps MinIO and friends happy; harmless on AWS.
        config=Config(s3={"addressing_style": "path"}),
    )


def _get_client():
    global _client
    if _client is None:
        _client = _build_client(settings.storage_endpoint_url)
    return _client


def _get_presign_client():
    """Presigned URLs embed the signing endpoint — in split-horizon setups
    (docker-compose: core uploads via http://minio:9000, the browser only
    reaches http://localhost:9000) signing must use the PUBLIC endpoint.
    Presigning is local computation, so a second client costs nothing."""
    global _presign_client
    if _presign_client is None:
        public = settings.storage_public_endpoint_url
        if public and public != settings.storage_endpoint_url:
            _presign_client = _build_client(public)
        else:
            _presign_client = _get_client()
    return _presign_client


def ext_for_mime(mime: str) -> str:
    ext = _EXT_BY_MIME.get(mime)
    if ext is None:
        guessed = mimetypes.guess_extension(mime)
        ext = guessed.lstrip(".") if guessed else "bin"
    return ext


def media_key(contact_id: uuid.UUID, message_id: uuid.UUID, mime: str) -> str:
    return f"{MEDIA_KEY_PREFIX}{contact_id}/{message_id}.{ext_for_mime(mime)}"


def is_media_key(value: str | None) -> bool:
    return bool(value and value.startswith(MEDIA_KEY_PREFIX))


async def put_media(key: str, data: bytes, content_type: str) -> None:
    """Upload one object. Raises on failure — callers decide the fallback."""
    client = _get_client()
    await asyncio.to_thread(
        client.put_object,
        Bucket=settings.storage_bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
    )


def presigned_get(key: str, expires: int = PRESIGN_EXPIRES_SECONDS) -> str:
    """Short-lived GET URL — local computation, no network round-trip."""
    client = _get_presign_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.storage_bucket, "Key": key},
        ExpiresIn=expires,
    )
