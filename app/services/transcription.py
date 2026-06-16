"""Speech-to-text fallback — transcribe inbound audio when the LLM can't hear.

ADR-010. A voice note arrives as an `audio` message (bytes inlined as a `data:`
URI). Native-audio models (Gemini) get the audio directly; every other model
(`caps.audio = False`) would otherwise dead-end in the orchestrator's graceful
"ask them to type it" fallback. When STT is configured this module transcribes
the audio first, so the turn proceeds as if the user had typed it.

Design (ADR-010):
- **OpenAI-compatible** `POST {base_url}/audio/transcriptions` (multipart). The
  shape is a de-facto standard — OpenAI and Groq are byte-identical — so one
  `httpx` client serves any compatible host by swapping `STT_BASE_URL`.
- **Groq `whisper-large-v3-turbo`** is the default: it accepts OGG/Opus natively
  (WhatsApp/Telegram voice) so there is no transcoding step, and it's cheapest.
- **Best-effort:** any failure returns None and the caller keeps today's graceful
  text fallback. STT never breaks a turn.
- httpx only — no provider SDK (httpx is already a core dependency).
"""

import logging

import httpx

from app.core import storage
from app.core.config import settings

logger = logging.getLogger(__name__)

# provider → default base_url (an explicit STT_BASE_URL overrides these).
_DEFAULT_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openai": "https://api.openai.com/v1",
}


def stt_enabled() -> bool:
    """True when a provider + key are set and a base_url can be resolved."""
    return settings.stt_configured and bool(_base_url())


def _base_url() -> str:
    return (
        settings.stt_base_url or _DEFAULT_BASE_URLS.get(settings.stt_provider, "")
    ).rstrip("/")


async def transcribe(audio: bytes, mime: str) -> str | None:
    """Transcribe audio bytes to text. None on any failure (caller falls back).

    OGG/Opus is sent as-is — the default provider (Groq) accepts it natively, so
    there is no ffmpeg/transcoding step (ADR-010).
    """
    if not stt_enabled():
        return None
    if len(audio) > settings.stt_max_bytes:
        logger.warning(
            "Inbound audio is %d bytes (> STT cap %d) — skipping transcription",
            len(audio),
            settings.stt_max_bytes,
        )
        return None

    ext = storage.ext_for_mime(mime) or "ogg"
    files = {"file": (f"audio.{ext}", audio, mime)}
    data = {"model": settings.stt_model, "response_format": "text"}
    if settings.stt_language:
        data["language"] = settings.stt_language
    headers = {"Authorization": f"Bearer {settings.stt_api_key}"}

    try:
        async with httpx.AsyncClient(timeout=settings.stt_timeout_seconds) as client:
            response = await client.post(
                f"{_base_url()}/audio/transcriptions",
                data=data,
                files=files,
                headers=headers,
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("STT request failed (%s) — text fallback", exc)
        return None

    # response_format=text → plain-text body (not JSON).
    transcript = response.text.strip()
    return transcript or None
