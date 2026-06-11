"""Canonical outbound seam (ADR-004) — core → channel gateway `POST /send`.

The mirror of /ingest: the core resolves WHERE to send from .env
(`CHANNEL_<channel>_SEND_URL`, one var per channel) and speaks the canonical
shapes only — it never imports a channel SDK. Gateway errors carry a `code`
(`NO_WA_ID`, `WINDOW_EXPIRED`, `SEND_FAILED`, ...) that flows through to the
admin so the panel can explain WhatsApp's 24h window instead of a bare 502.
"""

import logging

import httpx

from app.core.config import settings
from app.models import Contact

logger = logging.getLogger(__name__)

SEND_TIMEOUT_SECONDS = 20.0


class ChannelSendError(Exception):
    """A gateway send that didn't happen — `code` is part of the contract."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def send_url_for(channel: str) -> str | None:
    """Resolve the gateway /send URL for a channel from settings.

    Convention: a `channel_<name>_send_url` settings field per channel —
    adding a channel is one .env var, zero core code paths.
    """
    return getattr(settings, f"channel_{channel}_send_url", None)


def _error_from_response(response: httpx.Response) -> ChannelSendError:
    code, message = "SEND_FAILED", f"Gateway send failed (HTTP {response.status_code})"
    try:
        detail = response.json().get("detail")
        if isinstance(detail, dict):
            code = detail.get("code", code)
            message = detail.get("message", message)
    except Exception:  # non-JSON error body — keep the generic mapping
        pass
    return ChannelSendError(code, message)


async def send_text(contact: Contact, text: str) -> dict:
    """POST one canonical text message through the contact's channel gateway.

    Returns the gateway's success body; raises ChannelSendError otherwise.
    """
    url = send_url_for(contact.channel)
    if not url:
        raise ChannelSendError(
            "CHANNEL_NOT_CONFIGURED",
            f"No send URL configured for channel '{contact.channel}' "
            f"(set CHANNEL_{contact.channel.upper()}_SEND_URL)",
        )

    payload = {
        "contact": {
            "channel": contact.channel,
            "external_id": contact.external_id,
            "wa_id": contact.wa_id,
        },
        "message": {"type": "text", "text": text},
    }
    headers = (
        {"X-Internal-API-Key": settings.internal_api_key}
        if settings.internal_api_key
        else {}
    )

    try:
        async with httpx.AsyncClient(timeout=SEND_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        logger.error("Gateway unreachable for channel '%s': %s", contact.channel, exc)
        raise ChannelSendError(
            "GATEWAY_UNREACHABLE", f"Could not reach the {contact.channel} gateway"
        ) from exc

    if response.status_code >= 400:
        raise _error_from_response(response)
    return response.json()
