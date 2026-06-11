"""Handoff notifications (ADR-004) — webhook and/or SMTP email, best-effort.

Both channels are optional (.env) and independent: each failure logs and is
swallowed (the embeddings/storage posture — alerting must never break a
turn). `dispatch_handoff()` fires them as a background task so the agent
turn never waits on Slack or an SMTP handshake.

SMTP is stdlib `smtplib` on purpose (zero dependencies): a handoff alert is
one short text message, and SMTP is the universal seam — Brevo, Mailgun,
SES or a Gmail app password all speak it (587 STARTTLS / 465 implicit SSL).
"""

import asyncio
import logging
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage

import httpx

from app.core.config import settings
from app.models import Contact

logger = logging.getLogger(__name__)

WEBHOOK_TIMEOUT_SECONDS = 10.0

# Keep background tasks referenced until done (asyncio drops weak refs)
_pending: set[asyncio.Task] = set()


def build_handoff_event(contact: Contact, reason: str) -> dict:
    """The JSON payload POSTed to NOTIFY_WEBHOOK_URL (and the email source)."""
    return {
        "event": "handoff",
        "reason": reason,
        "at": datetime.now(timezone.utc).isoformat(),
        "contact": {
            "id": str(contact.id),
            "channel": contact.channel,
            "external_id": contact.external_id,
            "wa_id": contact.wa_id,
            "display_name": contact.display_name,
        },
    }


async def send_webhook(event: dict) -> None:
    """POST the handoff event to NOTIFY_WEBHOOK_URL. Raises on failure."""
    async with httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT_SECONDS) as client:
        response = await client.post(settings.notify_webhook_url, json=event)
        response.raise_for_status()


def _send_email_sync(event: dict) -> None:
    """One plain-text email through the configured SMTP relay (blocking)."""
    contact = event["contact"]
    who = contact["display_name"] or contact["wa_id"] or contact["external_id"]

    message = EmailMessage()
    message["Subject"] = f"[Chasqui] Human attention needed: {who}"
    message["From"] = settings.smtp_from
    message["To"] = ", ".join(
        addr.strip() for addr in settings.notify_email_to.split(",") if addr.strip()
    )
    message.set_content(
        "A conversation was handed off to a human.\n\n"
        f"Contact: {who} ({contact['channel']})\n"
        f"Reason: {event['reason']}\n"
        f"At: {event['at']}\n\n"
        "Open the admin panel to take over the conversation."
    )

    if settings.smtp_port == 465:
        server: smtplib.SMTP = smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=15)
    else:
        server = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15)
    try:
        if settings.smtp_port != 465:
            server.starttls()
        if settings.smtp_user and settings.smtp_password:
            server.login(settings.smtp_user, settings.smtp_password)
        server.send_message(message)
    finally:
        server.quit()


async def send_email(event: dict) -> None:
    """Email the handoff event via SMTP (off the event loop). Raises on failure."""
    await asyncio.to_thread(_send_email_sync, event)


async def _notify(event: dict) -> None:
    """Run every configured sender; each one fails independently."""
    if settings.notify_webhook_url:
        try:
            await send_webhook(event)
        except Exception:
            logger.exception("Handoff webhook notification failed")
    if settings.smtp_configured:
        try:
            await send_email(event)
        except Exception:
            logger.exception("Handoff email notification failed")


def dispatch_handoff(contact: Contact, reason: str) -> None:
    """Fire-and-forget the handoff notifications (never blocks the turn)."""
    if not settings.notify_webhook_url and not settings.smtp_configured:
        return
    task = asyncio.create_task(_notify(build_handoff_event(contact, reason)))
    _pending.add(task)
    task.add_done_callback(_pending.discard)
