"""Handoff module — hand the conversation to a human / capture a lead.

Sprint 7 (ADR-004) made both tools real: `human_handoff` flips
`conversations.mode` to "human" (the ingest pipeline then silences the
agent) and fires the configured notifications; `lead_capture` writes a row
in the module-owned `leads` table, with operator-configurable required and
extra fields (the Sprint 5 auto-form contract).

The mode flip takes effect on the NEXT inbound message — the current turn
still replies, which is exactly right: the bot confirms the handoff, then
goes quiet.
"""

import logging
from datetime import datetime, timezone

from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, Field

from app.models import Contact, Conversation
from app.services import notify_service
from app.services.agent_context import TurnContext

from app.modules.handoff.models import Lead  # noqa: F401 — lands the table in metadata

logger = logging.getLogger(__name__)


class LeadCaptureConfig(BaseModel):
    """Knobs surfaced in the admin panel (agent_config.tool_config['lead_capture'])."""

    require_email: bool = Field(
        default=True, description="Require an email address before saving a lead"
    )
    require_phone: bool = Field(
        default=True,
        description=(
            "Require a phone number before saving a lead "
            "(the contact's known WhatsApp number counts)"
        ),
    )
    extra_fields: str = Field(
        default="",
        description=(
            'Comma-separated extra questions to collect (e.g. "company, city") — '
            "stored with the lead"
        ),
    )


def _config(ctx: TurnContext) -> LeadCaptureConfig:
    raw = (ctx.config.tool_config or {}).get("lead_capture", {})
    try:
        return LeadCaptureConfig(**raw)
    except Exception:  # bad admin-entered config must not break the turn
        logger.warning("Invalid lead_capture tool_config %r; using defaults", raw)
        return LeadCaptureConfig()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _get_conversation(runtime: ToolRuntime[TurnContext]) -> Conversation:
    ctx = runtime.context
    conversation = await ctx.session.get(Conversation, ctx.conversation_id)
    assert conversation is not None  # the turn always runs inside one
    return conversation


@tool
async def human_handoff(reason: str, runtime: ToolRuntime[TurnContext]) -> str:
    """Hand the conversation off to a human agent.

    Use this tool when:
    - The user explicitly asks to talk to a person.
    - The user is upset or frustrated and you cannot help them.
    - The case requires actions your tools cannot perform.

    Args:
        reason: Brief handoff reason (e.g. "asks for a sales rep").
    """
    conversation = await _get_conversation(runtime)
    conversation.mode = "human"  # ingest silences the agent from the next message on
    # Reassign (don't mutate) so SQLAlchemy detects the JSONB change
    conversation.conversation_state = {
        **conversation.conversation_state,
        "handoff": {"requested": True, "reason": reason, "at": _now_iso()},
    }
    runtime.context.session.add(conversation)

    contact = await runtime.context.session.get(Contact, runtime.context.contact_id)
    if contact is not None:
        notify_service.dispatch_handoff(contact, reason)

    return (
        "Conversation handed off: a human now owns this thread and you will "
        "go silent after this reply. Tell the user someone from the team "
        "will continue the conversation right here shortly."
    )


@tool
async def lead_capture(
    name: str,
    interest: str,
    runtime: ToolRuntime[TurnContext],
    phone: str | None = None,
    email: str | None = None,
    notes: str | None = None,
    extra: dict | None = None,
) -> str:
    """Register the user as a lead (interested potential customer).

    Use this tool when the user shows purchase intent or asks to be
    contacted. Call it as soon as you have their name and interest — if
    required information is still missing, the tool will tell you exactly
    what to ask for; collect it naturally in conversation and call again.

    Args:
        name: The user's name.
        interest: Product/service they are interested in.
        phone: Contact phone (optional — known WhatsApp numbers are used
            automatically).
        email: Contact email (optional unless configured as required).
        notes: Extra context useful for the sales team (optional).
        extra: Answers to any extra questions the tool asked for, as
            key-value pairs (e.g. {"company": "ACME"}).
    """
    ctx = runtime.context
    config = _config(ctx)
    contact = await ctx.session.get(Contact, ctx.contact_id)

    phone_value = phone or (contact.wa_id if contact else None)
    extra = extra or {}
    wanted = [f.strip() for f in config.extra_fields.split(",") if f.strip()]

    missing: list[str] = []
    if config.require_email and not email:
        missing.append("email")
    if config.require_phone and not phone_value:
        missing.append("phone")
    missing += [f for f in wanted if not str(extra.get(f, "") or "").strip()]

    if missing:
        return (
            "Lead NOT saved yet — required information is missing: "
            f"{', '.join(missing)}. Ask the user for it naturally (one thing "
            "at a time), then call this tool again with everything, passing "
            "extra answers in the `extra` argument."
        )

    ctx.session.add(
        Lead(
            contact_id=ctx.contact_id,
            name=name,
            interest=interest,
            email=email,
            phone=phone_value,
            notes=notes,
            extra={k: str(v) for k, v in extra.items() if str(v or "").strip()},
        )
    )
    return "Lead saved. Thank the user and confirm the team will contact them."


class HandoffModule:
    """Human handoff (mode flip + notify) + lead capture (leads table)."""

    name = "handoff"
    config_key = "lead_capture"  # where the knobs live in agent_config.tool_config

    def register_tools(self):
        return [human_handoff, lead_capture]

    def register_models(self):
        return [Lead]

    def register_admin_routes(self, router):
        from app.modules.handoff.admin import register

        register(router)

    def config_schema(self):
        return LeadCaptureConfig


module = HandoffModule()
