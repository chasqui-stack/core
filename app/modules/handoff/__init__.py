"""Handoff module — flag the conversation for a human / capture a lead.

Both tools persist into `conversations.conversation_state` (JSONB), which
the admin panel surfaces in Sprint 5. Shows the ToolRuntime pattern: tools
reach the DB session and conversation through `runtime.context`.
"""

from datetime import datetime, timezone

from langchain.tools import ToolRuntime, tool

from app.models import Conversation
from app.services.agent_context import TurnContext


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
    # Reassign (don't mutate) so SQLAlchemy detects the JSONB change
    conversation.conversation_state = {
        **conversation.conversation_state,
        "handoff": {"requested": True, "reason": reason, "at": _now_iso()},
    }
    runtime.context.session.add(conversation)
    return (
        "Conversation flagged for human attention. Tell the user someone "
        "from the team will reach out soon through this same chat."
    )


@tool
async def lead_capture(
    name: str,
    interest: str,
    runtime: ToolRuntime[TurnContext],
    phone: str | None = None,
    email: str | None = None,
    notes: str | None = None,
) -> str:
    """Register the user as a lead (interested potential customer).

    Use this tool when the user shows purchase intent or asks to be
    contacted, and you already have at least their name and interest.
    Naturally ask for any missing details before calling it.

    Args:
        name: The user's name.
        interest: Product/service they are interested in.
        phone: Contact phone (optional).
        email: Contact email (optional).
        notes: Extra context useful for the sales team (optional).
    """
    conversation = await _get_conversation(runtime)
    lead = {
        "name": name,
        "interest": interest,
        "phone": phone,
        "email": email,
        "notes": notes,
        "at": _now_iso(),
    }
    state = conversation.conversation_state
    conversation.conversation_state = {
        **state,
        "leads": [*state.get("leads", []), lead],
    }
    runtime.context.session.add(conversation)
    return "Lead saved. Thank the user and confirm the team will contact them."


class HandoffModule:
    """Human handoff + lead capture (writes to conversation_state)."""

    name = "handoff"

    def register_tools(self):
        return [human_handoff, lead_capture]


module = HandoffModule()
