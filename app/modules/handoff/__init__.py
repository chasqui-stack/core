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
    """Deriva la conversación a un agente humano.

    Usa esta herramienta cuando:
    - El usuario pide explícitamente hablar con una persona.
    - El usuario está molesto o frustrado y no logras ayudarlo.
    - El caso requiere acciones que no puedes realizar con tus herramientas.

    Args:
        reason: Motivo breve de la derivación (ej: "solicita asesor de ventas").
    """
    conversation = await _get_conversation(runtime)
    # Reassign (don't mutate) so SQLAlchemy detects the JSONB change
    conversation.conversation_state = {
        **conversation.conversation_state,
        "handoff": {"requested": True, "reason": reason, "at": _now_iso()},
    }
    runtime.context.session.add(conversation)
    return (
        "Conversación marcada para atención humana. Informa al usuario que "
        "una persona del equipo lo contactará pronto por este mismo chat."
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
    """Registra al usuario como lead (cliente potencial interesado).

    Usa esta herramienta cuando el usuario muestre intención de compra o
    pida que lo contacten, y ya tengas al menos su nombre y qué le interesa.
    Pide los datos que falten de forma natural antes de llamarla.

    Args:
        name: Nombre del usuario.
        interest: Producto/servicio que le interesa.
        phone: Teléfono de contacto (opcional).
        email: Correo de contacto (opcional).
        notes: Contexto adicional útil para el equipo comercial (opcional).
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
    return "Lead registrado. Agradece al usuario y confirma que el equipo lo contactará."


class HandoffModule:
    """Human handoff + lead capture (writes to conversation_state)."""

    name = "handoff"

    def register_tools(self):
        return [human_handoff, lead_capture]


module = HandoffModule()
