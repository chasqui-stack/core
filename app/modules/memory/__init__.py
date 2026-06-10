"""Memory module — the agent saves durable facts about the contact.

Extraction-by-tool: instead of a second LLM pass after every turn, the
model calls `save_memory` silently when the user shares something worth
remembering (proven pattern). Retrieval happens before the turn in
`memory_service.retrieve_relevant` (pgvector).
"""

import logging

from langchain.tools import ToolRuntime, tool

from app.models import Memory
from app.services.agent_context import TurnContext

logger = logging.getLogger(__name__)


@tool
async def save_memory(content: str, runtime: ToolRuntime[TurnContext]) -> str:
    """Guarda un dato duradero sobre el usuario para futuras conversaciones.

    Usa esta herramienta SILENCIOSAMENTE cuando el usuario comparta
    información estable y útil: su nombre, preferencias, contexto personal
    o laboral, decisiones tomadas. NO confirmes al usuario que guardaste algo.

    NO guardes información trivial o pasajera (saludos, el mensaje literal).

    Args:
        content: Una oración concisa y objetiva.
            Ej: "Prefiere que lo contacten por las tardes".
    """
    ctx = runtime.context

    embedding = None
    try:
        from app.core.embeddings import get_embeddings

        embedding = await get_embeddings().aembed_query(content)
    except Exception:  # pragma: no cover - network/key issues must not kill the turn
        logger.warning("Embedding failed; saving memory without vector", exc_info=True)

    ctx.session.add(
        Memory(contact_id=ctx.contact_id, content=content, embedding=embedding)
    )
    return "Dato guardado. No lo menciones al usuario."


class MemoryModule:
    """Long-term memory writing (retrieval is wired in the orchestrator)."""

    name = "memory"

    def register_tools(self):
        return [save_memory]


module = MemoryModule()
