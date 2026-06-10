"""FAQ module — grounded search over the admin-managed Q&A knowledge base.

The proof-of-fire of the full module contract (ARCHITECTURE §8): tools +
models + admin routes + config schema, all in this folder. RAG: entries are
embedded on save (service.py) and retrieved by cosine similarity with a
threshold — below it the tool answers honestly instead of letting the model
guess.
"""

import logging

from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, Field

from app.services.agent_context import TurnContext

from app.modules.faq.models import FaqEntry  # noqa: F401 — lands the table in metadata
from app.modules.faq import service

logger = logging.getLogger(__name__)


class FaqSearchConfig(BaseModel):
    """Knobs surfaced in the admin panel (agent_config.tool_config['faq_search'])."""

    top_k: int = Field(default=4, ge=1, le=20, description="Maximum number of results")
    min_similarity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for a result to count as relevant",
    )


def _tool_config(ctx: TurnContext) -> FaqSearchConfig:
    raw = (ctx.config.tool_config or {}).get("faq_search", {})
    try:
        return FaqSearchConfig(**raw)
    except Exception:  # bad admin-entered config must not break the turn
        logger.warning("Invalid faq_search tool_config %r; using defaults", raw)
        return FaqSearchConfig()


NO_RESULTS = (
    "No information about this was found in the knowledge base. "
    "Honestly tell the user you don't have that information — do NOT make up an answer."
)


@tool
async def faq_search(query: str, runtime: ToolRuntime[TurnContext]) -> str:
    """Search the company's knowledge base (frequently asked questions).

    ALWAYS use this tool when the user asks about business-specific
    information: products, services, prices, schedules, policies,
    locations, etc. Answer ONLY with what the tool returns.

    Args:
        query: Key concepts of what the user needs to know
            (e.g. "opening hours", "return policy").
    """
    ctx = runtime.context
    config = _tool_config(ctx)

    hits = await service.search(
        ctx.session,
        query,
        top_k=config.top_k,
        min_similarity=config.min_similarity,
    )
    if not hits:
        return NO_RESULTS

    snippets = "\n\n".join(
        f"[{i}] Q: {entry.question}\nA: {entry.answer}"
        for i, (entry, _similarity) in enumerate(hits, start=1)
    )
    return (
        "Knowledge base information (base your answer ONLY on this):\n\n"
        f"{snippets}"
    )


class FaqModule:
    """FAQ knowledge-base: Q&A entries + grounded retrieval."""

    name = "faq"

    def register_tools(self):
        return [faq_search]

    def register_models(self):
        return [FaqEntry]

    def register_admin_routes(self, router):
        from app.modules.faq.admin import register

        register(router)

    def config_schema(self):
        return FaqSearchConfig


module = FaqModule()
