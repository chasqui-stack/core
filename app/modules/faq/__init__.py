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

from app.models import AgentConfig
from app.modules import registry
from app.services import agent_config_service
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
    inject_question_index: bool = Field(
        default=False,
        description=(
            "Publish the FAQ question index in the system prompt every turn "
            "so the model knows what it can look up (costs tokens per turn)"
        ),
    )
    question_index_max: int = Field(
        default=40,
        ge=1,
        le=200,
        description="Skip the index entirely when the FAQ has more entries than this",
    )


def _tool_config(config: AgentConfig) -> FaqSearchConfig:
    raw = (config.tool_config or {}).get("faq_search", {})
    try:
        return FaqSearchConfig(**raw)
    except Exception:  # bad admin-entered config must not break the turn
        logger.warning("Invalid faq_search tool_config %r; using defaults", raw)
        return FaqSearchConfig()


NO_RESULTS = (
    "No information about this was found in the knowledge base. "
    "Honestly tell the user you don't have that information — do NOT make up an answer."
)

# A miss here is not the end of the road when the documents retriever is on:
# the model otherwise reads NO_RESULTS as "stop" and never tries the sibling.
NO_RESULTS_TRY_DOCUMENTS = (
    "No matching FAQ entry was found. The answer may be inside the uploaded "
    "documents: if you have not already, call search_documents with the same "
    "query before replying. Only if that also finds nothing, honestly tell the "
    "user you don't have that information — do NOT make up an answer."
)


# Hits can be near-misses (similar wording, different topic) — same handover.
HITS_TRY_SIBLING = (
    "\n\nIf none of this answers the question and you have not already, call "
    "search_documents with the same query before replying — it searches the uploaded documents."
)


def _sibling_on(config: AgentConfig) -> bool:
    sibling = "search_documents"
    return registry.has_tool(sibling) and agent_config_service.tool_enabled(
        config, sibling
    )


@tool
async def faq_search(query: str, runtime: ToolRuntime[TurnContext]) -> str:
    """Search the operator-curated FAQ of this specific business or project.

    Short official answers. Call this for general questions about the business — schedules,
    location, contact, payments, shipping, policies, concepts, terminology
    and other common questions. You do NOT know these details from
    training; never answer them from memory. Answer ONLY with what the
    tool returns. For details that live inside uploaded documents (product
    specs, price lists, manuals, step-by-step procedures, contracts), use
    search_documents instead; when unsure which one holds the answer, call
    both.

    Args:
        query: Key concepts of what the user needs to know
            (e.g. "opening hours", "return policy").
    """
    ctx = runtime.context
    config = _tool_config(ctx.config)

    hits = await service.search(
        ctx.session,
        query,
        top_k=config.top_k,
        min_similarity=config.min_similarity,
    )
    if not hits:
        return NO_RESULTS_TRY_DOCUMENTS if _sibling_on(ctx.config) else NO_RESULTS

    snippets = "\n\n".join(
        f"[{i}] Q: {entry.question}\nA: {entry.answer}"
        for i, (entry, _similarity) in enumerate(hits, start=1)
    )
    return (
        "Knowledge base information (base your answer ONLY on this):\n\n"
        f"{snippets}"
    ) + (HITS_TRY_SIBLING if _sibling_on(ctx.config) else "")


# Over-cap warning fires once, not per turn (reset if the count drops back).
_index_cap_warned = False


class FaqModule:
    """FAQ knowledge-base: Q&A entries + grounded retrieval."""

    name = "faq"
    config_key = "faq_search"  # where the knobs live in agent_config.tool_config

    def register_tools(self):
        return [faq_search]

    async def system_prompt_fragment(self, context, query: str) -> str | None:
        """The FAQ question index — a table of contents of what's look-up-able.

        Opt-in (`inject_question_index`, ADR-012): it costs tokens on every
        turn. Framing is load-bearing — the list must read as "call the tool
        for these", never "you already know these" (that wording is the bug
        chasqui#30 opens with). Questions only, never answers.
        """
        global _index_cap_warned
        config = _tool_config(context.config)
        if not config.inject_question_index:
            return None
        if not agent_config_service.tool_enabled(context.config, "faq_search"):
            return None  # advertising a disabled tool would misroute the model

        questions = await service.list_questions(context.session)
        if not questions:
            return None
        if len(questions) > config.question_index_max:
            # A silently truncated list is worse than none — the model would
            # treat it as exhaustive and refuse what's not on it.
            if not _index_cap_warned:
                logger.warning(
                    "FAQ question index skipped: %d entries exceed "
                    "question_index_max=%d — raise the cap or disable "
                    "inject_question_index",
                    len(questions),
                    config.question_index_max,
                )
                _index_cap_warned = True
            return None
        _index_cap_warned = False

        lines = "\n".join(f"- {q}" for q in questions)
        return (
            "The knowledge base can answer the questions listed below. When "
            "the user asks about any of them — or anything similar — call "
            "`faq_search` to retrieve the answer. These are things you can "
            "LOOK UP, not things you know: never answer them from memory.\n"
            f"{lines}"
        )

    def register_models(self):
        return [FaqEntry]

    def register_admin_routes(self, router):
        from app.modules.faq.admin import register

        register(router)

    def config_schema(self):
        return FaqSearchConfig


module = FaqModule()
