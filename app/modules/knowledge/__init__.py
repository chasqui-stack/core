"""Knowledge module — Document-RAG over uploaded files.

Operators upload documents (pdf, docx, txt, md, html); the module extracts
their text, splits it into overlapping chunks and embeds each chunk into
pgvector, tracking per-document state (`pending → processing → ready | error`).
The agent reaches them on demand through `search_documents`.

Coexists with the FAQ module: FAQ is curated, atomic Q&A; this is bulk
reference material (manuals, catalogs, policies). The two tool docstrings
carry the boundary — each one names what the OTHER tool is for — so the
model routes between them without system-prompt surgery.
"""

import logging

from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, Field

from app.models import AgentConfig
from app.modules import registry
from app.services import agent_config_service
from app.services.agent_context import TurnContext

# Imported at module level so the tables reach SQLModel.metadata on discover()
from app.modules.knowledge.models import Document, DocumentChunk
from app.modules.knowledge import service

logger = logging.getLogger(__name__)

class KnowledgeSearchConfig(BaseModel):
    """Knobs surfaced in the admin panel (agent_config.tool_config['document_search'])."""

    top_k: int = Field(default=4, ge=1, le=20, description="Maximum number of passages")
    min_similarity: float = Field(
        # Raw chunks score lower than curated Q&A pairs — hence below faq's 0.5
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for a passage to count as relevant",
    )
    inject_document_index: bool = Field(
        default=False,
        description=(
            "Publish the list of uploaded documents in the system prompt every "
            "turn so the model knows what it can search (costs tokens per turn)"
        ),
    )
    document_index_max: int = Field(
        default=40,
        ge=1,
        le=200,
        description="Skip the index entirely when there are more documents than this",
    )


def _tool_config(config: AgentConfig) -> KnowledgeSearchConfig:
    raw = (config.tool_config or {}).get("document_search", {})
    try:
        return KnowledgeSearchConfig(**raw)
    except Exception:  # bad admin-entered config must not break the turn
        logger.warning("Invalid document_search tool_config %r; using defaults", raw)
        return KnowledgeSearchConfig()


NO_RESULTS = (
    "No relevant passages were found in the uploaded documents. "
    "Honestly tell the user you don't have that information — do NOT make up an answer."
)

# Mirror of faq's: a miss hands over to the sibling retriever when it is on.
NO_RESULTS_TRY_FAQ = (
    "No relevant passages were found in the uploaded documents. The answer may "
    "be in the curated FAQ: if you have not already, call faq_search with the "
    "same query before replying. Only if that also finds nothing, honestly tell "
    "the user you don't have that information — do NOT make up an answer."
)


# Hits can be near-misses (similar wording, different topic) — same handover.
HITS_TRY_SIBLING = (
    "\n\nIf none of this answers the question and you have not already, call "
    "faq_search with the same query before replying — it searches the curated FAQ."
)


def _sibling_on(config: AgentConfig) -> bool:
    sibling = "faq_search"
    return registry.has_tool(sibling) and agent_config_service.tool_enabled(
        config, sibling
    )


@tool
async def search_documents(query: str, runtime: ToolRuntime[TurnContext]) -> str:
    """Search the documents the operator uploaded (manuals, catalogs, price lists).

    Returns the most relevant passages of those files (guides and contracts
    too). Call this for anything a file would hold: a specific product's price,
    specs, warranty, care or usage instructions, step-by-step procedures,
    detailed terms. You do NOT know these details from training; never
    answer them from memory. Answer ONLY with what the tool returns. For
    general questions about the business (schedules, location, contact,
    payments, shipping, policies), use faq_search instead; when unsure
    which one holds the answer, call both.

    Args:
        query: Key concepts of what the user needs to know, not their
            literal message (e.g. "warranty coverage", "installation steps").
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
        return NO_RESULTS_TRY_FAQ if _sibling_on(ctx.config) else NO_RESULTS

    # Whole chunks: they are bounded by construction (service.CHUNK_SIZE), and
    # truncating them drops the answer whenever it sits in the second half.
    passages = "\n\n".join(
        f"[{document.filename}] {chunk.content.strip()}"
        for chunk, document, _similarity in hits
    )
    return (
        "Relevant passages from the uploaded documents "
        "(base your answer ONLY on this):\n\n"
        f"{passages}"
    ) + (HITS_TRY_SIBLING if _sibling_on(ctx.config) else "")


# Over-cap warning fires once, not per turn (reset if the count drops back).
_index_cap_warned = False


class KnowledgeModule:
    """Document knowledge base: upload, chunk, embed, search."""

    name = "knowledge"
    config_key = "document_search"  # where the knobs live in agent_config.tool_config

    def register_tools(self):
        return [search_documents]

    async def system_prompt_fragment(self, context, query: str) -> str | None:
        """The uploaded-documents index — filenames the agent can search.

        Same contract as faq's question index (ADR-012): opt-in, capped,
        silent when the tool is off, and framed as "search these", never
        "you know these". Filenames only, never content.
        """
        global _index_cap_warned
        config = _tool_config(context.config)
        if not config.inject_document_index:
            return None
        if not agent_config_service.tool_enabled(context.config, "search_documents"):
            return None  # advertising a disabled tool would misroute the model

        filenames = await service.list_ready_filenames(context.session)
        if not filenames:
            return None
        if len(filenames) > config.document_index_max:
            # A silently truncated list is worse than none — the model would
            # treat it as exhaustive and skip documents that aren't on it.
            if not _index_cap_warned:
                logger.warning(
                    "Document index skipped: %d documents exceed "
                    "document_index_max=%d — raise the cap or disable "
                    "inject_document_index",
                    len(filenames),
                    config.document_index_max,
                )
                _index_cap_warned = True
            return None
        _index_cap_warned = False

        lines = "\n".join(f"- {name}" for name in filenames)
        return (
            "The operator uploaded the documents listed below. When the user "
            "asks about anything they may cover, call `search_documents` to "
            "retrieve the relevant passages. These are files you can SEARCH, "
            "not things you know: never answer about them from memory.\n"
            f"{lines}"
        )

    def register_models(self):
        return [Document, DocumentChunk]

    def register_admin_routes(self, router):
        from app.modules.knowledge.admin import register

        register(router)

    def config_schema(self):
        return KnowledgeSearchConfig


module = KnowledgeModule()
