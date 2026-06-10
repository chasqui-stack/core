"""Memory module — the agent saves, corrects and forgets durable facts.

Extraction-by-tool: instead of a second LLM pass after every turn, the
model calls these tools silently (proven pattern). Retrieval happens before
the turn in `memory_service.retrieve_relevant` (pgvector).

Sprint 4 (carry-over): memory is no longer append-only —
- `save_memory` dedups on save: a near-identical existing memory is updated
  in place instead of duplicated.
- `update_memory` / `forget_memory` let the model correct contradictions
  ("I'm not single anymore, I got married") that similarity alone can't
  catch: the hint is embedded and matched against the contact's nearest
  memory.
"""

import logging

from langchain.tools import ToolRuntime, tool

from app.models import Memory
from app.services import memory_service
from app.services.agent_context import TurnContext

logger = logging.getLogger(__name__)

# Near-duplicates ("His name is Willy" vs "The user is called Willy") —
# tight on purpose: false merges destroy facts, duplicates are just noise.
DEDUP_MAX_DISTANCE = 0.10

# Hint matching for update/forget — looser: the model paraphrases the old
# memory, but beyond this it's likely a different fact (then save/no-op).
HINT_MAX_DISTANCE = 0.45


async def _embed(text: str) -> list[float] | None:
    try:
        from app.core.embeddings import get_embeddings

        return await get_embeddings().aembed_query(text)
    except Exception:  # pragma: no cover - network/key issues must not kill the turn
        logger.warning("Embedding failed for memory text", exc_info=True)
        return None


@tool
async def save_memory(content: str, runtime: ToolRuntime[TurnContext]) -> str:
    """Save a durable fact about the user for future conversations.

    Use this tool SILENTLY whenever the user shares stable, useful
    information: their name, preferences, personal or work context,
    decisions made. Do NOT tell the user you saved anything.

    Do NOT save trivial or transient information (greetings, the literal
    message). If the fact CORRECTS something you already remember, use
    `update_memory` instead.

    Args:
        content: One concise, objective sentence.
            E.g. "Prefers to be contacted in the afternoon".
    """
    ctx = runtime.context
    embedding = await _embed(content)

    if embedding is not None:  # dedup-on-save: update the near-identical row
        existing = await memory_service.find_nearest(
            ctx.session, ctx.contact_id, embedding, max_distance=DEDUP_MAX_DISTANCE
        )
        if existing is not None:
            existing.content = content
            existing.embedding = embedding
            ctx.session.add(existing)
            return "Fact updated (a similar one existed). Do not mention it to the user."

    ctx.session.add(
        Memory(contact_id=ctx.contact_id, content=content, embedding=embedding)
    )
    return "Fact saved. Do not mention it to the user."


@tool
async def update_memory(
    old_content_hint: str, new_content: str, runtime: ToolRuntime[TurnContext]
) -> str:
    """Correct a fact you remember about the user that changed or was wrong.

    Use this tool SILENTLY when the user contradicts or updates something
    from "Facts you remember" (e.g. you remember "Is single" and they say
    they got married). Do NOT tell the user you updated anything.

    Args:
        old_content_hint: The outdated fact, as you remember it.
            E.g. "Is single".
        new_content: The corrected fact, concise and objective.
            E.g. "Is married".
    """
    ctx = runtime.context

    hint_embedding = await _embed(old_content_hint)
    new_embedding = await _embed(new_content)

    target = None
    if hint_embedding is not None:
        target = await memory_service.find_nearest(
            ctx.session, ctx.contact_id, hint_embedding, max_distance=HINT_MAX_DISTANCE
        )

    if target is None:  # nothing matched — store the corrected fact as new
        ctx.session.add(
            Memory(contact_id=ctx.contact_id, content=new_content, embedding=new_embedding)
        )
        return "Old fact not found; saved the new one. Do not mention it to the user."

    target.content = new_content
    target.embedding = new_embedding
    ctx.session.add(target)
    return "Fact corrected. Do not mention it to the user."


@tool
async def forget_memory(content_hint: str, runtime: ToolRuntime[TurnContext]) -> str:
    """Delete a fact you remember about the user, at their request.

    Use this tool when the user explicitly asks you to forget or delete
    something about them.

    Args:
        content_hint: The fact to forget, as you remember it.
            E.g. "Their ID number is 12345678".
    """
    ctx = runtime.context

    embedding = await _embed(content_hint)
    if embedding is None:
        return "Could not process the request right now; apologize briefly."

    target = await memory_service.find_nearest(
        ctx.session, ctx.contact_id, embedding, max_distance=HINT_MAX_DISTANCE
    )
    if target is None:
        return "No such fact is stored; tell the user you don't remember it."

    await ctx.session.delete(target)
    return "Fact forgotten. Briefly confirm it to the user."


class MemoryModule:
    """Long-term memory writing/correcting (retrieval is wired in the orchestrator)."""

    name = "memory"

    def register_tools(self):
        return [save_memory, update_memory, forget_memory]


module = MemoryModule()
