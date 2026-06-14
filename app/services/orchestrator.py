"""Agent orchestrator — the real LangGraph turn (ARCHITECTURE §6, §8).

LangChain v1 `create_agent` gives us the router → ToolNode → respond loop;
Chasqui supplies the pieces around it:

- system prompt: DB-editable (agent_config) + retrieved long-term memories
- history: the conversation's persisted messages (text-only window)
- current message: multimodal content blocks (image/audio) when the
  configured model supports them (app/core/llm_capabilities.py), graceful
  text fallback when it doesn't
- tools: discovered from app/modules/ and filtered by ToolFilterMiddleware;
  ToolErrorMiddleware turns tool crashes into recoverable ToolMessages

`run_turn()` is the seam ingest_service calls — same contract as the
Sprint 1 stub, now with the session (history/memories/tools need the DB).
"""

import logging
from datetime import datetime, timezone

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core import storage
from app.core.config import settings
from app.core.llm_capabilities import ModelCapabilities, resolve_capabilities
from app.models import AgentConfig, Conversation, Memory, Message
from app.modules import registry
from app.schemas.ingest import InboundMessage, OutboundMessage
from app.services import agent_config_service, memory_service
from app.services.agent_context import TurnContext
from app.services.agent_middleware import ToolErrorMiddleware, ToolFilterMiddleware

logger = logging.getLogger(__name__)

# End-user-facing (sent verbatim on errors) → operator-configurable via .env
# (FALLBACK_REPLY). Everything LLM-facing below is English: the system prompt
# rule "reply in the user's language" handles localization.

_agent = None  # built once per process (default model + discovered tools)


def _build_agent(model: BaseChatModel):
    registry.discover()  # idempotent — ensures tools exist outside app startup
    return create_agent(
        model=model,
        tools=registry.get_tools(),
        middleware=[ToolFilterMiddleware(), ToolErrorMiddleware()],
        context_schema=TurnContext,
    )


def _get_agent():
    global _agent
    if _agent is None:
        from app.core.llm import get_chat_model

        _agent = _build_agent(get_chat_model())
    return _agent


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def _system_message(config: AgentConfig, memories: list[Memory]) -> SystemMessage:
    parts = [config.system_prompt]
    if memories:
        facts = "\n".join(f"- {m.content}" for m in memories)
        parts.append(
            "Facts you remember about the user (long-term memory):\n"
            f"{facts}\n"
            "If the user corrects or contradicts any of these facts, "
            "silently update it with `update_memory`."
        )
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts.append(f"Fecha y hora actual: {now}")
    return SystemMessage("\n\n".join(parts))


async def _history_messages(
    session: AsyncSession, conversation_id, limit: int, *, exclude_pending: bool = False
) -> list[HumanMessage | AIMessage]:
    """Last `limit` persisted messages, oldest-first, as chat messages.

    History is text-only (media stays in the current-turn message): old
    media would blow up the token budget for little gain.

    `exclude_pending` (coalesced turn, ADR-008): drop inbound that hasn't been
    processed yet — those ARE the current batch, fed as the turn's input, so
    they must not also appear as history.
    """
    query = select(Message).where(Message.conversation_id == conversation_id)
    if exclude_pending:
        # Keep outbound + already-processed inbound; drop pending inbound.
        query = query.where(
            ~((Message.direction == "in") & (Message.processed_at.is_(None)))
        )
    result = await session.exec(
        query.order_by(Message.created_at.desc()).limit(limit)
    )
    rows = list(result.all())[::-1]

    history: list[HumanMessage | AIMessage] = []
    for m in rows:
        text = m.text or f"[{m.type}]"
        history.append(HumanMessage(text) if m.direction == "in" else AIMessage(text))
    return history


def _parse_data_uri(uri: str | None) -> tuple[str, str] | None:
    """'data:<mime>;base64,<payload>' → (mime, payload). None if not a data URI."""
    if not uri or not uri.startswith("data:"):
        return None
    header, sep, payload = uri.partition(",")
    if not sep or not payload:
        return None
    mime = header[5:].split(";")[0] or "application/octet-stream"
    return mime, payload


def _current_message(inbound: InboundMessage, caps: ModelCapabilities) -> HumanMessage:
    """The inbound message as the model should see it (multimodal when possible)."""
    media = _parse_data_uri(inbound.media_url)

    if inbound.type == "image":
        if caps.vision and media:
            mime, b64 = media
            caption = (
                f'The user sent an image with the message: "{inbound.text}". '
                if inbound.text
                else "The user sent an image. "
            )
            return HumanMessage(
                content=[
                    {"type": "text", "text": caption + "Look at it and respond naturally."},
                    {"type": "image", "base64": b64, "mime_type": mime},
                ]
            )
        logger.warning(
            "Image received but model '%s:%s' lacks vision (or no media data) — text fallback",
            settings.llm_provider,
            settings.llm_model,
        )
        return HumanMessage(
            inbound.text
            or "[The user sent an image you cannot see. Ask them to describe it in text.]"
        )

    if inbound.type == "audio":
        if caps.audio and media:
            mime, b64 = media
            return HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "The user sent a voice message. Listen to it and respond "
                            "to its content naturally. Do NOT say you transcribed it."
                        ),
                    },
                    {"type": "audio", "base64": b64, "mime_type": mime},
                ]
            )
        logger.warning(
            "Audio received but model '%s:%s' lacks audio input (or no media data) — text fallback",
            settings.llm_provider,
            settings.llm_model,
        )
        return HumanMessage(
            "[The user sent a voice message you cannot listen to. "
            "Kindly ask them to write it as text.]"
        )

    # text / button / anything else the gateway normalized to text
    return HumanMessage(inbound.text or f"[{inbound.type}]")


def _extract_text(message) -> str:
    """Final answer text (Gemini may return content as block lists)."""
    text = getattr(message, "text", None)
    if text:  # property in langchain-core 1.x (callable-str compat wrapper)
        return str(text)
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
    return str(content)


# ---------------------------------------------------------------------------
# The turn
# ---------------------------------------------------------------------------


def _capabilities() -> ModelCapabilities:
    return resolve_capabilities(
        settings.llm_provider,
        settings.llm_model,
        vision_override=settings.llm_supports_vision,
        audio_override=settings.llm_supports_audio,
    )


async def _invoke(
    session: AsyncSession,
    conversation: Conversation,
    config: AgentConfig,
    messages: list,
    model: BaseChatModel | None,
) -> list[OutboundMessage]:
    """Run the assembled message list through the agent → canonical reply."""
    agent = _build_agent(model) if model is not None else _get_agent()
    context = TurnContext(
        session=session,
        contact_id=conversation.contact_id,
        conversation_id=conversation.id,
        config=config,
    )
    try:
        result = await agent.ainvoke({"messages": messages}, context=context)
        reply = _extract_text(result["messages"][-1]).strip()
    except Exception:
        logger.exception("Agent turn failed for conversation %s", conversation.id)
        reply = settings.fallback_reply

    return [OutboundMessage(type="text", text=reply or settings.fallback_reply)]


async def run_turn(
    session: AsyncSession,
    conversation: Conversation,
    inbound: InboundMessage,
    *,
    model: BaseChatModel | None = None,
) -> list[OutboundMessage]:
    """Produce the agent's reply (1..N messages) for one inbound message.

    `model` overrides the configured LLM (tests inject a scripted fake).
    """
    config = await agent_config_service.get_config(session)
    memories = await memory_service.retrieve_relevant(
        session, conversation.contact_id, inbound.text or ""
    )
    messages = [
        _system_message(config, memories),
        *await _history_messages(session, conversation.id, settings.history_limit),
        _current_message(inbound, _capabilities()),
    ]
    return await _invoke(session, conversation, config, messages, model)


async def _message_to_inbound(message: Message) -> InboundMessage:
    """Re-hydrate a persisted inbound row into a turn-ready InboundMessage.

    Media bytes live in the bucket (ADR-003); the row holds only the object
    key. Re-fetch and rebuild the `data:` URI so the coalesced turn is as
    multimodal as the synchronous one. No bucket (or fetch failure) → media
    drops to None, which `_current_message` renders as a text fallback.
    """
    media_url = None
    if storage.is_media_key(message.media_url) and storage.is_configured():
        try:
            media_url = await storage.get_media_data_uri(message.media_url)
        except Exception:
            logger.warning(
                "Could not re-hydrate media %s for coalesced turn — text fallback",
                message.media_url,
            )
    return InboundMessage(
        type=message.type, text=message.text, media_url=media_url, raw=message.meta or {}
    )


async def run_coalesced_turn(
    session: AsyncSession,
    conversation: Conversation,
    batch: list[Message],
    *,
    model: BaseChatModel | None = None,
) -> list[OutboundMessage]:
    """One turn over a burst of inbound messages (ADR-008).

    `batch` is the ordered list of pending inbound rows. They are fed as the
    turn's current input (one Human message each, in order) and excluded from
    history. Memory retrieval is keyed on their concatenated text.
    """
    config = await agent_config_service.get_config(session)
    query = "\n".join(m.text for m in batch if m.text)
    memories = await memory_service.retrieve_relevant(
        session, conversation.contact_id, query
    )
    caps = _capabilities()
    current = [
        _current_message(await _message_to_inbound(m), caps) for m in batch
    ]
    messages = [
        _system_message(config, memories),
        *await _history_messages(
            session, conversation.id, settings.history_limit, exclude_pending=True
        ),
        *current,
    ]
    return await _invoke(session, conversation, config, messages, model)
