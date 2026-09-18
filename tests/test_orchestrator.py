"""Sprint 3 acceptance: LangGraph orchestrator + Tool Registry (DB-backed).

A scripted fake chat model drives the graph deterministically (no network):
- tool calls round-trip through the agent
- disabled tools are never offered to the model
- a tool exception becomes an error ToolMessage (the turn survives)
- prompt assembly: DB system prompt + memories + history + multimodal blocks
"""

import base64

import pytest
from langchain.tools import tool
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from pydantic import Field

from app.models import Contact, Conversation, Memory, Message
from app.modules import registry
from app.schemas.ingest import InboundMessage
from app.services import agent_config_service, orchestrator

PNG_B64 = base64.b64encode(b"fake-png-bytes").decode()


class ScriptedModel(GenericFakeChatModel):
    """Fake chat model that records what it was offered and asked."""

    offered_tools: list = Field(default_factory=list)  # tool names per model call
    received: list = Field(default_factory=list)  # message lists per model call

    def bind_tools(self, tools, **kwargs):
        self.offered_tools.append(
            [
                getattr(t, "name", None) or t.get("function", {}).get("name")
                for t in tools
            ]
        )
        return self.bind(
            tools=list(tools), **kwargs
        )  # base class raises NotImplementedError

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def scripted(*responses) -> ScriptedModel:
    return ScriptedModel(messages=iter(responses))


def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


@pytest.fixture(autouse=True)
def no_network_memory(monkeypatch):
    """Default: no embeddings network calls — retrieval empty, embedder down.

    Tests that need vectors (e.g. save_memory) monkeypatch their own fake on
    top; everything else exercises the graceful-degradation paths.
    """

    async def empty_retrieve(session, contact_id, query, limit=5):
        return []

    from app.services import memory_service

    monkeypatch.setattr(memory_service, "retrieve_relevant", empty_retrieve)

    import app.core.embeddings as embeddings_mod

    def no_embeddings():
        raise RuntimeError("embeddings disabled in tests")

    monkeypatch.setattr(embeddings_mod, "get_embeddings", no_embeddings)


async def make_conversation(session) -> Conversation:
    contact = Contact(channel="whatsapp", external_id="bsuid-ORCH-TEST")
    session.add(contact)
    await session.flush()
    conversation = Conversation(contact_id=contact.id)
    session.add(conversation)
    await session.flush()
    return conversation


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


async def test_turn_uses_db_system_prompt_history_and_memories(session, monkeypatch):
    conversation = await make_conversation(session)

    # Editable prompt from DB
    config = await agent_config_service.get_config(session)
    config.system_prompt = "Eres Chasqui-Bot, el asistente de PRUEBA S.A."
    session.add(config)

    # Prior history + a stored memory (with vector retrieval stubbed out)
    session.add(
        Message(
            conversation_id=conversation.id, direction="in", type="text", text="Hola"
        ),
    )
    session.add(
        Message(
            conversation_id=conversation.id,
            direction="out",
            type="text",
            text="¡Hola! ¿En qué te ayudo?",
        ),
    )
    await session.flush()

    memory = Memory(contact_id=conversation.contact_id, content="Se llama Willy")

    async def fake_retrieve(session_, contact_id, query, limit=5):
        return [memory]

    from app.services import memory_service

    monkeypatch.setattr(memory_service, "retrieve_relevant", fake_retrieve)

    model = scripted(AIMessage("Claro Willy, te ayudo."))
    replies = await orchestrator.run_turn(
        session,
        conversation,
        InboundMessage(type="text", text="¿Me ayudas?"),
        model=model,
    )

    assert replies[0].text == "Claro Willy, te ayudo."

    prompt_messages = model.received[0]
    system = prompt_messages[0]
    assert isinstance(system, SystemMessage)
    assert "Chasqui-Bot" in system.content  # DB prompt respected
    assert "Se llama Willy" in system.content  # memory injected
    texts = [getattr(m, "content", "") for m in prompt_messages]
    assert "Hola" in texts  # history present, oldest-first
    assert texts[-1] == "¿Me ayudas?"  # current message last


# ---------------------------------------------------------------------------
# Tool Registry through the graph
# ---------------------------------------------------------------------------


async def test_tool_call_round_trips_through_the_graph(session):
    conversation = await make_conversation(session)
    model = scripted(
        tool_call("faq_search", {"query": "horario de atención"}),
        AIMessage("Atendemos de 9 a 6."),
    )

    replies = await orchestrator.run_turn(
        session,
        conversation,
        InboundMessage(type="text", text="¿Horario?"),
        model=model,
    )

    assert replies[0].text == "Atendemos de 9 a 6."
    # Second model call saw the ToolMessage with the stub's output
    tool_msgs = [m for m in model.received[1] if isinstance(m, ToolMessage)]
    assert tool_msgs and "knowledge base" in tool_msgs[0].content


async def test_disabled_tool_is_not_offered_to_the_model(session):
    conversation = await make_conversation(session)
    config = await agent_config_service.get_config(session)
    config.enabled_tools = {"faq_search": False}
    session.add(config)
    await session.flush()

    model = scripted(AIMessage("ok"))
    await orchestrator.run_turn(
        session, conversation, InboundMessage(type="text", text="hola"), model=model
    )

    offered = model.offered_tools[0]
    assert "faq_search" not in offered
    assert "human_handoff" in offered  # everything else stays available


async def test_tool_exception_becomes_tool_message_and_turn_survives(session):
    @tool
    def boom(x: str) -> str:
        """Siempre falla (solo para tests)."""
        raise RuntimeError("kaput")

    class BoomModule:
        name = "boom-test"

        def register_tools(self):
            return [boom]

    registry.register_module(BoomModule())
    try:
        conversation = await make_conversation(session)
        model = scripted(
            tool_call("boom", {"x": "1"}),
            AIMessage("Tuve un problema con esa consulta, pero sigo aquí."),
        )
        replies = await orchestrator.run_turn(
            session,
            conversation,
            InboundMessage(type="text", text="rompe"),
            model=model,
        )
    finally:
        registry._MODULES[:] = [m for m in registry._MODULES if m.name != "boom-test"]

    assert "sigo aquí" in replies[0].text
    error_msgs = [
        m
        for m in model.received[1]
        if isinstance(m, ToolMessage) and m.status == "error"
    ]
    assert error_msgs and "kaput" in error_msgs[0].content


async def test_handoff_tool_flags_the_conversation(session):
    conversation = await make_conversation(session)
    model = scripted(
        tool_call("human_handoff", {"reason": "pide asesor"}),
        AIMessage("Te contacto con una persona del equipo."),
    )

    await orchestrator.run_turn(
        session,
        conversation,
        InboundMessage(type="text", text="quiero hablar con alguien"),
        model=model,
    )

    # Persist (as ingest_service's final flush does) and re-read from the DB
    await session.flush()
    await session.refresh(conversation)
    handoff = conversation.conversation_state.get("handoff")
    assert handoff and handoff["requested"] is True
    assert handoff["reason"] == "pide asesor"


# ---------------------------------------------------------------------------
# Multimodal gating (llm_capabilities)
# ---------------------------------------------------------------------------


async def test_image_becomes_content_blocks_when_model_has_vision(session, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "llm_supports_vision", True)
    conversation = await make_conversation(session)
    model = scripted(AIMessage("Veo un gato."))

    inbound = InboundMessage(
        type="image", text="mira esto", media_url=f"data:image/png;base64,{PNG_B64}"
    )
    await orchestrator.run_turn(session, conversation, inbound, model=model)

    current = model.received[0][-1]
    blocks = current.content
    assert isinstance(blocks, list)
    # Media block FIRST (Google's documented order) — with tools bound,
    # gemini-2.5 refuses to "see" text-first images (0/6 vs 6/6 live A/B).
    assert blocks[0]["type"] == "image"
    assert blocks[0]["base64"] == PNG_B64
    assert blocks[0]["mime_type"] == "image/png"
    assert "mira esto" in blocks[1]["text"]  # caption travels with the image
    # The system prompt asserts the model's real senses (persona prompts make
    # gemini-2.5 deny seeing media otherwise).
    system = model.received[0][0]
    assert "CAN see attached images" in system.content


async def test_attachments_line_never_promises_missing_senses(session, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "llm_supports_vision", False)
    monkeypatch.setattr(settings, "llm_supports_audio", False)
    conversation = await make_conversation(session)
    model = scripted(AIMessage("ok"))

    await orchestrator.run_turn(
        session, conversation, InboundMessage(type="text", text="hola"), model=model
    )

    system = model.received[0][0]
    assert "Attachments:" not in system.content  # text-only model: no false claim


async def test_media_turn_uses_the_no_thinking_model(session, monkeypatch):
    """A media turn with no model override picks llm.get_media_chat_model()."""
    from app.core import llm
    from app.core.config import settings

    monkeypatch.setattr(settings, "llm_supports_vision", True)
    media_model = scripted(AIMessage("Veo un círculo rojo."))
    picked: list = []

    def fake_media_model(**kwargs):
        picked.append(kwargs)
        return media_model

    monkeypatch.setattr(llm, "get_media_chat_model", fake_media_model)

    conversation = await make_conversation(session)
    inbound = InboundMessage(type="image", media_url=f"data:image/png;base64,{PNG_B64}")
    replies = await orchestrator.run_turn(session, conversation, inbound)

    assert picked, "media turn must swap in the media chat model"
    assert replies[0].text == "Veo un círculo rojo."


def test_media_model_disables_gemini_thinking(monkeypatch):
    from app.core import llm
    from app.core.config import settings

    captured: dict = {}
    monkeypatch.setattr(llm, "get_chat_model", lambda **kw: captured.update(kw))

    monkeypatch.setattr(settings, "llm_provider", "google")
    llm.get_media_chat_model()
    assert captured["thinking_budget"] == 0

    captured.clear()
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    llm.get_media_chat_model()
    assert "thinking_budget" not in captured  # other providers untouched


async def test_text_turn_keeps_the_default_model(session, monkeypatch):
    """Text turns must NOT pay the media-model swap (thinking stays on)."""
    from app.core import llm

    def boom(**kwargs):  # would fail the test if called
        raise AssertionError("text turn must not use the media model")

    monkeypatch.setattr(llm, "get_media_chat_model", boom)

    conversation = await make_conversation(session)
    model = scripted(AIMessage("hola"))
    replies = await orchestrator.run_turn(
        session, conversation, InboundMessage(type="text", text="hola"), model=model
    )
    assert replies[0].text == "hola"


async def test_audio_falls_back_to_text_when_model_lacks_audio(
    session, monkeypatch, caplog
):
    from app.core.config import settings

    monkeypatch.setattr(settings, "llm_supports_audio", False)
    conversation = await make_conversation(session)
    model = scripted(AIMessage("¿Podrías escribirlo?"))

    inbound = InboundMessage(type="audio", media_url=f"data:audio/ogg;base64,{PNG_B64}")
    with caplog.at_level("WARNING"):
        replies = await orchestrator.run_turn(
            session, conversation, inbound, model=model
        )

    current = model.received[0][-1]
    assert isinstance(current.content, str)  # no audio block sent
    assert "voice message" in current.content
    assert replies[0].text == "¿Podrías escribirlo?"
    assert any("lacks audio" in r.message for r in caplog.records)


async def test_audio_transcribed_when_stt_enabled(session, monkeypatch):
    """STT on + audio-less LLM → the transcript reaches the agent as text (ADR-010)."""
    from app.core.config import settings
    from app.services import transcription

    monkeypatch.setattr(settings, "llm_supports_audio", False)
    monkeypatch.setattr(transcription, "stt_enabled", lambda: True)

    async def fake_transcribe(audio, mime):
        assert mime == "audio/ogg"  # OGG handed through as-is
        return "Hola, ¿cuál es el horario de atención?"

    monkeypatch.setattr(transcription, "transcribe", fake_transcribe)

    conversation = await make_conversation(session)
    model = scripted(AIMessage("Atendemos de 9 a 6."))
    inbound = InboundMessage(type="audio", media_url=f"data:audio/ogg;base64,{PNG_B64}")

    replies = await orchestrator.run_turn(session, conversation, inbound, model=model)

    current = model.received[0][-1]
    assert isinstance(current.content, str)  # rendered as text, no audio block
    assert "horario de atención" in current.content  # the transcript is the content
    assert replies[0].text == "Atendemos de 9 a 6."


async def test_native_audio_llm_never_transcribes(session, monkeypatch):
    """caps.audio True (Gemini) → STT is skipped, audio block sent natively."""
    from app.core.config import settings
    from app.services import transcription

    monkeypatch.setattr(settings, "llm_supports_audio", True)
    monkeypatch.setattr(transcription, "stt_enabled", lambda: True)

    calls: list = []

    async def spy_transcribe(audio, mime):
        calls.append(mime)
        return "should not be used"

    monkeypatch.setattr(transcription, "transcribe", spy_transcribe)

    conversation = await make_conversation(session)
    model = scripted(AIMessage("Escuché tu nota."))
    inbound = InboundMessage(type="audio", media_url=f"data:audio/ogg;base64,{PNG_B64}")

    await orchestrator.run_turn(session, conversation, inbound, model=model)

    assert calls == []  # native audio path: transcribe never called
    blocks = model.received[0][-1].content
    assert isinstance(blocks, list)
    assert blocks[0]["type"] == "audio"  # sent natively, media-first order


# ---------------------------------------------------------------------------
# Memory write path (save_memory tool, embeddings stubbed)
# ---------------------------------------------------------------------------


async def test_save_memory_tool_persists_a_memory(session, monkeypatch):
    import app.core.embeddings as embeddings_mod

    class FakeEmbeddings:
        async def aembed_query(self, text):
            return [0.0] * 768

    monkeypatch.setattr(embeddings_mod, "get_embeddings", lambda: FakeEmbeddings())

    conversation = await make_conversation(session)
    model = scripted(
        tool_call("save_memory", {"content": "Prefiere atención por las tardes"}),
        AIMessage("Entendido, ¿algo más?"),
    )

    await orchestrator.run_turn(
        session,
        conversation,
        InboundMessage(type="text", text="mejor en las tardes"),
        model=model,
    )

    from sqlmodel import select

    memories = (await session.exec(select(Memory))).all()
    assert len(memories) == 1
    assert memories[0].content == "Prefiere atención por las tardes"
    assert memories[0].contact_id == conversation.contact_id


# ---------------------------------------------------------------------------
# Module system-prompt fragments (ADR-012)
# ---------------------------------------------------------------------------


def _remove_module(name: str) -> None:
    registry._MODULES[:] = [m for m in registry._MODULES if m.name != name]


async def test_module_fragment_lands_in_the_system_prompt(session):
    class FragModule:
        name = "frag-test"

        def register_tools(self):
            return []

        async def system_prompt_fragment(self, context, query):
            return f"FRAGMENT seen query: {query}"

    registry.register_module(FragModule())
    try:
        conversation = await make_conversation(session)
        model = scripted(AIMessage("ok"))
        await orchestrator.run_turn(
            session,
            conversation,
            InboundMessage(type="text", text="hola frag"),
            model=model,
        )
    finally:
        _remove_module("frag-test")

    system = model.received[0][0]
    assert isinstance(system, SystemMessage)
    # The hook received the inbound text and its return was appended
    assert "FRAGMENT seen query: hola frag" in system.content
    # Modules WITHOUT the hook (faq default-off, handoff) didn't break anything
    assert model.offered_tools[0]  # turn ran with the full registry loaded


async def test_broken_fragment_is_isolated_and_turn_survives(session):
    class BrokenFragModule:
        name = "broken-frag"

        def register_tools(self):
            return []

        async def system_prompt_fragment(self, context, query):
            raise RuntimeError("fragment kaput")

    registry.register_module(BrokenFragModule())
    try:
        conversation = await make_conversation(session)
        model = scripted(AIMessage("sigo vivo"))
        replies = await orchestrator.run_turn(
            session,
            conversation,
            InboundMessage(type="text", text="hola"),
            model=model,
        )
    finally:
        _remove_module("broken-frag")

    assert replies[0].text == "sigo vivo"  # the turn never saw the crash


async def test_fragment_returning_none_is_omitted(session):
    class SilentFragModule:
        name = "silent-frag"

        def register_tools(self):
            return []

        async def system_prompt_fragment(self, context, query):
            return None

    registry.register_module(SilentFragModule())
    try:
        conversation = await make_conversation(session)
        model = scripted(AIMessage("ok"))
        await orchestrator.run_turn(
            session,
            conversation,
            InboundMessage(type="text", text="hola"),
            model=model,
        )
    finally:
        _remove_module("silent-frag")

    system = model.received[0][0]
    assert "silent-frag" not in system.content
    assert "None" not in system.content.split("Fecha")[0]  # no stray empty parts
