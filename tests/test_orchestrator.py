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
    received: list = Field(default_factory=list)       # message lists per model call

    def bind_tools(self, tools, **kwargs):
        self.offered_tools.append(
            [getattr(t, "name", None) or t.get("function", {}).get("name") for t in tools]
        )
        return self.bind(tools=list(tools), **kwargs)  # base class raises NotImplementedError

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def scripted(*responses) -> ScriptedModel:
    return ScriptedModel(messages=iter(responses))


def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
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
        Message(conversation_id=conversation.id, direction="in", type="text", text="Hola"),
    )
    session.add(
        Message(conversation_id=conversation.id, direction="out", type="text", text="¡Hola! ¿En qué te ayudo?"),
    )
    await session.flush()

    memory = Memory(contact_id=conversation.contact_id, content="Se llama Willy")

    async def fake_retrieve(session_, contact_id, query, limit=5):
        return [memory]

    from app.services import memory_service

    monkeypatch.setattr(memory_service, "retrieve_relevant", fake_retrieve)

    model = scripted(AIMessage("Claro Willy, te ayudo."))
    replies = await orchestrator.run_turn(
        session, conversation, InboundMessage(type="text", text="¿Me ayudas?"), model=model
    )

    assert replies[0].text == "Claro Willy, te ayudo."

    prompt_messages = model.received[0]
    system = prompt_messages[0]
    assert isinstance(system, SystemMessage)
    assert "Chasqui-Bot" in system.content          # DB prompt respected
    assert "Se llama Willy" in system.content       # memory injected
    texts = [getattr(m, "content", "") for m in prompt_messages]
    assert "Hola" in texts                          # history present, oldest-first
    assert texts[-1] == "¿Me ayudas?"               # current message last


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
        session, conversation, InboundMessage(type="text", text="¿Horario?"), model=model
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
            session, conversation, InboundMessage(type="text", text="rompe"), model=model
        )
    finally:
        registry._MODULES[:] = [m for m in registry._MODULES if m.name != "boom-test"]

    assert "sigo aquí" in replies[0].text
    error_msgs = [
        m for m in model.received[1] if isinstance(m, ToolMessage) and m.status == "error"
    ]
    assert error_msgs and "kaput" in error_msgs[0].content


async def test_handoff_tool_flags_the_conversation(session):
    conversation = await make_conversation(session)
    model = scripted(
        tool_call("human_handoff", {"reason": "pide asesor"}),
        AIMessage("Te contacto con una persona del equipo."),
    )

    await orchestrator.run_turn(
        session, conversation, InboundMessage(type="text", text="quiero hablar con alguien"), model=model
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
    image_block = next(b for b in blocks if b["type"] == "image")
    assert image_block["base64"] == PNG_B64
    assert image_block["mime_type"] == "image/png"
    assert "mira esto" in blocks[0]["text"]  # caption travels with the image


async def test_audio_falls_back_to_text_when_model_lacks_audio(session, monkeypatch, caplog):
    from app.core.config import settings

    monkeypatch.setattr(settings, "llm_supports_audio", False)
    conversation = await make_conversation(session)
    model = scripted(AIMessage("¿Podrías escribirlo?"))

    inbound = InboundMessage(
        type="audio", media_url=f"data:audio/ogg;base64,{PNG_B64}"
    )
    with caplog.at_level("WARNING"):
        replies = await orchestrator.run_turn(session, conversation, inbound, model=model)

    current = model.received[0][-1]
    assert isinstance(current.content, str)  # no audio block sent
    assert "voice message" in current.content
    assert replies[0].text == "¿Podrías escribirlo?"
    assert any("lacks audio" in r.message for r in caplog.records)


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
        session, conversation, InboundMessage(type="text", text="mejor en las tardes"), model=model
    )

    from sqlmodel import select

    memories = (await session.exec(select(Memory))).all()
    assert len(memories) == 1
    assert memories[0].content == "Prefiere atención por las tardes"
    assert memories[0].contact_id == conversation.contact_id
