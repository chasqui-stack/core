"""Sprint 4 carry-over: memory is no longer append-only.

- save_memory dedups near-identical facts (update, not duplicate)
- update_memory corrects a contradicted fact (LLM-driven, hint-matched)
- forget_memory deletes on request

Same fake-embeddings approach as test_faq.py: pgvector computes real cosine
distances over deterministic vectors.
"""

from types import SimpleNamespace

import pytest
from sqlmodel import select

from app.core.config import settings
from app.models import AgentConfig, Contact, Conversation, Memory
from app.modules.memory import forget_memory, save_memory, update_memory
from app.services.agent_context import TurnContext


def vec(*head: float) -> list[float]:
    v = [0.0] * settings.embedding_dim
    v[: len(head)] = head
    return v


MAPPING = {
    "Is single": vec(1.0, 0.0),
    "The user is single": vec(0.999, 0.01),  # near-duplicate of "Is single"
    "Is married": vec(0.0, 1.0),
    "Lives in Lima": vec(0.0, 0.0, 1.0),  # unrelated fact
    "Their dog is called Firulais": vec(0.0, 0.0, 0.0, 1.0),
}


class FakeEmbeddings:
    def __init__(self):
        self.query_calls = []

    async def aembed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return MAPPING[text]  # unknown text = test bug, fail loudly


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch) -> FakeEmbeddings:
    import app.core.embeddings as embeddings_mod

    fake = FakeEmbeddings()
    monkeypatch.setattr(embeddings_mod, "get_embeddings", lambda: fake)
    return fake


@pytest.fixture
async def runtime(session) -> SimpleNamespace:
    contact = Contact(channel="whatsapp", external_id="bsuid-MEM-TEST")
    session.add(contact)
    await session.flush()
    conversation = Conversation(contact_id=contact.id)
    session.add(conversation)
    await session.flush()
    ctx = TurnContext(
        session=session,
        contact_id=contact.id,
        conversation_id=conversation.id,
        config=AgentConfig(),
    )
    return SimpleNamespace(context=ctx)


async def all_memories(session) -> list[Memory]:
    return list((await session.exec(select(Memory))).all())


async def test_save_memory_dedups_near_identical_facts(session, runtime):
    await save_memory.coroutine(content="Is single", runtime=runtime)
    await save_memory.coroutine(content="The user is single", runtime=runtime)

    memories = await all_memories(session)
    assert len(memories) == 1  # updated in place, not duplicated
    assert memories[0].content == "The user is single"


async def test_save_memory_keeps_distinct_facts_apart(session, runtime):
    await save_memory.coroutine(content="Is single", runtime=runtime)
    await save_memory.coroutine(content="Lives in Lima", runtime=runtime)

    assert len(await all_memories(session)) == 2


async def test_update_memory_corrects_a_contradiction(session, runtime):
    await save_memory.coroutine(content="Is single", runtime=runtime)

    await update_memory.coroutine(
        old_content_hint="Is single", new_content="Is married", runtime=runtime
    )

    memories = await all_memories(session)
    assert len(memories) == 1
    assert memories[0].content == "Is married"  # the Sprint 3 e2e contradiction, fixed


async def test_update_memory_saves_as_new_when_nothing_matches(session, runtime):
    await save_memory.coroutine(content="Lives in Lima", runtime=runtime)

    await update_memory.coroutine(
        old_content_hint="Is single", new_content="Is married", runtime=runtime
    )

    contents = {m.content for m in await all_memories(session)}
    assert contents == {"Lives in Lima", "Is married"}


async def test_forget_memory_deletes_the_matched_fact(session, runtime):
    await save_memory.coroutine(content="Their dog is called Firulais", runtime=runtime)
    await save_memory.coroutine(content="Lives in Lima", runtime=runtime)

    result = await forget_memory.coroutine(
        content_hint="Their dog is called Firulais", runtime=runtime
    )

    assert "forgotten" in result
    contents = {m.content for m in await all_memories(session)}
    assert contents == {"Lives in Lima"}


async def test_forget_memory_is_honest_when_nothing_matches(session, runtime):
    await save_memory.coroutine(content="Lives in Lima", runtime=runtime)

    result = await forget_memory.coroutine(content_hint="Is single", runtime=runtime)

    assert "don't remember" in result
    assert len(await all_memories(session)) == 1  # nothing deleted
