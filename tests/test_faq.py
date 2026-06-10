"""Sprint 4 acceptance: FAQ module — RAG service, tool and admin CRUD.

Embeddings are faked with deterministic vectors so pgvector computes REAL
cosine distances over them (the test DB has the extension): ranking and
threshold behavior are exercised end-to-end without network.
"""

import uuid
from types import SimpleNamespace

import pytest
from sqlmodel import select

from app.core.config import settings
from app.models import AgentConfig, Contact, Conversation
from app.modules.faq import NO_RESULTS, faq_search
from app.modules.faq import service as faq_service
from app.modules.faq.models import FaqEntry
from app.services.admin_service import create_admin_access_token
from app.services.agent_context import TurnContext


def vec(*head: float) -> list[float]:
    """A settings.embedding_dim-wide vector with `head` as leading components."""
    v = [0.0] * settings.embedding_dim
    v[: len(head)] = head
    return v


class FakeEmbeddings:
    """Deterministic embedder: known texts map to fixed vectors."""

    def __init__(self, mapping: dict[str, list[float]], default: list[float] | None = None):
        self.mapping = mapping
        self.default = default if default is not None else vec(1.0)
        self.query_calls: list[str] = []
        self.document_calls: list[list[str]] = []

    async def aembed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self.mapping.get(text, self.default)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [self.mapping.get(t, self.default) for t in texts]


HOURS_TEXT = "What are your opening hours?\nMon-Fri 9:00-18:00."
RETURNS_TEXT = "What is the return policy?\n30 days with receipt."

MAPPING = {
    HOURS_TEXT: vec(1.0, 0.0),
    RETURNS_TEXT: vec(0.0, 1.0),
    "opening hours": vec(0.95, 0.05),  # close to HOURS, far from RETURNS
    "unrelated topic": vec(0.0, 0.0, 1.0),  # orthogonal to everything
}


@pytest.fixture
def fake_embeddings(monkeypatch) -> FakeEmbeddings:
    import app.core.embeddings as embeddings_mod

    fake = FakeEmbeddings(MAPPING)
    monkeypatch.setattr(embeddings_mod, "get_embeddings", lambda: fake)
    return fake


async def make_entries(session) -> tuple[FaqEntry, FaqEntry]:
    hours = await faq_service.create_entry(
        session,
        question="What are your opening hours?",
        answer="Mon-Fri 9:00-18:00.",
        tags=["hours"],
    )
    returns = await faq_service.create_entry(
        session, question="What is the return policy?", answer="30 days with receipt."
    )
    return hours, returns


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


async def test_create_embeds_and_search_ranks_by_similarity(session, fake_embeddings):
    hours, returns = await make_entries(session)
    assert hours.embedding is not None  # embedded on save

    hits = await faq_service.search(session, "opening hours", top_k=4, min_similarity=0.5)

    assert [e.id for e, _ in hits] == [hours.id]  # returns filtered by threshold
    assert hits[0][1] > 0.9  # similarity, best first


async def test_search_below_threshold_returns_nothing(session, fake_embeddings):
    await make_entries(session)
    hits = await faq_service.search(session, "unrelated topic", min_similarity=0.5)
    assert hits == []


async def test_search_survives_embeddings_outage(session, fake_embeddings, monkeypatch):
    import app.core.embeddings as embeddings_mod

    def broken():
        raise RuntimeError("provider down")

    monkeypatch.setattr(embeddings_mod, "get_embeddings", broken)
    assert await faq_service.search(session, "opening hours") == []


async def test_update_reembeds_only_on_content_change(session, fake_embeddings):
    hours, _ = await make_entries(session)
    embed_calls_before = len(fake_embeddings.query_calls)

    await faq_service.update_entry(session, hours, tags=["schedule"])  # tags only
    assert len(fake_embeddings.query_calls) == embed_calls_before  # no re-embed

    await faq_service.update_entry(session, hours, answer="Mon-Sat 9:00-13:00.")
    assert len(fake_embeddings.query_calls) == embed_calls_before + 1  # re-embedded


async def test_reembed_all_is_one_batched_call(session, fake_embeddings):
    await make_entries(session)

    count = await faq_service.reembed_all(session)

    assert count == 2
    assert len(fake_embeddings.document_calls) == 1  # aembed_documents, batched


# ---------------------------------------------------------------------------
# Tool (grounded / honest-miss behavior)
# ---------------------------------------------------------------------------


async def make_runtime(session, tool_config: dict | None = None) -> SimpleNamespace:
    contact = Contact(channel="whatsapp", external_id="bsuid-FAQ-TEST")
    session.add(contact)
    await session.flush()
    conversation = Conversation(contact_id=contact.id)
    session.add(conversation)
    await session.flush()
    ctx = TurnContext(
        session=session,
        contact_id=contact.id,
        conversation_id=conversation.id,
        config=AgentConfig(tool_config=tool_config or {}),
    )
    return SimpleNamespace(context=ctx)


async def test_faq_search_tool_returns_grounded_snippets(session, fake_embeddings):
    await make_entries(session)
    runtime = await make_runtime(session)

    result = await faq_search.coroutine(query="opening hours", runtime=runtime)

    assert "Mon-Fri 9:00-18:00." in result
    assert "ONLY" in result  # grounding instruction


async def test_faq_search_tool_is_honest_on_miss(session, fake_embeddings):
    await make_entries(session)
    runtime = await make_runtime(session)

    result = await faq_search.coroutine(query="unrelated topic", runtime=runtime)

    assert result == NO_RESULTS


async def test_faq_search_respects_admin_tool_config(session, fake_embeddings):
    await make_entries(session)
    # Admin loosened the threshold: now everything within reach, but top_k=1
    runtime = await make_runtime(
        session, tool_config={"faq_search": {"top_k": 1, "min_similarity": 0.0}}
    )

    result = await faq_search.coroutine(query="opening hours", runtime=runtime)

    assert "Mon-Fri 9:00-18:00." in result
    assert "[2]" not in result  # top_k honored


# ---------------------------------------------------------------------------
# Admin routes (module contract: register_admin_routes)
# ---------------------------------------------------------------------------


@pytest.fixture
def admin_headers() -> dict:
    token = create_admin_access_token(uuid.uuid4(), "admin@test.local", "super_admin")
    return {"Authorization": f"Bearer {token}"}


async def test_module_admin_routes_require_auth(client):
    response = await client.get("/admin/modules/faq/entries")
    assert response.status_code == 401


async def test_admin_crud_flow(client, session, fake_embeddings, admin_headers):
    created = await client.post(
        "/admin/modules/faq/entries",
        json={
            "question": "What are your opening hours?",
            "answer": "Mon-Fri 9:00-18:00.",
            "tags": ["hours"],
        },
        headers=admin_headers,
    )
    assert created.status_code == 201
    body = created.json()
    assert body["has_embedding"] is True
    entry_id = body["id"]

    listed = await client.get("/admin/modules/faq/entries", headers=admin_headers)
    assert [e["id"] for e in listed.json()] == [entry_id]

    updated = await client.put(
        f"/admin/modules/faq/entries/{entry_id}",
        json={"answer": "Mon-Sat 9:00-13:00."},
        headers=admin_headers,
    )
    assert updated.status_code == 200
    assert updated.json()["answer"] == "Mon-Sat 9:00-13:00."

    deleted = await client.delete(
        f"/admin/modules/faq/entries/{entry_id}", headers=admin_headers
    )
    assert deleted.status_code == 204

    missing = await client.get(
        f"/admin/modules/faq/entries/{entry_id}", headers=admin_headers
    )
    assert missing.status_code == 404


async def test_admin_reembed_endpoint(client, session, fake_embeddings, admin_headers):
    await make_entries(session)

    response = await client.post("/admin/modules/faq/reembed", headers=admin_headers)

    assert response.status_code == 200
    assert response.json() == {"reembedded": 2}
    assert len(fake_embeddings.document_calls) == 1


async def test_admin_search_preview(client, session, fake_embeddings, admin_headers):
    """Operators preview retrieval with scores — no floor by default, so
    below-threshold entries show up (that's how you tune min_similarity)."""
    await make_entries(session)

    response = await client.get(
        "/admin/modules/faq/search",
        params={"q": "opening hours"},
        headers=admin_headers,
    )

    assert response.status_code == 200
    hits = response.json()
    assert len(hits) == 2  # both entries, even the barely-similar one
    assert hits[0]["entry"]["question"] == "What are your opening hours?"
    assert hits[0]["similarity"] > hits[1]["similarity"]
    assert "embedding" not in hits[0]["entry"]

    floored = await client.get(
        "/admin/modules/faq/search",
        params={"q": "opening hours", "min_similarity": 0.5},
        headers=admin_headers,
    )
    assert len(floored.json()) == 1


async def test_admin_search_preview_degrades_on_outage(
    client, monkeypatch, admin_headers
):
    import app.core.embeddings as embeddings_mod

    def boom():
        raise RuntimeError("provider down")

    monkeypatch.setattr(embeddings_mod, "get_embeddings", boom)

    response = await client.get(
        "/admin/modules/faq/search", params={"q": "anything"}, headers=admin_headers
    )

    assert response.status_code == 200
    assert response.json() == []


async def test_module_contract_is_fully_exercised(session):
    """The faq module is the proof-of-fire of the whole module contract."""
    from app.modules.faq import module

    assert module.register_tools()
    assert module.register_models() == [FaqEntry]
    assert module.config_schema().__name__ == "FaqSearchConfig"
    # entries created above never touched app/models — table comes from the module
    result = await session.exec(select(FaqEntry))
    assert result is not None
