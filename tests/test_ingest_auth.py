"""Gateway↔core shared secret on /ingest (X-Internal-API-Key)."""

import pytest

from app.core.config import settings
from app.schemas.ingest import OutboundMessage
from app.services import orchestrator
from tests.test_ingest import canonical_payload


@pytest.fixture(autouse=True)
def _stub_agent_turn(monkeypatch):
    """These tests gate on the shared secret, not the LLM — keep them hermetic
    so they need no real model (and no GOOGLE_API_KEY) in CI. Without this the
    valid-key cases run the real orchestrator and build ChatGoogleGenerativeAI."""

    async def fake_run_turn(session, conversation, inbound, **kwargs):
        return [OutboundMessage(type="text", text="ok")]

    monkeypatch.setattr(orchestrator, "run_turn", fake_run_turn)


@pytest.fixture
def internal_key(monkeypatch):
    monkeypatch.setattr(settings, "internal_api_key", "test-secret")
    return "test-secret"


async def test_ingest_rejects_missing_key(client, internal_key):
    resp = await client.post("/ingest", json=canonical_payload())
    assert resp.status_code == 401


async def test_ingest_rejects_wrong_key(client, internal_key):
    resp = await client.post(
        "/ingest", json=canonical_payload(), headers={"X-Internal-API-Key": "nope"}
    )
    assert resp.status_code == 401


async def test_ingest_accepts_valid_key(client, internal_key):
    resp = await client.post(
        "/ingest", json=canonical_payload(), headers={"X-Internal-API-Key": internal_key}
    )
    assert resp.status_code == 200


async def test_ingest_open_when_key_unset(client, monkeypatch):
    monkeypatch.setattr(settings, "internal_api_key", None)
    resp = await client.post("/ingest", json=canonical_payload())
    assert resp.status_code == 200
