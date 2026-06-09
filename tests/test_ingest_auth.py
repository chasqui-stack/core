"""Gateway↔core shared secret on /ingest (X-Internal-API-Key)."""

import pytest

from app.core.config import settings
from tests.test_ingest import canonical_payload


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
