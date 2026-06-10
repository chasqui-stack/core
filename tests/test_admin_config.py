"""Sprint 5: /admin/config and /admin/tools — the panel's configuration API.

Writes are validated against the tool registry (unknown tool names and
schema-violating knob values must never reach the DB).
"""

import uuid

import pytest
from sqlmodel import select

from app.models import AgentConfig
from app.services.admin_service import create_admin_access_token


@pytest.fixture
def admin_headers() -> dict:
    token = create_admin_access_token(uuid.uuid4(), "admin@test.local", "super_admin")
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# /admin/config
# ---------------------------------------------------------------------------


async def test_config_requires_auth(client):
    assert (await client.get("/admin/config")).status_code == 401
    assert (await client.put("/admin/config", json={})).status_code == 401


async def test_get_config_seeds_if_missing(client, session, admin_headers):
    """Fresh DBs (create_all, no migration seed) must self-heal, not 500."""
    response = await client.get("/admin/config", headers=admin_headers)

    assert response.status_code == 200
    body = response.json()
    assert "system_prompt" in body and body["system_prompt"]
    assert body["enabled_tools"] == {}
    assert body["tool_config"] == {}

    # The seed actually landed
    config = (await session.exec(select(AgentConfig))).first()
    assert config is not None


async def test_put_config_partial_update_keeps_other_fields(
    client, session, admin_headers
):
    await client.put(
        "/admin/config",
        json={"enabled_tools": {"faq_search": False}},
        headers=admin_headers,
    )

    response = await client.put(
        "/admin/config",
        json={"system_prompt": "You are a pirate."},
        headers=admin_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["system_prompt"] == "You are a pirate."
    assert body["enabled_tools"] == {"faq_search": False}  # untouched


async def test_put_config_rejects_unknown_tool(client, admin_headers):
    response = await client.put(
        "/admin/config",
        json={"enabled_tools": {"not_a_tool": False}},
        headers=admin_headers,
    )

    assert response.status_code == 422
    assert "not_a_tool" in response.json()["detail"]


async def test_put_config_accepts_valid_tool_config(client, admin_headers):
    response = await client.put(
        "/admin/config",
        json={"tool_config": {"faq_search": {"top_k": 6}}},
        headers=admin_headers,
    )

    assert response.status_code == 200
    assert response.json()["tool_config"] == {"faq_search": {"top_k": 6}}


async def test_put_config_rejects_unknown_config_key(client, admin_headers):
    response = await client.put(
        "/admin/config",
        json={"tool_config": {"nope": {"x": 1}}},
        headers=admin_headers,
    )

    assert response.status_code == 422
    assert "nope" in response.json()["detail"]


async def test_put_config_rejects_schema_violation(client, admin_headers):
    # FaqSearchConfig: top_k must be >= 1
    response = await client.put(
        "/admin/config",
        json={"tool_config": {"faq_search": {"top_k": 0}}},
        headers=admin_headers,
    )

    assert response.status_code == 422
    assert "faq_search" in response.json()["detail"]


# ---------------------------------------------------------------------------
# /admin/tools
# ---------------------------------------------------------------------------


async def test_tools_requires_auth(client):
    assert (await client.get("/admin/tools")).status_code == 401


async def test_tools_lists_registry_with_schema_and_config(client, admin_headers):
    response = await client.get("/admin/tools", headers=admin_headers)

    assert response.status_code == 200
    modules = {m["name"]: m for m in response.json()["modules"]}

    faq = modules["faq"]
    assert faq["config_key"] == "faq_search"
    assert [t["name"] for t in faq["tools"]] == ["faq_search"]
    assert all(t["enabled"] for t in faq["tools"])  # missing key = enabled
    # JSON Schema feeds the admin auto-form
    props = faq["config_schema"]["properties"]
    assert set(props) == {"top_k", "min_similarity"}
    assert props["top_k"]["minimum"] == 1
    # Effective config = schema defaults when nothing is stored
    assert faq["config"] == {"top_k": 4, "min_similarity": 0.5}

    # Modules without knobs expose null schema/config
    assert modules["memory"]["config_schema"] is None
    assert modules["memory"]["config"] is None


async def test_tools_reflects_stored_state(client, admin_headers):
    await client.put(
        "/admin/config",
        json={
            "enabled_tools": {"faq_search": False},
            "tool_config": {"faq_search": {"top_k": 7}},
        },
        headers=admin_headers,
    )

    response = await client.get("/admin/tools", headers=admin_headers)
    faq = {m["name"]: m for m in response.json()["modules"]}["faq"]

    assert faq["tools"][0]["enabled"] is False
    # Stored values merged over schema defaults
    assert faq["config"] == {"top_k": 7, "min_similarity": 0.5}
