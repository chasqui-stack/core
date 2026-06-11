"""Sprint 7: lead_capture v2 — leads table + config-driven collection."""

import uuid
from types import SimpleNamespace

import pytest
from sqlmodel import select

from app.models import AgentConfig, Contact, Conversation
from app.modules.handoff import lead_capture
from app.modules.handoff.models import Lead
from app.services.admin_service import create_admin_access_token
from app.services.agent_context import TurnContext


@pytest.fixture
def admin_headers() -> dict:
    token = create_admin_access_token(uuid.uuid4(), "admin@test.local", "super_admin")
    return {"Authorization": f"Bearer {token}"}


async def make_runtime(
    session, *, wa_id: str | None = None, tool_config: dict | None = None
) -> SimpleNamespace:
    contact = Contact(
        channel="whatsapp", external_id=f"bsuid-{uuid.uuid4().hex[:8]}", wa_id=wa_id
    )
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
    return SimpleNamespace(context=ctx, contact=contact)


async def all_leads(session) -> list[Lead]:
    return list((await session.exec(select(Lead))).all())


async def test_defaults_require_email_and_phone(session):
    runtime = await make_runtime(session, wa_id=None)

    result = await lead_capture.coroutine(
        name="Juan", interest="solar panels", runtime=runtime
    )

    assert "NOT saved" in result
    assert "email" in result and "phone" in result
    assert await all_leads(session) == []


async def test_known_wa_id_satisfies_phone(session):
    runtime = await make_runtime(session, wa_id="51999000111")

    result = await lead_capture.coroutine(
        name="Juan", interest="solar panels", email="juan@x.com", runtime=runtime
    )

    assert result.startswith("Lead saved")
    lead = (await session.exec(select(Lead))).one()
    assert lead.contact_id == runtime.contact.id
    assert lead.phone == "51999000111"  # the contact's number, asked for nothing
    assert lead.email == "juan@x.com"


async def test_extra_fields_are_collected_then_stored(session):
    config = {
        "lead_capture": {
            "require_email": False,
            "require_phone": False,
            "extra_fields": "company, city",
        }
    }
    runtime = await make_runtime(session, tool_config=config)

    first = await lead_capture.coroutine(name="Ada", interest="ERP", runtime=runtime)
    assert "NOT saved" in first
    assert "company" in first and "city" in first

    second = await lead_capture.coroutine(
        name="Ada",
        interest="ERP",
        extra={"company": "ACME", "city": "Lima"},
        runtime=runtime,
    )
    assert second.startswith("Lead saved")
    lead = (await session.exec(select(Lead))).one()
    assert lead.extra == {"company": "ACME", "city": "Lima"}
    assert lead.email is None and lead.phone is None


async def test_invalid_admin_config_falls_back_to_defaults(session):
    runtime = await make_runtime(
        session, wa_id="51999000111", tool_config={"lead_capture": {"require_email": "??"}}
    )

    # Defaults (require email) apply instead of crashing the turn
    result = await lead_capture.coroutine(name="X", interest="Y", runtime=runtime)
    assert "email" in result


async def test_leads_listing_endpoint(client, session, admin_headers):
    runtime = await make_runtime(session, wa_id="51999000111")
    other = await make_runtime(session, wa_id="51888000222")
    session.add(
        Lead(
            contact_id=runtime.contact.id,
            name="Juan",
            interest="solar",
            phone="51999000111",
            extra={"city": "Lima"},
        )
    )
    session.add(Lead(contact_id=other.contact.id, name="Eva", interest="wind"))
    await session.commit()

    assert (await client.get("/admin/modules/handoff/leads")).status_code == 401

    resp = await client.get("/admin/modules/handoff/leads", headers=admin_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    names = {item["name"] for item in body["items"]}
    assert names == {"Juan", "Eva"}
    juan = next(i for i in body["items"] if i["name"] == "Juan")
    assert juan["extra"] == {"city": "Lima"}
    assert juan["contact_id"] == str(runtime.contact.id)

    filtered = await client.get(
        "/admin/modules/handoff/leads",
        params={"contact_id": str(other.contact.id)},
        headers=admin_headers,
    )
    assert filtered.json()["total"] == 1
    assert filtered.json()["items"][0]["name"] == "Eva"
