"""Sprint 7: handoff notifications — webhook + SMTP, both best-effort."""

import asyncio

import pytest

from app.core.config import settings
from app.models import Contact
from app.services import notify_service


@pytest.fixture
def contact() -> Contact:
    return Contact(
        channel="whatsapp",
        external_id="bsuid-NOTIFY",
        wa_id="51999000111",
        display_name="Juan",
    )


def test_event_shape(contact):
    event = notify_service.build_handoff_event(contact, "asks for a person")

    assert event["event"] == "handoff"
    assert event["reason"] == "asks for a person"
    assert event["at"]
    assert event["contact"]["display_name"] == "Juan"
    assert event["contact"]["channel"] == "whatsapp"


class FakeSMTP:
    """Records the SMTP conversation; shared by SMTP and SMTP_SSL fakes."""

    instances: list["FakeSMTP"] = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port = host, port
        self.calls: list = []
        FakeSMTP.instances.append(self)

    def starttls(self):
        self.calls.append("starttls")

    def login(self, user, password):
        self.calls.append(("login", user, password))

    def send_message(self, message):
        self.calls.append(("send", message))

    def quit(self):
        self.calls.append("quit")


@pytest.fixture(autouse=True)
def _reset_fake_smtp():
    FakeSMTP.instances = []


async def test_email_starttls_587(monkeypatch, contact):
    monkeypatch.setattr(settings, "smtp_host", "smtp-relay.brevo.com")
    monkeypatch.setattr(settings, "smtp_port", 587)
    monkeypatch.setattr(settings, "smtp_user", "user")
    monkeypatch.setattr(settings, "smtp_password", "pass")
    monkeypatch.setattr(settings, "smtp_from", "chasqui@x.com")
    monkeypatch.setattr(settings, "notify_email_to", "a@x.com, b@y.com")
    monkeypatch.setattr(notify_service.smtplib, "SMTP", FakeSMTP)

    await notify_service.send_email(
        notify_service.build_handoff_event(contact, "sales")
    )

    server = FakeSMTP.instances[0]
    assert server.host == "smtp-relay.brevo.com"
    assert server.calls[0] == "starttls"
    assert server.calls[1] == ("login", "user", "pass")
    kind, message = server.calls[2]
    assert kind == "send"
    assert message["To"] == "a@x.com, b@y.com"
    assert "Juan" in message["Subject"]
    assert "sales" in message.get_content()
    assert server.calls[-1] == "quit"


async def test_email_implicit_ssl_465_skips_starttls(monkeypatch, contact):
    monkeypatch.setattr(settings, "smtp_host", "smtp.example.com")
    monkeypatch.setattr(settings, "smtp_port", 465)
    monkeypatch.setattr(settings, "smtp_from", "chasqui@x.com")
    monkeypatch.setattr(settings, "notify_email_to", "ops@x.com")
    monkeypatch.setattr(notify_service.smtplib, "SMTP_SSL", FakeSMTP)

    await notify_service.send_email(notify_service.build_handoff_event(contact, "r"))

    server = FakeSMTP.instances[0]
    assert "starttls" not in server.calls  # implicit SSL
    assert server.calls[0][0] == "send"  # no creds configured → no login


async def test_notify_failures_are_independent_and_swallowed(monkeypatch, contact):
    monkeypatch.setattr(settings, "notify_webhook_url", "https://hooks.example/x")
    monkeypatch.setattr(settings, "smtp_host", "smtp.example.com")
    monkeypatch.setattr(settings, "smtp_from", "chasqui@x.com")
    monkeypatch.setattr(settings, "notify_email_to", "ops@x.com")

    emails: list[dict] = []

    async def broken_webhook(event):
        raise RuntimeError("hook down")

    async def record_email(event):
        emails.append(event)

    monkeypatch.setattr(notify_service, "send_webhook", broken_webhook)
    monkeypatch.setattr(notify_service, "send_email", record_email)

    # Must not raise — and the email still goes out despite the webhook failing
    await notify_service._notify(notify_service.build_handoff_event(contact, "r"))
    assert len(emails) == 1


async def test_dispatch_is_a_noop_when_nothing_configured(monkeypatch, contact):
    # conftest clears notify settings; building the event would be the first step
    def must_not_build(*args):
        raise AssertionError("dispatch must return early when unconfigured")

    monkeypatch.setattr(notify_service, "build_handoff_event", must_not_build)
    notify_service.dispatch_handoff(contact, "r")  # no exception = early return


async def test_dispatch_fires_background_task(monkeypatch, contact):
    monkeypatch.setattr(settings, "notify_webhook_url", "https://hooks.example/x")
    notified: list[dict] = []

    async def record(event):
        notified.append(event)

    monkeypatch.setattr(notify_service, "_notify", record)

    notify_service.dispatch_handoff(contact, "sales")
    await asyncio.sleep(0)  # let the task run

    assert len(notified) == 1
    assert notified[0]["reason"] == "sales"
