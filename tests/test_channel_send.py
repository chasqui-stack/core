"""Sprint 7: the canonical outbound seam (core → gateway /send, ADR-004)."""

import httpx
import pytest

from app.core.config import settings
from app.models import Contact
from app.services import channel_send


@pytest.fixture
def contact() -> Contact:
    return Contact(
        channel="whatsapp", external_id="bsuid-SEND", wa_id="51999000111"
    )


class FakeResponse:
    def __init__(self, status_code: int, body=None, json_raises: bool = False):
        self.status_code = status_code
        self._body = body
        self._json_raises = json_raises

    def json(self):
        if self._json_raises:
            raise ValueError("not json")
        return self._body


class FakeAsyncClient:
    """Stands in for httpx.AsyncClient — records the POST, returns a script."""

    def __init__(self, response: FakeResponse, recorded: list):
        self._response = response
        self._recorded = recorded

    def __call__(self, **kwargs):  # the AsyncClient(...) constructor call
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, json=None, headers=None):
        if isinstance(self._response, Exception):
            raise self._response
        self._recorded.append({"url": url, "json": json, "headers": headers})
        return self._response


def _stub_http(monkeypatch, response) -> list:
    recorded: list = []
    monkeypatch.setattr(
        channel_send.httpx, "AsyncClient", FakeAsyncClient(response, recorded)
    )
    return recorded


async def test_unconfigured_channel_raises_clear_code(contact):
    with pytest.raises(channel_send.ChannelSendError) as exc:
        await channel_send.send_message(contact, text="hola")
    assert exc.value.code == "CHANNEL_NOT_CONFIGURED"
    assert "CHANNEL_WHATSAPP_SEND_URL" in exc.value.message


async def test_send_message_posts_canonical_payload(monkeypatch, contact):
    monkeypatch.setattr(settings, "channel_whatsapp_send_url", "http://gw:8000/send")
    monkeypatch.setattr(settings, "internal_api_key", "sekret")
    recorded = _stub_http(
        monkeypatch, FakeResponse(200, {"status": "sent", "message_id": "wamid.X"})
    )

    result = await channel_send.send_message(contact, text="hola")

    assert result["status"] == "sent"
    call = recorded[0]
    assert call["url"] == "http://gw:8000/send"
    assert call["headers"] == {"X-Internal-API-Key": "sekret"}
    assert call["json"]["contact"]["wa_id"] == "51999000111"
    assert call["json"]["message"] == {
        "type": "text",
        "text": "hola",
        "media_url": None,
        "filename": None,
    }


async def test_send_message_carries_media(monkeypatch, contact):
    monkeypatch.setattr(settings, "channel_whatsapp_send_url", "http://gw:8000/send")
    recorded = _stub_http(monkeypatch, FakeResponse(200, {"status": "sent"}))

    await channel_send.send_message(
        contact,
        type="document",
        text="el manual",
        media_url="data:application/pdf;base64,JVBERg==",
        filename="manual.pdf",
    )

    assert recorded[0]["json"]["message"] == {
        "type": "document",
        "text": "el manual",
        "media_url": "data:application/pdf;base64,JVBERg==",
        "filename": "manual.pdf",
    }


async def test_gateway_error_code_passes_through(monkeypatch, contact):
    monkeypatch.setattr(settings, "channel_whatsapp_send_url", "http://gw:8000/send")
    _stub_http(
        monkeypatch,
        FakeResponse(
            502,
            {"detail": {"code": "WINDOW_EXPIRED", "message": "24h window closed"}},
        ),
    )

    with pytest.raises(channel_send.ChannelSendError) as exc:
        await channel_send.send_message(contact, text="hola")
    assert exc.value.code == "WINDOW_EXPIRED"
    assert exc.value.message == "24h window closed"


async def test_non_json_error_maps_to_send_failed(monkeypatch, contact):
    monkeypatch.setattr(settings, "channel_whatsapp_send_url", "http://gw:8000/send")
    _stub_http(monkeypatch, FakeResponse(500, json_raises=True))

    with pytest.raises(channel_send.ChannelSendError) as exc:
        await channel_send.send_message(contact, text="hola")
    assert exc.value.code == "SEND_FAILED"


async def test_unreachable_gateway_maps_to_gateway_unreachable(monkeypatch, contact):
    monkeypatch.setattr(settings, "channel_whatsapp_send_url", "http://gw:8000/send")
    _stub_http(monkeypatch, httpx.ConnectError("boom"))

    with pytest.raises(channel_send.ChannelSendError) as exc:
        await channel_send.send_message(contact, text="hola")
    assert exc.value.code == "GATEWAY_UNREACHABLE"
