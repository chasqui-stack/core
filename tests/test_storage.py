"""Sprint 6: app/core/storage.py — S3-compatible media storage (ADR-003).

No network: the boto3 client is stubbed at the _get_client seam.
"""

import uuid

import pytest

from app.core import storage
from app.core.config import settings


class FakeS3Client:
    def __init__(self):
        self.puts: list[dict] = []

    def put_object(self, **kwargs):
        self.puts.append(kwargs)

    def generate_presigned_url(self, operation, Params, ExpiresIn):
        return (
            f"https://bucket.test/{Params['Key']}"
            f"?sig=fake&expires={ExpiresIn}&op={operation}"
        )


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeS3Client()
    monkeypatch.setattr(storage, "_get_client", lambda: client)
    monkeypatch.setattr(storage, "_get_presign_client", lambda: client)
    return client


def test_ext_for_mime_known_map():
    assert storage.ext_for_mime("image/jpeg") == "jpg"
    assert storage.ext_for_mime("audio/ogg") == "ogg"
    assert storage.ext_for_mime("audio/mpeg") == "mp3"


def test_ext_for_mime_falls_back_to_mimetypes_then_bin():
    assert storage.ext_for_mime("image/gif") == "gif"  # mimetypes long tail
    assert storage.ext_for_mime("application/x-unknown-thing") == "bin"


def test_media_key_shape():
    contact_id, message_id = uuid.uuid4(), uuid.uuid4()
    key = storage.media_key(contact_id, message_id, "image/png")
    assert key == f"media/{contact_id}/{message_id}.png"
    assert storage.is_media_key(key)


def test_is_media_key_rejects_foreign_urls():
    assert not storage.is_media_key(None)
    assert not storage.is_media_key("")
    assert not storage.is_media_key("https://example.com/photo.jpg")
    assert not storage.is_media_key("data:image/png;base64,AAA")


def test_is_configured_requires_bucket_and_keys(monkeypatch):
    monkeypatch.setattr(settings, "storage_bucket", None)
    assert not storage.is_configured()

    monkeypatch.setattr(settings, "storage_bucket", "chasqui-media")
    monkeypatch.setattr(settings, "storage_access_key", "ak")
    monkeypatch.setattr(settings, "storage_secret_key", "sk")
    assert storage.is_configured()


async def test_put_media_sends_bucket_key_and_content_type(monkeypatch, fake_client):
    monkeypatch.setattr(settings, "storage_bucket", "chasqui-media")

    await storage.put_media("media/c/m.jpg", b"bytes", "image/jpeg")

    assert fake_client.puts == [
        {
            "Bucket": "chasqui-media",
            "Key": "media/c/m.jpg",
            "Body": b"bytes",
            "ContentType": "image/jpeg",
        }
    ]


def test_presigned_get_uses_default_expiry(monkeypatch, fake_client):
    monkeypatch.setattr(settings, "storage_bucket", "chasqui-media")

    url = storage.presigned_get("media/c/m.jpg")

    assert "media/c/m.jpg" in url
    assert f"expires={storage.PRESIGN_EXPIRES_SECONDS}" in url
