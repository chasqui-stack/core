"""Unit tests for password hashing and JWT (no DB)."""

import uuid
from datetime import timedelta

import pytest

from app.services.auth_service import (
    create_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.services.admin_service import (
    create_admin_access_token,
    create_admin_refresh_token,
)


def test_password_hash_roundtrip():
    h = hash_password("secret123")
    assert h != "secret123"
    assert verify_password("secret123", h)
    assert not verify_password("wrong", h)


def test_token_roundtrip_and_type():
    token = create_token({"sub": "abc"}, timedelta(minutes=5), "access")
    payload = decode_token(token)
    assert payload["sub"] == "abc"
    assert payload["type"] == "access"


def test_decode_invalid_token_raises():
    with pytest.raises(ValueError):
        decode_token("not-a-jwt")


def test_admin_access_token_claims():
    admin_id = uuid.uuid4()
    token = create_admin_access_token(admin_id, "admin@chasqui.local", "super_admin")
    payload = decode_token(token)
    assert payload["sub"] == str(admin_id)
    assert payload["email"] == "admin@chasqui.local"
    assert payload["role"] == "super_admin"
    assert payload["type"] == "admin_access"


def test_admin_refresh_token_type():
    token = create_admin_refresh_token(uuid.uuid4())
    assert decode_token(token)["type"] == "admin_refresh"
