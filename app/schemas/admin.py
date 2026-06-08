"""Admin auth request/response schemas."""

import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr


class AdminLoginRequest(BaseModel):
    """Admin login request (JSON body)."""

    email: EmailStr
    password: str


class AdminAuthUser(BaseModel):
    """Admin context extracted from a JWT."""

    admin_id: uuid.UUID
    email: str
    role: str


class AdminLoginResponse(BaseModel):
    """Login/refresh response with tokens."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class AdminRefreshRequest(BaseModel):
    """Refresh token request."""

    refresh_token: str


class AdminUserResponse(BaseModel):
    """Admin user info."""

    id: uuid.UUID
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: datetime
    last_login_at: datetime | None


class PaginatedResponse[T](BaseModel):
    """Generic paginated response (reusable for future list endpoints)."""

    items: list[T]
    total: int
    page: int
    page_size: int
    total_pages: int
