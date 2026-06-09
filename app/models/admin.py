"""Platform admin users - separate from tenant users for security."""

import uuid
from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


class AdminUser(SQLModel, table=True):
    """
    Platform administrator - has access to all organizations and users.

    Completely isolated from tenant User table for security.
    Different auth flow, different JWT claims.
    """

    __tablename__ = "admin_users"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    # Credentials
    email: str = Field(
        max_length=255,
        unique=True,
        index=True,
        nullable=False,
        description="Admin email (unique, stored lowercase)",
    )
    password_hash: str = Field(max_length=255, nullable=False)

    # Profile
    full_name: str = Field(max_length=100, nullable=False)

    # Role (for future expansion: super_admin, support, viewer)
    role: str = Field(
        default="super_admin",
        max_length=20,
        nullable=False,
        description="Admin role: 'super_admin'",
    )

    # Account status
    is_active: bool = Field(default=True)

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )
    last_login_at: datetime | None = Field(default=None)
