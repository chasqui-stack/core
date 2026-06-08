"""Admin service — authentication and admin-user management."""

import uuid
from datetime import datetime, timedelta

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import settings
from app.models.admin import AdminUser
from app.services.auth_service import create_token, hash_password, verify_password


def create_admin_access_token(admin_id: uuid.UUID, email: str, role: str) -> str:
    """Create an access token for an admin."""
    data = {"sub": str(admin_id), "email": email, "role": role}
    expires_delta = timedelta(minutes=settings.access_token_expire_minutes)
    return create_token(data, expires_delta, "admin_access")


def create_admin_refresh_token(admin_id: uuid.UUID) -> str:
    """Create a refresh token for an admin."""
    data = {"sub": str(admin_id), "is_admin": True}
    expires_delta = timedelta(days=settings.refresh_token_expire_days)
    return create_token(data, expires_delta, "admin_refresh")


async def authenticate_admin(
    session: AsyncSession, email: str, password: str
) -> AdminUser | None:
    """Authenticate an admin. Returns the AdminUser or None."""
    email = email.lower().strip()

    statement = select(AdminUser).where(AdminUser.email == email)
    result = await session.execute(statement)
    admin = result.scalar_one_or_none()

    if not admin or not admin.is_active:
        return None
    if not verify_password(password, admin.password_hash):
        return None

    admin.last_login_at = datetime.utcnow()
    session.add(admin)
    await session.commit()
    await session.refresh(admin)
    return admin


async def get_admin_by_id(session: AsyncSession, admin_id: uuid.UUID) -> AdminUser | None:
    """Get an admin by id."""
    statement = select(AdminUser).where(AdminUser.id == admin_id)
    result = await session.execute(statement)
    return result.scalar_one_or_none()


async def create_admin(
    session: AsyncSession,
    email: str,
    password: str,
    full_name: str,
    role: str = "super_admin",
) -> AdminUser:
    """Create a new admin user."""
    email = email.lower().strip()

    statement = select(AdminUser).where(AdminUser.email == email)
    result = await session.execute(statement)
    if result.scalar_one_or_none():
        raise ValueError("Este email de admin ya está registrado")

    admin = AdminUser(
        email=email,
        password_hash=hash_password(password),
        full_name=full_name,
        role=role,
    )
    session.add(admin)
    await session.commit()
    await session.refresh(admin)
    return admin
