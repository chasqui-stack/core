"""FastAPI dependencies for admin authentication.

End users (WhatsApp contacts) never authenticate — only platform admins do.
"""

import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.schemas.admin import AdminAuthUser
from app.services.auth_service import decode_token

# OAuth2 scheme for admins (drives the "Authorize" button in /docs)
admin_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/admin/auth/login")


async def get_current_admin(
    token: Annotated[str, Depends(admin_oauth2_scheme)],
) -> AdminAuthUser:
    """Resolve the current admin from a JWT (type='admin_access')."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate admin credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_token(token)

        if payload.get("type") != "admin_access":
            raise credentials_exception

        admin_id = payload.get("sub")
        email = payload.get("email")
        role = payload.get("role")

        if not admin_id or not email or not role:
            raise credentials_exception

        return AdminAuthUser(
            admin_id=uuid.UUID(admin_id),
            email=email,
            role=role,
        )

    except (ValueError, KeyError):
        raise credentials_exception


def require_admin(min_role: str = "super_admin"):
    """Dependency factory requiring an authenticated admin (role hierarchy ready)."""

    async def admin_checker(
        admin: AdminAuthUser = Depends(get_current_admin),
    ) -> AdminAuthUser:
        if admin.role != min_role and min_role == "super_admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient admin permissions",
            )
        return admin

    return admin_checker


CurrentAdmin = Annotated[AdminAuthUser, Depends(get_current_admin)]
