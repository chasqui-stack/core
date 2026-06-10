"""Admin authentication endpoints."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.dependencies import CurrentAdmin
from app.db.session import get_session
from app.schemas.admin import (
    AdminLoginResponse,
    AdminRefreshRequest,
    AdminUserResponse,
)
from app.services.admin_service import (
    authenticate_admin,
    create_admin_access_token,
    create_admin_refresh_token,
    get_admin_by_id,
)
from app.services.auth_service import decode_token

router = APIRouter()


@router.post("/login", response_model=AdminLoginResponse)
async def admin_login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_session),
):
    """
    Authenticate an admin user.

    For Swagger UI: username = email, password = password
    Returns access and refresh tokens on success.
    """
    admin = await authenticate_admin(session, form_data.username, form_data.password)

    if not admin:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    access_token = create_admin_access_token(admin.id, admin.email, admin.role)
    refresh_token = create_admin_refresh_token(admin.id)

    return AdminLoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.post("/refresh", response_model=AdminLoginResponse)
async def admin_refresh(
    request: AdminRefreshRequest,
    session: AsyncSession = Depends(get_session),
):
    """Refresh an admin access token."""
    try:
        payload = decode_token(request.refresh_token)

        # Verify it's an admin refresh token
        if payload.get("type") != "admin_refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )

        admin_id = payload.get("sub")
        if not admin_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )

        # Get admin to verify they still exist and are active
        admin = await get_admin_by_id(session, uuid.UUID(admin_id))
        if not admin or not admin.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Admin not found or inactive",
            )

        # Create new tokens
        access_token = create_admin_access_token(admin.id, admin.email, admin.role)
        refresh_token = create_admin_refresh_token(admin.id)

        return AdminLoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
        )

    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )


@router.get("/me", response_model=AdminUserResponse)
async def get_current_admin_info(
    current_admin: CurrentAdmin,
    session: AsyncSession = Depends(get_session),
):
    """Get current admin user information."""
    admin = await get_admin_by_id(session, current_admin.admin_id)

    if not admin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Admin not found",
        )

    return AdminUserResponse(
        id=admin.id,
        email=admin.email,
        full_name=admin.full_name,
        role=admin.role,
        is_active=admin.is_active,
        created_at=admin.created_at,
        last_login_at=admin.last_login_at,
    )
