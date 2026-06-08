from fastapi import Header, HTTPException, status

from app.core.config import settings

API_KEY_HEADER_NAME = "X-Internal-API-Key"


async def verify_api_key(
    x_internal_api_key: str | None = Header(None, alias=API_KEY_HEADER_NAME),
) -> str:
    if not settings.internal_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key no configurada en el servidor",
        )

    if not x_internal_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Falta el header {API_KEY_HEADER_NAME}",
        )

    if x_internal_api_key != settings.internal_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key inválida",
        )

    return x_internal_api_key
