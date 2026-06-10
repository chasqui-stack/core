"""Agent config endpoints — the singleton row the operator edits (Sprint 5).

GET returns the row (self-healing if a fresh DB has no seed); PUT is a
partial update. Writes are validated against the tool registry so the admin
UI can never persist a typo'd tool name or an out-of-range knob:

- `enabled_tools` keys must be registered tool names.
- each `tool_config` key must match a module's `config_key`, and its value
  must validate against that module's `config_schema()`.
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import ValidationError
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.session import get_session
from app.modules import registry
from app.schemas.admin_config import AgentConfigResponse, AgentConfigUpdate
from app.services.agent_config_service import get_config

logger = logging.getLogger(__name__)

router = APIRouter()


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _registered_tool_names() -> set[str]:
    return {tool.name for tool in registry.get_tools()}


def _config_schemas() -> dict[str, type]:
    """Map config_key -> Pydantic schema class for every module with knobs."""
    schemas: dict[str, type] = {}
    for module in registry.get_modules():
        get_schema = getattr(module, "config_schema", None)
        if get_schema is None:
            continue
        schema = get_schema()
        if schema is None:
            continue
        key = getattr(module, "config_key", module.name)
        schemas[key] = schema
    return schemas


def _validate_enabled_tools(enabled_tools: dict[str, bool]) -> None:
    unknown = set(enabled_tools) - _registered_tool_names()
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Unknown tools in enabled_tools: {sorted(unknown)}",
        )


def _validate_tool_config(tool_config: dict[str, dict]) -> None:
    schemas = _config_schemas()
    for key, value in tool_config.items():
        schema = schemas.get(key)
        if schema is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Unknown tool_config key: '{key}'",
            )
        try:
            schema.model_validate(value)
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Invalid config for '{key}': {exc.errors()}",
            )


@router.get("", response_model=AgentConfigResponse)
async def read_config(session: AsyncSession = Depends(get_session)):
    config = await get_config(session)
    await session.commit()  # persist the self-heal seed if it just happened
    return config


@router.put("", response_model=AgentConfigResponse)
async def update_config(
    payload: AgentConfigUpdate, session: AsyncSession = Depends(get_session)
):
    if payload.enabled_tools is not None:
        _validate_enabled_tools(payload.enabled_tools)
    if payload.tool_config is not None:
        _validate_tool_config(payload.tool_config)

    config = await get_config(session)
    if payload.system_prompt is not None:
        config.system_prompt = payload.system_prompt
    if payload.enabled_tools is not None:
        config.enabled_tools = payload.enabled_tools
    if payload.tool_config is not None:
        config.tool_config = payload.tool_config
    config.updated_at = _utcnow_naive()

    session.add(config)
    await session.flush()  # refresh() does NOT autoflush — flush first
    await session.commit()
    logger.info("Agent config updated (fields: %s)", sorted(payload.model_dump(exclude_none=True)))
    return config
