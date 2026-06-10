"""Tool-registry listing — feeds the admin Tools page (Sprint 5).

One read-only endpoint: every registered module with its tools, enable
state, config key and JSON Schema. The admin auto-renders a settings form
from `config_schema` — a new module's knobs appear in the UI with zero
admin-code changes (ARCHITECTURE §8).
"""

import logging

from fastapi import APIRouter, Depends
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.session import get_session
from app.modules import registry
from app.schemas.admin_tools import ModuleInfo, ToolInfo, ToolRegistryResponse
from app.services.agent_config_service import get_config, tool_enabled

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("", response_model=ToolRegistryResponse)
async def list_tools(session: AsyncSession = Depends(get_session)):
    config = await get_config(session)

    modules: list[ModuleInfo] = []
    for module in registry.get_modules():
        tools = [
            ToolInfo(
                name=tool.name,
                description=tool.description or "",
                enabled=tool_enabled(config, tool.name),
            )
            for tool in module.register_tools()
        ]

        config_key = getattr(module, "config_key", module.name)
        get_schema = getattr(module, "config_schema", None)
        schema = get_schema() if get_schema is not None else None

        json_schema = None
        current = None
        if schema is not None:
            json_schema = schema.model_json_schema()
            stored = (config.tool_config or {}).get(config_key, {})
            try:
                # Stored values merged over schema defaults
                current = schema.model_validate(stored).model_dump()
            except Exception:  # bad legacy data must not break the listing
                logger.warning(
                    "Invalid stored config for '%s'; showing defaults", config_key
                )
                current = schema().model_dump()

        modules.append(
            ModuleInfo(
                name=module.name,
                tools=tools,
                config_key=config_key,
                config_schema=json_schema,
                config=current,
            )
        )

    return ToolRegistryResponse(modules=modules)
