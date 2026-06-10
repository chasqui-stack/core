"""Schemas for the tool-registry admin listing (/admin/tools)."""

from pydantic import BaseModel


class ToolInfo(BaseModel):
    name: str
    description: str
    enabled: bool  # agent_config.enabled_tools, missing key = enabled


class ModuleInfo(BaseModel):
    name: str
    tools: list[ToolInfo]
    # Key inside agent_config.tool_config holding this module's settings
    # (module.config_key, defaults to the module name)
    config_key: str
    # JSON Schema from config_schema().model_json_schema(), None if the
    # module has no knobs — the admin auto-renders a form from it
    config_schema: dict | None
    # Current effective values (stored ones validated against the schema,
    # falling back to schema defaults) — None if no schema
    config: dict | None


class ToolRegistryResponse(BaseModel):
    modules: list[ModuleInfo]
