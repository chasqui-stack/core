"""Schemas for the agent-config admin endpoints (/admin/config)."""

import uuid
from datetime import datetime

from pydantic import BaseModel


class AgentConfigResponse(BaseModel):
    id: uuid.UUID
    system_prompt: str
    enabled_tools: dict[str, bool]
    tool_config: dict[str, dict]
    updated_at: datetime


class AgentConfigUpdate(BaseModel):
    """Partial update — any subset of the editable fields.

    `enabled_tools` and `tool_config` REPLACE the stored dicts (the UI sends
    the full map it loaded; per-key merging would make deletes impossible).
    """

    system_prompt: str | None = None
    enabled_tools: dict[str, bool] | None = None
    tool_config: dict[str, dict] | None = None
