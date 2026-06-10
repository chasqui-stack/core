"""Per-turn context injected into the agent runtime (ToolRuntime.context).

Tools and middleware read this instead of importing services directly:
the DB session (tools may persist), who is talking, and the editable
agent config (enabled tools / per-module settings).
"""

import uuid
from dataclasses import dataclass

from sqlmodel.ext.asyncio.session import AsyncSession

from app.models import AgentConfig


@dataclass
class TurnContext:
    """Everything a tool/middleware may need during one agent turn."""

    session: AsyncSession
    contact_id: uuid.UUID
    conversation_id: uuid.UUID
    config: AgentConfig
