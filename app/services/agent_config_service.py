"""Agent config access — the singleton row the admin edits (Sprint 5 UI).

Migration 003 seeds it; get_config() self-heals if the row is missing
(fresh test DBs created via create_all have no seed).
"""

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models import AgentConfig


async def get_config(session: AsyncSession) -> AgentConfig:
    """Return the singleton agent config, creating the default if absent."""
    result = await session.exec(select(AgentConfig).limit(1))
    config = result.first()
    if config is None:
        config = AgentConfig()
        session.add(config)
        await session.flush()
    return config


def tool_enabled(config: AgentConfig, tool_name: str) -> bool:
    """Missing key = enabled, so new modules work without touching config."""
    return bool(config.enabled_tools.get(tool_name, True))
