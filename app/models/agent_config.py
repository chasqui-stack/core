"""Agent configuration — the DB-editable knobs behind the orchestrator.

Singleton table (Chasqui = one project per deployment, §4): the system prompt
the operator edits from the admin (Sprint 5), which tools are enabled, and
per-module settings. Migration 003 seeds the default row.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


def _utcnow_naive() -> datetime:
    """Naive UTC now (asyncpg requires naive datetimes for TIMESTAMP columns)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# Seed value — operators rewrite it from the admin panel in their own
# language/voice. "Always reply in the user's language" is what localizes
# the agent (all LLM-facing strings in the codebase are English).
DEFAULT_SYSTEM_PROMPT = (
    "You are the company's virtual assistant. You serve customers over chat "
    "in a cordial, clear and concise way (this is a messaging channel: "
    "short replies, no heavy Markdown).\n\n"
    "Rules:\n"
    "- Always reply in the user's language.\n"
    "- Use the available tools whenever they help you answer.\n"
    "- If you don't know something, say so honestly; never make up facts.\n"
    "- If the user asks to talk to a person, use the human handoff tool."
)


class AgentConfig(SQLModel, table=True):
    """Editable agent settings (system prompt, tool enable/config)."""

    __tablename__ = "agent_config"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    # The persona/rules the orchestrator injects as the system message
    system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        sa_column=Column(Text, nullable=False),
    )

    # {tool_name: bool} — missing key = enabled (new modules work out of the box)
    enabled_tools: dict = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
    )

    # {module_name: {…}} — validated by each module's config_schema()
    tool_config: dict = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
    )

    updated_at: datetime = Field(default_factory=_utcnow_naive, nullable=False)
