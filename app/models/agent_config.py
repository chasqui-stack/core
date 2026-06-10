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


DEFAULT_SYSTEM_PROMPT = (
    "Eres el asistente virtual de la empresa. Atiendes a clientes por chat "
    "de forma cordial, clara y concisa (este es un canal de mensajería: "
    "respuestas cortas, sin Markdown pesado).\n\n"
    "Reglas:\n"
    "- Responde siempre en el idioma del usuario.\n"
    "- Usa las herramientas disponibles cuando ayuden a responder.\n"
    "- Si no sabes algo, dilo honestamente; no inventes datos.\n"
    "- Si el usuario pide hablar con una persona, usa la herramienta de "
    "derivación a humano."
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
