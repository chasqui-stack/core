"""agent_config — DB-editable system prompt + tool enable/config (singleton)

Revision ID: 003_agent_config
Revises: 002_domain
Create Date: 2026-06-09

"""
import uuid
from datetime import datetime, timezone

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "003_agent_config"
down_revision: Union[str, Sequence[str], None] = "002_domain"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Keep in sync with app.models.agent_config.DEFAULT_SYSTEM_PROMPT
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


def upgrade() -> None:
    table = op.create_table(
        "agent_config",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("system_prompt", sa.Text(), nullable=False),
        sa.Column("enabled_tools", JSONB(), nullable=False, server_default="{}"),
        sa.Column("tool_config", JSONB(), nullable=False, server_default="{}"),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    # Seed the singleton row so a fresh deployment answers out of the box
    op.bulk_insert(
        table,
        [
            {
                "id": uuid.uuid4(),
                "system_prompt": DEFAULT_SYSTEM_PROMPT,
                "enabled_tools": {},
                "tool_config": {},
                "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
            }
        ],
    )


def downgrade() -> None:
    op.drop_table("agent_config")
