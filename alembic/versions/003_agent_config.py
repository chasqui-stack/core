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
    "You are the company's virtual assistant. You serve customers over chat "
    "in a cordial, clear and concise way (this is a messaging channel: "
    "short replies, no heavy Markdown).\n\n"
    "Rules:\n"
    "- Always reply in the user's language.\n"
    "- Use the available tools whenever they help you answer.\n"
    "- If you don't know something, say so honestly; never make up facts.\n"
    "- If the user asks to talk to a person, use the human handoff tool."
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
