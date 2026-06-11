"""conversations.mode (agent|human) + leads table (handoff module) — ADR-004

Revision ID: 006_conversation_mode_leads
Revises: 005_vector_indexes
Create Date: 2026-06-10

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = "006_conversation_mode_leads"
down_revision: Union[str, Sequence[str], None] = "005_vector_indexes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Conversation mode — a real indexed column, not a JSONB flag (ADR-004).
    op.add_column(
        "conversations",
        sa.Column("mode", sa.String(length=8), nullable=False, server_default="agent"),
    )
    op.create_index("ix_conversations_mode", "conversations", ["mode"])

    # Module-owned leads table (handoff module, register_models()).
    op.create_table(
        "leads",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("contact_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("interest", sa.String(length=500), nullable=True),
        sa.Column("email", sa.String(length=255), nullable=True),
        sa.Column("phone", sa.String(length=32), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("extra", JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["contact_id"], ["contacts.id"]),
    )
    op.create_index("ix_leads_contact_id", "leads", ["contact_id"])


def downgrade() -> None:
    op.drop_index("ix_leads_contact_id", table_name="leads")
    op.drop_table("leads")
    op.drop_index("ix_conversations_mode", table_name="conversations")
    op.drop_column("conversations", "mode")
