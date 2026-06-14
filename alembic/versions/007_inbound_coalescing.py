"""inbound debounce + coalescing — conversations.debounce_due_at + messages.processed_at (ADR-008)

Revision ID: 007_inbound_coalescing
Revises: 006_conversation_mode_leads
Create Date: 2026-06-14

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "007_inbound_coalescing"
down_revision: Union[str, Sequence[str], None] = "006_conversation_mode_leads"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # When the current debounce window elapses (NULL = nothing pending). The
    # coalesce worker only ever scans armed rows, so the index is partial.
    op.add_column(
        "conversations", sa.Column("debounce_due_at", sa.DateTime(), nullable=True)
    )
    op.create_index(
        "ix_conversations_debounce_due",
        "conversations",
        ["debounce_due_at"],
        postgresql_where=sa.text("debounce_due_at IS NOT NULL"),
    )

    # NULL = pending (not yet folded into a turn); set when a coalesced turn
    # consumes the message. Watermarking on created_at would be fragile (the
    # gateway clock is not monotonic), so we mark each row instead.
    op.add_column(
        "messages", sa.Column("processed_at", sa.DateTime(), nullable=True)
    )
    op.create_index(
        "ix_messages_pending_inbound",
        "messages",
        ["conversation_id", "created_at"],
        postgresql_where=sa.text("processed_at IS NULL AND direction = 'in'"),
    )

    # Backfill: every existing inbound is historical — mark it processed so a
    # freshly-started worker never mistakes old history for a pending batch.
    op.execute(
        "UPDATE messages SET processed_at = created_at WHERE direction = 'in'"
    )


def downgrade() -> None:
    op.drop_index("ix_messages_pending_inbound", table_name="messages")
    op.drop_column("messages", "processed_at")
    op.drop_index("ix_conversations_debounce_due", table_name="conversations")
    op.drop_column("conversations", "debounce_due_at")
