"""core domain: contacts, conversations, messages, memories (BSUID-first)

Revision ID: 002_domain
Revises: 001_initial
Create Date: 2026-06-09

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = "002_domain"
down_revision: Union[str, Sequence[str], None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Provision-time: the vector width comes from .env (EMBEDDING_DIM) at the
# moment this migration first runs. Changing it later requires a column
# migration + re-embedding. See ADR-001 (parent repo, docs/design/).
from app.core.config import settings

EMBEDDING_DIM = settings.embedding_dim


def upgrade() -> None:
    # Contacts — BSUID-first identity: unique (channel, external_id), wa_id optional
    op.create_table(
        "contacts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("channel", sa.String(length=32), nullable=False),
        sa.Column("external_id", sa.String(length=255), nullable=False),
        sa.Column("wa_id", sa.String(length=32), nullable=True),
        sa.Column("display_name", sa.String(length=255), nullable=True),
        sa.Column("metadata", JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("channel", "external_id", name="uq_contacts_channel_external_id"),
    )
    op.create_index("ix_contacts_channel", "contacts", ["channel"])
    op.create_index("ix_contacts_wa_id", "contacts", ["wa_id"])

    # Conversations — single thread per contact (unique contact_id)
    op.create_table(
        "conversations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("contact_id", sa.Uuid(), nullable=False),
        sa.Column("conversation_state", JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["contact_id"], ["contacts.id"]),
        sa.UniqueConstraint("contact_id"),
    )
    op.create_index("ix_conversations_contact_id", "conversations", ["contact_id"], unique=True)

    # Messages — full in/out history (the LLM context)
    op.create_table(
        "messages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("direction", sa.String(length=3), nullable=False),
        sa.Column("type", sa.String(length=20), nullable=False),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("media_url", sa.String(length=1024), nullable=True),
        sa.Column("metadata", JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
    )
    op.create_index(
        "ix_messages_conversation_created", "messages", ["conversation_id", "created_at"]
    )

    # Memories — long-term facts/summaries; embedding filled by Sprint 3/4 extractor
    op.create_table(
        "memories",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("contact_id", sa.Uuid(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["contact_id"], ["contacts.id"]),
    )
    op.create_index("ix_memories_contact_id", "memories", ["contact_id"])
    # NOTE: vector (HNSW/IVFFlat) index deferred until retrieval is wired (Sprint 3/4)


def downgrade() -> None:
    op.drop_index("ix_memories_contact_id", table_name="memories")
    op.drop_table("memories")
    op.drop_index("ix_messages_conversation_created", table_name="messages")
    op.drop_table("messages")
    op.drop_index("ix_conversations_contact_id", table_name="conversations")
    op.drop_table("conversations")
    op.drop_index("ix_contacts_wa_id", table_name="contacts")
    op.drop_index("ix_contacts_channel", table_name="contacts")
    op.drop_table("contacts")
