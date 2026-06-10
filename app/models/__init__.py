"""
Models package.

Imports all models so they register with SQLModel metadata and are
discovered by Alembic.
"""

from app.models.admin import AdminUser
from app.models.agent_config import AgentConfig
from app.models.contact import Contact
from app.models.conversation import Conversation
from app.models.memory import Memory
from app.models.message import Message

__all__ = ["AdminUser", "AgentConfig", "Contact", "Conversation", "Memory", "Message"]
