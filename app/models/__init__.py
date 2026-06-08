"""
Models package.

Imports all models so they register with SQLModel metadata and are
discovered by Alembic. Add domain models (contacts, conversations,
messages, memories) here as they land.
"""

from app.models.admin import AdminUser

__all__ = ["AdminUser"]
