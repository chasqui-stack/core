"""Admin controllers package."""

from app.controllers.admin.auth import router as auth_router
from app.controllers.admin.config import router as config_router
from app.controllers.admin.contacts import router as contacts_router
from app.controllers.admin.tools import router as tools_router

__all__ = ["auth_router", "config_router", "contacts_router", "tools_router"]
