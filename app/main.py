import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI

from app.controllers import base, ingest
from app.controllers.admin import (
    auth_router,
    config_router,
    contacts_router,
    tools_router,
)
from app.core.config import settings
from app.core.dependencies import get_current_admin
from app.core.middleware import setup_cors
from app.db.session import close_db, init_db
from app.modules import registry

# Discover tool modules at import time (idempotent) so module admin routes
# exist before the app starts serving (ARCHITECTURE §8).
registry.discover()


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    registry.discover()  # idempotent — covers reload/edge import orders
    from app.core.vector_search import index_strategy

    if index_strategy() == "exact":  # > 4000 dims: no ANN index possible (ADR-001)
        logger.warning(
            "EMBEDDING_DIM=%s exceeds halfvec's 4000-dim HNSW limit: pgvector "
            "searches run as exact scans (fine small, slow at scale).",
            settings.embedding_dim,
        )
    yield
    await close_db()


app = FastAPI(
    title=settings.app_name,
    description="Chasqui core — WhatsApp AI agent backend",
    version="0.1.0",
    debug=settings.debug,
    lifespan=lifespan,
)

setup_cors(app)

# Public / health
app.include_router(base.router)

# Canonical entry point — the only seam gateways talk to (§5)
app.include_router(ingest.router, tags=["ingest"])

# Admin authentication (admins only — end users never authenticate)
app.include_router(auth_router, prefix="/admin/auth", tags=["admin-auth"])

# Admin panel API (Sprint 5) — agent config, tool registry, conversation
# inspection. JWT-protected as a whole.
admin_router = APIRouter(prefix="/admin", dependencies=[Depends(get_current_admin)])
admin_router.include_router(config_router, prefix="/config", tags=["admin-config"])
admin_router.include_router(tools_router, prefix="/tools", tags=["admin-tools"])
admin_router.include_router(
    contacts_router, prefix="/contacts", tags=["admin-contacts"]
)
app.include_router(admin_router)

# Module admin routes — every module's register_admin_routes() mounts under
# /admin/modules/<name>, JWT-protected as a whole (§8)
modules_router = APIRouter(
    prefix="/admin/modules", dependencies=[Depends(get_current_admin)]
)
registry.mount_admin_routes(modules_router)
app.include_router(modules_router)
