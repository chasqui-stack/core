import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI

from app.controllers import base, conversations, ingest
from app.controllers.admin import (
    auth_router,
    config_router,
    contacts_router,
    media_router,
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

    # STT fallback (ADR-010): a provider set without a usable key/base_url means
    # voice notes silently keep the text fallback — warn so it isn't a surprise.
    if settings.stt_provider:
        from app.services import transcription

        if not transcription.stt_enabled():
            logger.warning(
                "STT_PROVIDER=%s but STT is not usable (missing STT_API_KEY or an "
                "unresolved base URL) — voice notes will keep the text fallback. "
                "Set STT_API_KEY (and STT_BASE_URL for a non-default provider).",
                settings.stt_provider,
            )

    # Inbound coalescing (ADR-008): start the deferred-dispatch worker. Replies
    # go out via the send seam, so warn loudly if no channel send URL is set.
    worker_stop: asyncio.Event | None = None
    worker_task: asyncio.Task | None = None
    if settings.inbound_debounce_seconds > 0:
        from app.services import channel_send, coalesce_worker

        if not any(
            channel_send.send_url_for(ch) for ch in ("whatsapp", "telegram", "web")
        ):
            logger.warning(
                "INBOUND_DEBOUNCE_SECONDS=%s but no CHANNEL_<CH>_SEND_URL is set — "
                "coalesced replies have nowhere to go. Set the send URL for each "
                "active channel, or set INBOUND_DEBOUNCE_SECONDS=0 for synchronous "
                "replies.",
                settings.inbound_debounce_seconds,
            )
        worker_stop = asyncio.Event()
        worker_task = asyncio.create_task(coalesce_worker.run_loop(worker_stop))

    yield

    if worker_task is not None:
        worker_stop.set()
        await worker_task
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

# Internal conversation read (ADR-011) — gateway-facing, INTERNAL_API_KEY.
# Generic, channel-scoped; lets a gateway rehydrate a thread without admin JWT.
app.include_router(conversations.router, tags=["internal"])

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
admin_router.include_router(media_router, prefix="/media", tags=["admin-media"])
app.include_router(admin_router)

# Module admin routes — every module's register_admin_routes() mounts under
# /admin/modules/<name>, JWT-protected as a whole (§8)
modules_router = APIRouter(
    prefix="/admin/modules", dependencies=[Depends(get_current_admin)]
)
registry.mount_admin_routes(modules_router)
app.include_router(modules_router)
