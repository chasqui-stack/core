from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.controllers import base, ingest
from app.controllers.admin import auth_router
from app.core.config import settings
from app.core.middleware import setup_cors
from app.db.session import close_db, init_db
from app.modules import registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    registry.discover()  # tool modules under app/modules/ (ARCHITECTURE §8)
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

# NOTE: admin config routes (prompts, FAQ-RAG, tools) land in later sprints.
