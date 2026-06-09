"""Test fixtures.

DB-backed tests run against a dedicated `<postgres_db>_test` database,
created on demand (extension + schema via SQLModel.metadata). Tables are
truncated before each test so tests stay independent.
"""

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

# Import models so SQLModel.metadata knows every table
import app.models  # noqa: F401
from app.core.config import settings
from app.db.session import get_session
from app.main import app

TEST_DB = f"{settings.postgres_db}_test"
TEST_DB_URL = (
    f"postgresql+asyncpg://{settings.postgres_user}:{settings.postgres_password}"
    f"@{settings.postgres_host}:{settings.postgres_port}/{TEST_DB}"
)

DOMAIN_TABLES = ["messages", "memories", "conversations", "contacts", "admin_users"]


async def _prepare_database() -> None:
    # 1) Create the test DB if missing (CREATE DATABASE needs autocommit)
    admin_engine = create_async_engine(settings.database_url, isolation_level="AUTOCOMMIT")
    async with admin_engine.connect() as conn:
        exists = await conn.scalar(
            text("SELECT 1 FROM pg_database WHERE datname = :db"), {"db": TEST_DB}
        )
        if not exists:
            await conn.execute(text(f'CREATE DATABASE "{TEST_DB}"'))
    await admin_engine.dispose()

    # 2) Extensions + schema
    engine = create_async_engine(TEST_DB_URL, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(SQLModel.metadata.create_all)
    await engine.dispose()


@pytest.fixture(scope="session", autouse=True)
def _test_db() -> None:
    """Create the test database + schema once per test session."""
    asyncio.run(_prepare_database())


@pytest.fixture
async def engine():
    """Per-test engine bound to the test's event loop."""
    engine = create_async_engine(TEST_DB_URL, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE {', '.join(DOMAIN_TABLES)} CASCADE"))
    yield engine
    await engine.dispose()


@pytest.fixture
async def session(engine):
    """Plain session for asserting DB state from tests."""
    async with AsyncSession(engine, expire_on_commit=False) as session:
        yield session


@pytest.fixture
async def client(engine):
    """HTTP client against the app, with get_session pointed at the test DB."""

    async def override_get_session():
        async with AsyncSession(engine, expire_on_commit=False) as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_session] = override_get_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()
