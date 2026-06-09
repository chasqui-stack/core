"""Test fixtures — DB isolation via transactional rollback.

Strategy (SQLAlchemy docs, "Joining a Session into an External Transaction"):

1. A dedicated `<postgres_db>_test` database is created once per session
   (config comes from the same env as the app; override POSTGRES_* env vars
   in CI to point elsewhere). Dev/prod data is never touched.
2. Each test runs inside an OUTER transaction on a single connection that is
   ROLLED BACK at the end. App/test sessions join it with
   `join_transaction_mode="create_savepoint"`, so the app's `commit()` only
   releases a savepoint — nothing ever truly lands in the database.

Result: tests can't contaminate each other (or the test DB) by design — no
TRUNCATE between tests, no cleanup plugins needed.
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

    # 2) Extensions + schema, and a pristine baseline (in case an older
    #    truncate-based run left rows behind)
    engine = create_async_engine(TEST_DB_URL, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(SQLModel.metadata.create_all)
        tables = ", ".join(t.name for t in SQLModel.metadata.sorted_tables)
        await conn.execute(text(f"TRUNCATE {tables} CASCADE"))
    await engine.dispose()


@pytest.fixture(scope="session", autouse=True)
def _test_db() -> None:
    """Create the test database + schema once per test session."""
    asyncio.run(_prepare_database())


@pytest.fixture
async def connection():
    """One connection + outer transaction per test; rolled back at the end."""
    engine = create_async_engine(TEST_DB_URL, poolclass=NullPool)
    async with engine.connect() as conn:
        transaction = await conn.begin()
        yield conn
        await transaction.rollback()
    await engine.dispose()


def _savepoint_session(connection) -> AsyncSession:
    """Session that joins the test's outer transaction via savepoints."""
    return AsyncSession(
        bind=connection,
        join_transaction_mode="create_savepoint",
        expire_on_commit=False,
    )


@pytest.fixture
async def session(connection):
    """Session for asserting DB state — sees the app's (savepoint) commits."""
    async with _savepoint_session(connection) as session:
        yield session


@pytest.fixture
async def client(connection):
    """HTTP client against the app, with get_session joined to the test txn."""

    async def override_get_session():
        async with _savepoint_session(connection) as session:
            try:
                yield session
                await session.commit()  # releases a savepoint, never a real commit
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_session] = override_get_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()
