-- PostgreSQL initialization for Chasqui core
-- Executed on first database creation.

-- Extensions (tables are created by Alembic migrations)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

GRANT ALL PRIVILEGES ON DATABASE chasqui TO postgres;
