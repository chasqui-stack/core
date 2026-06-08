# --------- Builder Stage ---------
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

# Set environment variables for uv
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies first (for better layer caching)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# Copy the project source code
COPY . /app

# Install the project in non-editable mode
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable

# --------- Final Stage ---------
FROM python:3.13-slim-bookworm

# Create a non-root user for security
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --shell /bin/bash --create-home app

# Copy the virtual environment from the builder stage
COPY --from=builder --chown=app:app /app/.venv /app/.venv

# Fix Python symlinks to point to system Python (uv creates symlinks to builder's Python)
RUN rm -f /app/.venv/bin/python /app/.venv/bin/python3 /app/.venv/bin/python3.13 && \
    ln -s /usr/local/bin/python3.13 /app/.venv/bin/python && \
    ln -s python /app/.venv/bin/python3 && \
    ln -s python /app/.venv/bin/python3.13

# Set the working directory
WORKDIR /app

# Copy the source code from the builder stage
COPY --from=builder --chown=app:app /app/app ./app
COPY --from=builder --chown=app:app /app/alembic.ini ./alembic.ini
COPY --from=builder --chown=app:app /app/alembic ./alembic
COPY --from=builder --chown=app:app /app/scripts ./scripts

# Ensure the virtual environment is in the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Switch to the non-root user
USER app

# Run with gunicorn for production
CMD ["gunicorn", "app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8090"]
