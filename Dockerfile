# syntax=docker/dockerfile:1

# Slack Q&A bot: a FastAPI / uvicorn webhook (Slack Events API). Built with uv
# against the committed lockfile, so the image resolves the exact same versions
# CI does.
FROM python:3.12-slim

# uv binary from its official image (pin by digest/tag in a real deploy).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 1) Runtime dependencies only, as a layer cached on the lockfile alone.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# 2) App code, then install the project itself (dev tools excluded).
COPY README.md ./
COPY app/ ./app/
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Run as a non-root user. data/ holds the read-only knowledge base and the
# checkpoint DB (both gitignored); mount it at runtime.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 3000
CMD ["python", "-m", "app.slack_app"]
