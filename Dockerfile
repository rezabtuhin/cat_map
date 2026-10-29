# syntax=docker/dockerfile:1
ARG UV_VERSION=0.11.26
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/
WORKDIR /app

# Keep dependencies cached when only application code changes.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --group dev --no-install-project

RUN groupadd --gid 10001 matcher \
    && useradd --uid 10001 --gid matcher --create-home matcher \
    && mkdir -p /app/models /app/artifacts /app/var /app/reports /home/matcher/.cache \
    && chown -R matcher:matcher /app /home/matcher
COPY --chown=matcher:matcher src/ ./src/
COPY --chown=matcher:matcher scripts/ ./scripts/
COPY --chown=matcher:matcher config/ ./config/
COPY --chown=matcher:matcher data/ ./data/
COPY --chown=matcher:matcher tests/ ./tests/
COPY --chown=matcher:matcher docker/ ./docker/
COPY --chown=matcher:matcher google_category_list.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --group dev \
    && sha256sum pyproject.toml uv.lock > /opt/project-inputs.sha256

ENV HOME=/home/matcher \
    XDG_CACHE_HOME=/home/matcher/.cache \
    UV_NO_SYNC=1
USER matcher
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20m --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3).close()"]
ENTRYPOINT ["python", "/app/docker/entrypoint.py"]
CMD ["serve"]
