# Docker development

Install Docker Engine with Compose v2, or Docker Desktop, then run from the
repository root:

```bash
docker compose up --build
```

This builds a Python 3.11 development image with locked dependencies, CPU-only
PyTorch, Ruff, and pytest. First startup downloads the pinned model snapshots and
builds/activates the Google taxonomy bundle inside Docker. This needs network
access, disk space for dependencies and weights, and time for CPU embedding.
Subsequent starts reuse the saved models and release. Inference stays local.

Open `http://127.0.0.1:8000/docs` after initialization. The health check uses
`/health/ready`; a running container can still be initializing. Logs explain which
stage is active:

```bash
docker compose logs -f api
docker compose ps
curl http://127.0.0.1:8000/health/ready
curl -X POST http://127.0.0.1:8000/v1/categories/match \
  -H 'Content-Type: application/json' \
  -d '{"taxonomy_key":"google","category":"Hoodies & Sweatshirts|Clothing"}'
```

If port 8000 is occupied, use `CATEGORY_MATCHER_PORT=8001 docker compose up --build`
and connect on port 8001. Update the plugin API URL to match. Inside a different
container, `127.0.0.1` refers to that container; connecting a containerized
WordPress installation needs a deliberate shared-network configuration.

## Storage and isolation

| Location | Storage | Purpose |
|---|---|---|
| `/app/src`, `scripts`, `config`, `data`, `tests`, `docker`, and selected root files | Read-only source mounts | Edit on the host; the container reads the current files |
| `/opt/venv` | Image | Locked Linux dependencies and editable package installation |
| `/app/models` | `models` volume | Downloaded weights and container-specific model metadata |
| `/app/artifacts` | `artifacts` volume | Immutable bundles, release descriptors, and active pointer |
| `/app/var` | `runtime` volume | Runtime SQLite database, WAL sidecars, and backups |
| `/app/reports` | `reports` volume | Generated diagnostics and retained holdout-access evidence |
| `/home/matcher/.cache` | `cache` volume | Download/tool caches |

Compose prefixes volume names with the project name. Host `models/`, `artifacts/`,
`var/`, and `.venv/` are excluded from the image and are not mounted. This avoids
mixing native macOS packages or absolute model paths with Linux bundles. Container
and native installations therefore have separate runtime history and releases.

`docker compose down` preserves these volumes. **Do not use `down --volumes` or
volume pruning to fix startup problems**: that removes persistent models,
runtime history, backups, and evidence. Back up valuable volumes before removal.
Use the existing runtime backup command for a consistent live SQLite snapshot;
copying only its live database file can omit WAL transactions.

```bash
docker compose exec api category-matcher runtime-backup --output var/backups/manual.sqlite3
mkdir -p var/backups
docker compose cp api:/app/var/backups/. ./var/backups/
```

## Edit and run commands

Edit files normally in the host IDE. Restart after API-only edits:

```bash
docker compose restart api
```

There is deliberately no automatic reload: core matching edits require rebuilding
artifacts, and restarting the HTTP process alone cannot update their identity.
To run developer tools when authorized, including without a running API:

```bash
docker compose run --rm api ruff check src tests scripts docker
docker compose run --rm api pytest
docker compose run --rm api bash
```

Commands supplied after `api` bypass startup preparation and execute directly;
running Ruff or opening a shell does not download models or rebuild artifacts.
Source mounts are read-only, so apply edits/formatting on the host. Save generated
outputs under `var/` or `reports/` and copy them out when needed. Development
dependencies are included; this image is not a minimal production distribution.

## Rebuild matching artifacts

For matching-code, taxonomy, alias, or ranking-profile changes, stop the API,
build/activate a new immutable release, then start it:

```bash
docker compose stop api
docker compose run --rm api category-matcher build --activate
docker compose up -d api
docker compose logs -f api
```

The first startup builds only when `active.json` is absent. Existing invalid or
incompatible releases fail validation instead of being silently replaced. A data
or ranking-profile edit requires the explicit build above even if startup still
validates the previous release. Inspect checksum errors before rebuilding; do not
edit manifests or release descriptors to silence them. See the full
[change/rebuild matrix](OPERATIONS.md#change-rebuild-and-activate).

For dependency changes, update `pyproject.toml` and `uv.lock` together, then rebuild
the image. Startup rejects a mounted dependency definition that differs from the
built environment. If manifest-pinned package versions change, rebuild artifacts
with the new image before starting the API:

```bash
docker compose stop api
docker compose build api
docker compose run --rm api category-matcher build --activate
docker compose up -d api
```

## Downloads, model changes, and shutdown

`CATEGORY_MATCHER_MODEL_DOWNLOAD=offline docker compose up` disables startup model
downloads; pinned snapshots must already exist in the Docker model volume. The
default `auto` permits downloading when model metadata is missing. It does not
repair damaged/missing weights behind existing metadata. To fetch the current
pins explicitly, with the API stopped:

```bash
docker compose run --rm api python scripts/prepare_models.py --download
```

For an intentional model-pin change, retain the old snapshots/releases, archive
the old model metadata, prepare the new pins, and rebuild as described in
[operations](OPERATIONS.md). Preparation refuses mismatched existing metadata.
Never copy native metadata into Docker or alter release hashes to fix paths.

Ctrl+C stops a foreground Compose service. Use `docker compose stop api` for a
detached service. One API owns inference; do not scale this Compose service to
multiple replicas sharing its volumes. Long native calls can outlast shutdown's
60-second grace period; retained artifacts are not rewritten during shutdown.

The container runs as UID/GID 10001 with dropped capabilities. It listens on
`0.0.0.0:8000` internally so Docker can forward traffic, while Compose publishes
only `127.0.0.1` on the host. The native launcher remains loopback-only. Other
containers on the same Docker network can reach its container port; attach only
trusted services. There is no authentication; this is a local developer setup.

Implementation references: [uv with Docker](https://docs.astral.sh/uv/guides/integration/docker/)
and [uv CPU-only PyTorch sources](https://docs.astral.sh/uv/guides/integration/pytorch/).
