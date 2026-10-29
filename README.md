# Category matcher

Local category matching over the Google product taxonomy. Exact lookup and reviewed
aliases run before BM25, fuzzy matching, BGE embeddings, and NLI reranking. Taxonomy
records and vectors live in SQLite with `sqlite-vec`; inference uses local CPU models.

## Setup

Use Python 3.11 and uv. Run from the project root:

```bash
uv sync --frozen
uv run python scripts/prepare_models.py --download
uv run category-matcher build --activate
uv run category-matcher-api
```

Model revisions are pinned in [config/model_pins.json](config/model_pins.json).
Omit `--download` to prepare models from the local cache. The ranking configuration
is [config/category_ranking.json](config/category_ranking.json).

The API listens at `http://127.0.0.1:8000`. Interactive documentation is at `/docs`.
Wait for `/health/ready` to return 200 before matching. Use **Ctrl+C** to stop the
server; Ctrl+Z pauses it and leaves its port and artifact lock occupied.

## Match a category

```bash
uv run category-matcher match 'Hoodies & Sweatshirts|Clothing'
uv run category-matcher match 'Clothing'
uv run category-matcher match --id 212
```

For store hierarchies, send the actual category name separately from its ancestors,
ordered from root to immediate parent:

```bash
curl -X POST http://127.0.0.1:8000/v1/categories/match \
  -H 'Content-Type: application/json' \
  -d '{"taxonomy_key":"google","category":"Collections","source_ancestors":["Clothing","Men","Tops","Hoodies & Sweatshirts|Clothing"]}'
```

`source_ancestors` are store context, not Google parent constraints. Use
`parent_category_id` only for a known Google category ID. `context` accepts a product
description; `category_id` performs a direct canonical lookup. IDs are strings.
The API accepts English inputs, up to 300 characters for the category and 1,000
characters for context. See `/docs` for all request fields and validation limits.

Only `status: "matched"` assigns `category`. A `provisional` response may include
`provisional_category`, the best ranked suggestion, with `requires_review: true`.
It is not an accepted mapping. Unclear inputs keep `category: null`; scores are
ranking signals, not correctness probabilities.

Generic store labels such as `Collections` can inherit a grounded product ancestor
provisionally. Conflicting ancestry requires clarification. Default labels such as
`Uncategorized`, `Miscellaneous`, and `N/A` have no candidate without explicit
product context.

## API operation

The service keeps resident models and a SQLite connection on one inference thread.
It processes one match at a time, with eight pending jobs by default. Excess work
receives 503 `queue_full`; the default HTTP deadline is 30 seconds. Health and
canonical lookups remain available during inference. Successful matches
are persisted in the separate runtime database.

Useful endpoints:

- `GET /health/live` and `GET /health/ready`
- `GET /v1/taxonomies`
- `GET /v1/taxonomies/google/categories/{id}`
- `POST /v1/categories/match`

Launcher options include `--port`, `--artifacts`, `--runtime-db`, `--queue-size`,
`--timeout-seconds`, and `--max-body-bytes`. The host is restricted to loopback.
Run `uv run category-matcher-api --help` for the complete options. Configuration
also accepts corresponding `CATEGORY_MATCHER_API_*` environment variables; consult
[settings.py](src/category_matcher/api/settings.py) for names and validation.

The WooCommerce mapper calls the API from PHP. Its default URL is
`http://127.0.0.1:8000`; trusted PHP configuration can override
`WPFM_CATEGORY_MATCHER_API_URL`. That address must be reachable from WordPress.
Provisional selections display **Provisioning needed** until reviewed.

## Files and updates

| Path | Purpose |
|---|---|
| `src/category_matcher/` | Matching, API, build, and evaluation code |
| `google_category_list.txt` | Source Google taxonomy |
| `config/` | Pinned model and ranking configuration |
| `data/google/aliases.jsonl` | Reviewed and pending aliases |
| `data/google/evaluation/` | Evaluation inputs and draft labels |
| `data/google/reviews/` | Human review packets and retained label audit evidence |
| `models/` | Downloaded weights and generated model metadata; ignored by Git |
| `artifacts/google/` | Versioned taxonomy/vector databases and release pointers; ignored by Git |
| `var/category_matcher.sqlite3` | Mutable match-event database; ignored by Git |
| `reports/` | Generated diagnostics; ignored by Git |
| `tests/` | Regression coverage |
| `scripts/` | Model preparation, API client, and reusable measurement tools |

Changes to taxonomy data, aliases, matching implementation, pinned dependencies,
or ranking configuration require a compatible rebuilt bundle. Do not edit a built
bundle or its manifest manually. To update:

```bash
uv run category-matcher build
# Stop the API with Ctrl+C, then activate the version printed by build.
uv run category-matcher activate --bundle '<artifact_version>'
uv run category-matcher-api
```

Build stages without activation unless `--activate` is supplied. Activation and
rollback require the API to be stopped. They retain immutable release metadata and
validate bundle/model checksums. Rollback also requires code and dependency
compatibility with the retained release:

```bash
uv run category-matcher rollback
uv run category-matcher inspect
```

Keep the installed environment, model weights, active/recovery bundles, and runtime
databases. Runtime database backups include a `.sha256` sidecar:

```bash
uv run category-matcher runtime-backup --output var/backups/pilot.sqlite3
uv run category-matcher runtime-restore --backup var/backups/pilot.sqlite3 \
  --output var/restored.sqlite3
```

Restore writes a new database. Stop the API before selecting restored state with
`--runtime-db var/restored.sqlite3`.

## Label review

Human label review is still pending. The 600 development/calibration/test proposals
are AI-authored and do not establish real-world accuracy. Retained audit notes are
historical proposals, not approvals or measurements of the current release.
See the [review workflow](data/google/reviews/README.md) and
[dataset notes](data/google/evaluation/README.md).

## Development tools

These are optional commands to run when needed:

```bash
uv run category-matcher verify
uv run category-matcher check-data
uv run category-matcher evaluate --allow-draft
uv run category-matcher benchmark --output reports/resource_report.json
uv run pytest
uv run ruff check src tests scripts
```

Evaluation writes `reports/retrieval_comparison.json`. The HTTP benchmark requires
a fresh CLI evaluation report for the same artifact and development dataset:

```bash
uv run python scripts/benchmark_api.py \
  --reference reports/retrieval_comparison.json --output-dir reports/api
```

The benchmark starts an isolated loopback server and runtime database. It measures
API/CLI agreement and timing, not independently validated matching accuracy.
`scripts/run_user_examples.py` compares the original five user examples.
`scripts/smoke_quantization.py` and `scripts/experiment_quantized_reranker.py`
retain optional CPU profile measurements without changing the active bundle.

`calibrate` fits a policy; `qualify` requires reviewed calibration and an independent
reviewed holdout. Draft policies only preview decisions. The persistent holdout
access ledger remains at `reports/sprint_2/holdout_access.jsonl` for compatibility;
if qualification creates it, preserve it with policy evidence and backups.
The current service has no qualified semantic auto-acceptance policy.

The original implementation and sprint plans remain as historical design references.
Generated sprint reports and one-off experiment scripts are no longer kept in the
working project; earlier versions remain available in Git history.
