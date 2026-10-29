# Operations and development workflow

## Requirements and first setup

For the container workflow, see [Docker development](DOCKER.md). The instructions
below describe the native installation; do not mix its model paths or bundles
with the container's isolated volumes.

Use Python 3.11 and uv from a POSIX environment; file leases depend on `fcntl`.
The current development installation is on macOS. Dependencies are pinned by
`uv.lock`; SQLite's vector extension is loaded through the `sqlite-vec` Python
package. A global SQLite extension installation alone does not replace that dependency.

```bash
uv sync --frozen
uv run python scripts/prepare_models.py --download
uv run category-matcher build --activate
uv run category-matcher-api
```

Model preparation writes `models/embedding_model.json` and
`models/nli_reranker_model.json` against downloaded snapshots. Omit `--download`
when the pinned snapshots are already cached. Existing metadata must agree with
the pins; the preparation script refuses incompatible metadata instead of overwriting it.

Build embeds the complete source taxonomy and creates SQLite/BM25 artifacts. It
can take substantially longer than startup and needs additional temporary resources.
Bundle manifests currently contain installation-specific absolute model paths;
on a different checkout/machine, prepare models and build locally.

## Service configuration

Flags override environment variables. Unknown `CATEGORY_MATCHER_API_*` names are
rejected, so a misspelled setting fails startup.

| Environment variable | Default | Limit or meaning |
|---|---|---|
| `CATEGORY_MATCHER_API_ARTIFACTS` | `artifacts/google` | Bundle/release root with `active.json` |
| `CATEGORY_MATCHER_API_RUNTIME_DB` | `var/category_matcher.sqlite3` | Writable state outside artifact storage |
| `CATEGORY_MATCHER_API_POLICY` | Unset | Optional assertion of the policy already pinned by activation; not an override |
| `CATEGORY_MATCHER_API_QUEUE_SIZE` | `8` | Integer 1–32 |
| `CATEGORY_MATCHER_API_TIMEOUT_SECONDS` | `30` | Greater than 0, up to 120 seconds |
| `CATEGORY_MATCHER_API_MAX_BODY_BYTES` | `16384` | Integer 1,024–65,536 |

Equivalent flags: `--artifacts`, `--runtime-db`, `--policy`, `--queue-size`,
`--timeout-seconds`, and `--max-body-bytes`. `--host` allows `127.0.0.1` or `::1`;
`--port` defaults to 8000 and accepts 1–65535. The launcher uses one worker.
For a locally changed port, update the calling plugin's configured API URL as well.

## Change, rebuild, and activate

| Change | Required action |
|---|---|
| Documentation or tests only | No runtime restart or artifact rebuild |
| API-only routes/schemas/settings/queue logic | Restart API; rebuild only if matching identity also changed |
| Taxonomy source, aliases, ranking profile, embedding metadata | Build and activate a compatible bundle, then restart API |
| Fingerprinted matching implementation or manifest-pinned package versions | Build and activate a compatible bundle, then restart API |
| Runtime schema | Append a migration; API initialization applies it |

The implementation fingerprint currently covers `domain.py`, `normalization.py`,
`taxonomy.py`, `storage.py`, `retrieval.py`, `encoder.py`, `matcher.py`, `query.py`,
`artifacts.py`, `decision.py`, and `reranker.py`. Even a formatting-only change to
one of those files changes its hash. Inspect `implementation_fingerprint()` when
changing build compatibility rules.

```bash
uv run category-matcher build
# Stop the running API using Ctrl+C.
uv run category-matcher activate --bundle '<artifact_version printed by build>'
uv run category-matcher-api
curl http://127.0.0.1:8000/health/ready
```

`build` stages without changing the selected release. `build --activate` combines
the operations and still requires stopping the API before activation. Never edit
`active.json`, built SQLite files, manifest checksums, release snapshots, or weights
in place to bypass an incompatibility check.

### Updated Google source

Replace the source with a valid Google-format text list or supply a separate path:

```bash
uv run category-matcher build --source path/to/updated-google-taxonomy.txt
```

The source hash creates a new taxonomy version. Review aliases against removed or
changed IDs; the builder validates referenced categories. Build, inspect, stop the
API, activate, and restart as above. Re-evaluate relevant examples before relying
on new mappings. Review packets bind to source taxonomy/data hashes and may need
regeneration after an update; keep earlier provenance rather than relabeling it.

### Alias changes

Maintain approved aliases explicitly in `data/google/aliases.jsonl`; approved
records require `reviewer` and `reviewed_at`. Only approved rows enter the built
lookup. Pending/rejected entries do not affect matching. Rebuild and activate after
edits. The removed `feedback-export` and `aliases-apply` commands are unavailable.

## Inspection, rollback, and backup

```bash
uv run category-matcher inspect
uv run category-matcher verify
# Stop API before switching to a retained compatible release:
uv run category-matcher rollback
uv run category-matcher-api
```

Rollback uses the previous release, or an explicit `--release-id`. It requires
retained files and compatibility with the current code/dependencies; it does not
undo a source checkout. Match-event history remains in the separate runtime DB.

```bash
uv run category-matcher runtime-migrate
uv run category-matcher runtime-backup --output var/backups/pilot.sqlite3
uv run category-matcher runtime-restore --backup var/backups/pilot.sqlite3 \
  --output var/restored.sqlite3
# Stop API before starting against the restored copy:
uv run category-matcher-api --runtime-db var/restored.sqlite3
```

Backups include a `.sha256` sidecar. Restore checks it and writes a new DB; existing
outputs are not overwritten. Historical migration SQL/checksums must remain unchanged.
The retired feedback table has no active handlers and remains as legacy state.

## Evaluation and qualification

```bash
uv run category-matcher check-data
uv run category-matcher evaluate --allow-draft
uv run category-matcher benchmark --output reports/resource_report.json
uv run python scripts/run_user_examples.py --output reports/user_examples.json
uv run python scripts/benchmark_api.py \
  --reference reports/retrieval_comparison.json --output-dir reports/api
```

`evaluate` defaults to development data and writes `reports/retrieval_comparison.json`.
The HTTP benchmark checks that reference against the active artifact/data and starts
an isolated loopback server/runtime DB. Use a fresh output location to retain
comparison evidence. Benchmarks load real models and measure development behavior,
not independently validated accuracy or production capacity.

```bash
uv run category-matcher review-export --dataset path/to/dataset.jsonl \
  --output path/to/review-packet.jsonl
uv run category-matcher review-apply --dataset path/to/dataset.jsonl \
  --reviews path/to/edited-review-packet.jsonl --output path/to/reviewed-dataset.jsonl
uv run category-matcher pilot-export --output var/pilot.unlabeled.jsonl
```

See the [review instructions](../data/google/reviews/README.md) for explicit human
fields and `capture-inputs`. `pilot-export` deduplicates recorded requests into
unlabeled development inputs; it does not produce an independent holdout or use
feedback. Dataset review does not automatically create aliases or change the server.

`calibrate` uses development/calibration data to fit a policy. `qualify` uses a
separate reviewed, independently collected holdout and checks minimum sizes,
precision, coverage, retrieval recall, and non-exact evidence. Use each command's
`--help` for dataset paths. `--allow-draft` enables diagnostics only;
`--experimental-policy` is a CLI preview that cannot authorize auto-acceptance.
A qualified policy is selected through `activate --bundle ... --policy ...`.

The persistent holdout access ledger currently defaults to
`reports/sprint_2/holdout_access.jsonl`. Preserve it, policy files, dataset provenance,
and relevant evidence when cleaning diagnostics; deleting it loses the reuse audit.
It is Git-ignored along with other reports, so include it in operational backups.

## Verification during development

Run checks appropriate to the change when authorized:

```bash
uv run pytest
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
```

Tests use synthetic bundles and injected model behavior where appropriate; real
artifact verification and model/HTTP benchmarks cover different concerns. Neither
synthetic regression tests nor AI-draft evaluation establish field accuracy.
Document what actually ran and any unverified behavior in the change description.

## Troubleshooting

| Symptom | Diagnosis and action |
|---|---|
| Address already in use on port 8000 | Find the owning process with `lsof -nP -iTCP:8000 -sTCP:LISTEN`; stop the original API with Ctrl+C or select another port |
| API paused after Ctrl+Z | Resume with `fg` in its original shell, then stop with Ctrl+C; pausing retains the socket and lease |
| Bundle is in use | A live/paused API holds the shared lease; stop it before activation/rollback |
| Matching implementation or package changed | Use locked dependencies and rebuild; do not rewrite the manifest hash |
| Readiness stays 503 | Inspect `startup_failed` logs for missing snapshots, artifact/checksum failures, or policy incompatibility |
| Inference returns 503 and service becomes failed | Inspect `inference_failed` logs, correct the cause, rebuild if needed, then restart |
| Wrong category from a vague store title | Reproduce the actual hierarchy/description; inspect `query.py`, `matcher.py`, and relevant regression tests before adding a global alias |
| High latency or queue_full | Distinguish model time from queue wait; measure the current profile before changing concurrency |
| Model/tokenizer warnings | Determine readiness and actual exception state first; a warning alone is not a startup failure |

Structured logs contain event IDs, timing, stage, status, and versions. The runtime
DB contains original input/context and responses, so include it deliberately when
sharing diagnostic material. Do not delete active bundles, model snapshots, DB
sidecars, migration history, or qualification evidence as routine cache cleanup.
