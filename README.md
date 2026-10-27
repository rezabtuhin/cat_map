# Category matcher

Local Google product-category matching using exact lookup, reviewed aliases,
BM25, fuzzy matching, and BGE semantic retrieval stored in SQLite with `sqlite-vec`.
The default hybrid pipeline adds category entailment rank fusion with a pinned
local NLI model. Canonical lookups avoid model inference.

Current [Sprint 2 status](reports/sprint_2/STATUS.md) and
[latest findings](reports/sprint_2/revision_3/FINDINGS.md) include store-label
abstention, ranking safeguards, review tooling, and native memory measurements. These reused
draft inputs do not establish independently validated accuracy.

The [CPU quantization comparison](reports/sprint_2/quantization/FINDINGS.md) retains
float32 as the default: full int8 loses development rankings, while embedding-only
int8 preserves the observed top-1 results but raises measured process memory.

**Non-exact results are provisional candidates.** Similarity and reranker scores
are ranking signals, not correctness probabilities. Sprint 2 implements calibration,
clarification/no-match policy rules, and qualification checks. The fitted draft
policy is not validated and can only preview decisions. A local development API
is implemented in [Sprint 3](reports/sprint_3/FINDINGS.md). Production qualification
remains pending.

[Sprint 4](reports/sprint_4/STATUS.md) adds persistent feedback, reviewed alias
promotion, staged builds, and recovery commands.

**Uncertain inputs stay unmapped.** Only `status: "matched"` may have an assigned
`category`. Every other result keeps `category: null`. Do not assign an ID from
the first `candidates` entry or from `decision_preview`. A broad suggestion is
not an automatic fallback when the input does not support a reliable category.

See [the sprint plan](CATEGORY_MATCHING_SPRINT_PLAN.md) and
[implementation plan](CATEGORY_MATCHING_IMPLEMENTATION_PLAN.md).

## Setup

Run these commands from the project root. Python 3.11 and uv are required.

```bash
uv sync --frozen
uv run python scripts/prepare_models.py --download
uv run category-matcher build --activate
uv run category-matcher verify
```

Model preparation uses [checked-in revisions](config/model_pins.json); omit
`--download` to use the local cache only. The download command requires network access. Build and inference
use the downloaded model locally. Model weights are cached under `models/`;
generated taxonomy/vector bundles live under `artifacts/google/`. Both are ignored
by Git. A source, alias, dependency, or matching-code change requires rebuilding;
the manifest pins revisions, configuration, file checksums, and implementation hashes.
The build reads [config/category_ranking.json](config/category_ranking.json) and
pins the selected ranking profile into the artifact. Use `build --retrieval-only`
to build a bundle without the ranking model.

## Try a category

```bash
uv run category-matcher match 'Clothing'
uv run category-matcher match 'Tops'
uv run category-matcher match 'Smart Phone'
uv run category-matcher match 'Hoodies & Sweatshirts|Clothing'
uv run category-matcher match 'Eco Friendly|Clothing'
uv run category-matcher match 'protective covers for cellphones' --method semantic
uv run category-matcher match 'mobil phones' --parent 267
uv run category-matcher match --id 2353
uv run category-matcher match 'Jackets'
uv run category-matcher match 'Fitness Equipment'
uv run category-matcher match 'Collections'
```

JSON output includes the canonical ID/path, result status, branch evidence, and
artifact version. `--method` can be `exact`, `fuzzy`, `bm25`, `semantic`, or `hybrid`.
`--limit` controls how many candidates are displayed, not retrieval depth.
Sprint 2 uses label-only BM25 with retrieval-only plural folding, plain BGE query
encoding, and weighted rank fusion. It retrieves up to 100 lexical, 100 semantic,
and 50 fuzzy candidates, retaining up to 200 plus five parents. Those choices
improved the draft development comparison; they do not establish holdout accuracy.
The active artifact records the encoding settings; downloaded model metadata alone
does not determine the runtime query instruction.

`label|parent` is interpreted as a category plus context. When the context uniquely
identifies a canonical category, retrieval stays inside its hierarchy. Generic
apparel inputs cannot introduce a baby/toddler clothing qualifier. Explicit baby
terms remain eligible. This narrow rule addresses the reported failure; it does
not claim to solve every audience qualifier. Attribute-only inputs such as
`Eco Friendly|Clothing` retain the broad context as a provisional candidate.

Unique whole-label spelling corrections receive ranking priority and remain
provisional. Short or ambiguous spellings expose alternatives instead of silently
becoming aliases. Literal general-category wrappers such as `Furniture in general`
can resolve a canonical label; meaningful qualifiers are preserved.

Whole alternatives in coordinated labels provide provisional group evidence:
`Jackets` ranks Coats & Jackets, and `Fitness Equipment` ranks Exercise & Fitness
with Fitness & Nutrition retained as another broad interpretation. Neither is
an approved alias. Store-section titles such as `Collections`, `Erin Recommends`,
and `New Luma Yoga Collection` remain unmapped when they do not identify products.
The matcher can use a supplied product description as context.

Without a qualified policy, canonical lookups, approved aliases, and conservative
literal general-category lookups return `matched`.
The `Tops` alias was
approved from the user's explicit confirmation of ID `212`. Other draft aliases
are inactive until reviewed. Rebuild after changing aliases.

## Run the local API

After setup, start the service from the project root:

```bash
uv run category-matcher-api
```

The server listens at `http://127.0.0.1:8000` with one worker. Interactive API docs
are at `/docs`; the schema is at `/openapi.json`. Startup validates the selected
bundle and loads and warms the models on the inference thread. `/health/live`
responds during initialization; `/health/ready` returns 503 until initialization
succeeds, and remains 503 if artifacts, models, or a configured policy fail
validation. Inspect the structured startup log for the cause.

```bash
curl http://127.0.0.1:8000/health/ready
curl http://127.0.0.1:8000/v1/taxonomies
curl http://127.0.0.1:8000/v1/taxonomies/google/categories/1604
curl -X POST http://127.0.0.1:8000/v1/categories/match \
  -H 'Content-Type: application/json' \
  -d '{"taxonomy_key":"google","category":"Clothing"}'
uv run python scripts/api_client.py 'Hoodies & Sweatshirts|Clothing'
```

`POST /v1/categories/match` accepts the following JSON fields. Unknown fields and
incorrect types are rejected with 422; category IDs are strings.

| Field | Contract |
|---|---|
| `taxonomy_key` | Required; currently only `"google"` |
| `taxonomy_version` | Optional; if supplied, must equal the active version listed by `/v1/taxonomies` |
| `category` | Category text, at most 300 characters; required unless `category_id` is supplied |
| `category_id` | Optional canonical category ID for a direct lookup |
| `parent_category_id` | Optional canonical parent ID restricting the matching hierarchy |
| `context` | Optional product description, at most 1,000 characters |
| `language` | Defaults to `"en"`; unsupported languages return `unsupported_language` |
| `limit` | Number of displayed alternatives, integer 1–10; defaults to 3 |

Responses include `status`, canonical `category` or `null`, `alternatives`, method,
reason, clarification, taxonomy/artifact versions, `event_id`, and queue/inference/
total timings in milliseconds. The `X-Request-ID` response header matches the
event ID in the response and structured logs. Only `status: "matched"` assigns a
category. Treat `alternatives` as suggestions requiring review. Inputs such as
`Collections` and `Erin Recommends` remain unmapped. The API uses the same hybrid
pipeline as the default CLI and does not expose experimental decision previews.

The service keeps one matcher, its SQLite connection, and its resident models on
one dedicated inference thread. Canonical GET lookups use an immutable category
snapshot. One match request runs at a time, with eight pending requests by default;
further requests receive 503 `queue_full` with `Retry-After: 1`. Health and canonical
lookup endpoints remain available while inference runs. Feedback uses the same
bounded queue. Match requests and their responses are stored in the separate
runtime database before a successful response is returned.

The default 30-second HTTP deadline includes reading the body, queueing, and
inference. An expired request receives 504; an expired queued job or one whose
application task is cancelled is removed. A disconnected HTTP client does not
necessarily cancel the application task; the deadline still bounds its wait.
A native model call already running finishes before another job starts,
even if its caller times out or its application task is cancelled. Shutdown waits for active inference and
closes SQLite on its owner thread. A native call that hangs requires process
recovery; the HTTP deadline cannot interrupt native model execution.

| Environment variable | Default | Allowed configuration |
|---|---|---|
| `CATEGORY_MATCHER_API_ARTIFACTS` | `artifacts/google` | Active-bundle root containing `active.json` |
| `CATEGORY_MATCHER_API_POLICY` | Unset | Optional assertion of the policy already pinned by activation; overrides are rejected |
| `CATEGORY_MATCHER_API_RUNTIME_DB` | `var/category_matcher.sqlite3` | Writable runtime state outside artifact storage |
| `CATEGORY_MATCHER_API_QUEUE_SIZE` | `8` | Integer 1–32 |
| `CATEGORY_MATCHER_API_TIMEOUT_SECONDS` | `30` | Greater than 0, at most 120 seconds |
| `CATEGORY_MATCHER_API_MAX_BODY_BYTES` | `16384` | Integer 1,024–65,536 bytes |

Equivalent launcher flags are `--artifacts`, `--policy`, `--runtime-db`, `--queue-size`,
`--timeout-seconds`, and `--max-body-bytes`. Flags override environment values.
`--port` selects another port; `--host` permits only loopback addresses. Unknown
`CATEGORY_MATCHER_API_*` settings fail startup. Oversized bodies return 413 and
non-JSON match requests return 415. Error responses contain `error.code`,
`error.message`, and `event_id`; requests made before readiness return 503.

The default service has no qualified semantic acceptance policy. It preserves the
canonical/approved-alias acceptance behavior described above and leaves other
results provisional or unmapped. This is a local development integration; trusted
label review, independent accuracy qualification, and the outstanding resource
gate remain separate work.

Reproduce the real-model HTTP checks with a new report directory:

```bash
uv run python scripts/benchmark_api.py --requests 200 \
  --output-dir reports/sprint_3/repeat_1
```

The benchmark starts and stops its own loopback server, compares all 200 frozen
development CLI outcomes, runs the ten supplied user examples, measures 200 warm
model-path requests, and exercises overload while probing health. It does not
score the test holdout or establish matching accuracy.

## Feedback and recovery

Restart the API to load Sprint 4. Match inputs, contexts, responses, and pending
feedback persist locally in `var/category_matcher.sqlite3`. Runtime migrations
are automatic and checksum-verified; taxonomy bundles remain read-only at schema
version 1. Runtime schema version 2 supports an upgrade from version 1.

Use the `event_id` from a match response to submit feedback:

```bash
curl -X POST http://127.0.0.1:8000/v1/categories/feedback \
  -H 'Content-Type: application/json' \
  -d '{"match_event_id":"<match event_id>","action":"confirm","reviewer":"Your name"}'
```

`confirm` accepts the assigned category, or requires `category_id` to select a
displayed alternative. `correct` requires a canonical `category_id`; `reject` and
`needs_clarification` omit it. Feedback stays **pending** and does not change
matching. References to older artifact generations retain the original version;
the feedback API requires that original taxonomy version to be active.

```bash
uv run category-matcher feedback-export --output var/feedback.pending.jsonl
# Edit eligible decisions with explicit human reviewer, timestamp, and reason.
uv run category-matcher aliases-apply --reviews var/feedback.pending.jsonl \
  --output data/google/aliases.reviewed.jsonl
uv run category-matcher build --aliases data/google/aliases.reviewed.jsonl
# Stop the API, then activate the version printed by build and restart it.
uv run category-matcher activate --bundle '<artifact_version>'
uv run category-matcher-api
```

Contextual, merchandising, and attribute-only feedback cannot become global
aliases. `build` stages and verifies without activation; `build --activate`
combines both operations while the API is stopped. Activation pins the bundle,
model-file checksums, and optional qualified `--policy` in an immutable release.
Changing weights, policies, or bundle files after activation fails readiness.
Activation/rollback is blocked while the updated API holds resident models.
Existing bundles need one `activate --bundle '<artifact_version>'` before first
serving under Sprint 4; the current local bundle has already been activated.

```bash
uv run category-matcher runtime-migrate
uv run category-matcher runtime-backup --output var/backups/pilot.sqlite3
uv run category-matcher runtime-restore --backup var/backups/pilot.sqlite3 \
  --output var/restored.sqlite3
# Stop the API before switching to restored state:
uv run category-matcher-api --runtime-db var/restored.sqlite3
# With the API stopped, restore the previous retained release:
uv run category-matcher rollback
```

Backups include a `.sha256` sidecar. Restore verifies it and writes a new database;
existing files are never overwritten. Rollback requires a retained release
compatible with the current code and lockfile; it preserves runtime feedback.
The initial activation has no previous release to roll back to.

`pilot-export --output var/pilot.unlabeled.jsonl` exports deduplicated inputs for
review. These already-inspected inputs are development data, not an independent
qualification holdout. To rerun isolated live/recovery checks, use
`uv run python scripts/check_sprint4.py --output /tmp/sprint4-checks.json`.

## Ranking configuration

```bash
uv run category-matcher match 'Smart Phone' --rerank
uv run category-matcher match 'Hoodies & Sweatshirts|Clothing' --no-rerank
uv run python scripts/run_user_examples.py \
  --output reports/sprint_2/revision_3/user_examples.json
```

Hybrid matching uses the artifact's pinned NLI rank-fusion profile by default;
`--no-rerank` selects retrieval only. Other retrieval methods do not use it.
The first 100 candidates receive model scores, combined with retrieval ranks;
all remaining candidates stay available. Reranked non-exact outputs remain
provisional. The comparison script does not change the decision policy.
The current pair wording is `The product category is …`; the configuration pins
both query and candidate templates in the artifact.
The observed Apple Silicon CPU runtime requires copying misaligned checkpoint
parameters into aligned storage; the adapter applies and tests that workaround.
The older MiniLM and BGE reranker experiments remain available through explicit
`--reranker-model` overrides; neither is the selected profile. When the model
supports only the supplied parent and rejects every scored product type, the
matcher preserves retrieval order and flags the specificity conflict for
clarification. It retains original model ranks and scores. This does not prove
the retrieval answer is correct.
The versioned weak-promotion guard also preserves the retrieval winner when its
semantic support is substantially higher and the model's promoted candidate has
weak entailment support. Both disagreement flags require clarification under a
decision policy.

## Evaluate and verify

```bash
uv run category-matcher evaluate --allow-draft --rerank \
  --output reports/sprint_2/revision_3/retrieval_comparison.json
uv run category-matcher evaluate --allow-draft \
  --dataset data/google/evaluation/user_examples.jsonl \
  --output reports/sprint_2/user_retrieval_evaluation.json
uv run category-matcher inspect
uv run category-matcher verify
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The 600 development/calibration/test proposals are AI-authored and **await human review**. The
evaluator refuses them unless `--allow-draft` is specified; resulting reports are
marked diagnostic only. They do not establish real-world accuracy. The five
user-provided inputs are stored separately, with confirmed and draft labels
distinguished. Draft calibration has been run; test proposals have not been scored.
An independently collected, reviewed holdout is still required.

Baseline reports separate exact and non-exact examples and include top-1, top-3,
common-budget recall at 20, final-shortlist recall, slice counts, and per-example
errors. Reported retrieval-only latency excludes batched query encoding and must
not be presented as end-to-end response time. Zero semantic auto-acceptance in
the default matcher is intentional; exact-only precision is not model accuracy.

`verify` checks canonical labels/paths/IDs across the real taxonomy and compares
SQLite cosine scores and scoped rankings against independent NumPy calculations.

## Calibrate and preview a decision policy

```bash
uv run category-matcher check-data
uv run category-matcher calibrate --allow-draft \
  --output reports/sprint_2/revision_3/draft_policy.json
uv run category-matcher match 'Smart Phone' \
  --experimental-policy reports/sprint_2/revision_3/draft_policy.json
uv run category-matcher benchmark --requests 200 --idle-baseline-seconds 5 \
  --output reports/sprint_2/revision_3/resource_report.json
```

Draft-policy previews appear in `decision_preview`; they cannot turn a provisional
candidate into an automatic match. Loading that file with `--policy` fails closed.
The policy is bound to the artifact version and ranking configuration, and includes
its calibration-data checksum, review counts, and metrics.

Once reviewed calibration and independent test inputs are available, run:

```bash
uv run category-matcher calibrate \
  --development data/google/evaluation/development.jsonl \
  --calibration path/to/reviewed_calibration.jsonl --output path/to/fitted_policy.json
uv run category-matcher qualify --policy path/to/fitted_policy.json \
  --development data/google/evaluation/development.jsonl \
  --calibration path/to/reviewed_calibration.jsonl \
  --test path/to/independent_test.jsonl --output path/to/qualified_policy.json
```

These paths are placeholders for reviewed files. Qualification requires separate
families/targets, at least 200 calibration and 200 test inputs, declared independent
holdout provenance, and passing accuracy/coverage/non-exact evidence checks. It
records prior holdout access and rejects reuse under a changed policy. Development
labels also need review. A failed gate never enables automatic semantic acceptance.

`benchmark` measures 200 serial warm requests including query encoding, retrieval,
and the selected ranking model, with model warmup reported separately. It samples
actual model-path inputs rather than canonical/general-category shortcuts. It
records idle, model-load, and inference swap observations separately; these remain
system-wide counters. It also samples this process's resident/compressed memory
and physical footprint on macOS. See [the current resource report](reports/sprint_2/revision_3/resource_report.json).
The benchmark does not establish production concurrency or matching accuracy.

## Collect and review inputs

Ready-to-edit packets cover the 200 development proposals, 200 calibration
proposals, and five new real store inputs. They contain pending decisions only.
See [the review instructions](data/google/reviews/README.md).

```bash
uv run category-matcher review-export --dataset path/to/dataset.jsonl \
  --output path/to/review-packet.jsonl
uv run category-matcher review-apply --dataset path/to/dataset.jsonl \
  --reviews path/to/edited-review-packet.jsonl --output path/to/reviewed-dataset.jsonl
uv run category-matcher capture-inputs --inputs path/to/real-inputs.jsonl \
  --source-name '<collection source>' --output path/to/collected.unlabeled.jsonl
```

These commands need no model weights or active artifact. Capture creates
unlabeled inputs; apply requires explicit human-review fields and validates the
unchanged source inputs, taxonomy IDs, and parent scope. It writes a new file and
preserves source/split provenance. It cannot create independent test evidence by
renaming AI fixtures or inputs already inspected during development.

## Current scope

- Google English taxonomy only; 5,582 categories and two vectors per category.
- CLI and local FastAPI development service, with local CPU inference.
- API inference is bounded to one active job with a configurable pending queue.
- Small modules separate domain records, parsing, storage, retrieval, and orchestration.
- No remote model calls during inference, frontend, or deployed site.
- Additional vendors and broader scaling work remain deferred until the matching
  approach has been validated, as requested.
