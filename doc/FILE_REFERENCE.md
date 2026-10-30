# File reference

Each maintained project file has one purpose line below. Generated model weights,
installed dependencies, caches, and per-run reports are described by role instead
of enumerating their installation-specific internal files. Paths are repository-relative.

## Root files

| File | Purpose |
|---|---|
| [.gitignore](../.gitignore) | Excludes generated environments, caches, models, bundles, runtime state, and reports from Git. |
| [.python-version](../.python-version) | Selects the Python version expected by the uv project. |
| [README.md](../README.md) | Provides the quick start, folder tree, and links into the developer guide. |
| [AGENTS.md](../AGENTS.md) | Provides Codex-native instructions and shared agent invariants, verification rules, and data-handling expectations. |
| [CLAUDE.md](../CLAUDE.md) | Imports AGENTS.md as the shared project instructions for Claude Code. |
| [GEMINI.md](../GEMINI.md) | Imports AGENTS.md as the shared project instructions for Gemini CLI. |
| [WORKFLOW.md](../WORKFLOW.md) | Defines the contributor/agent change lifecycle, checks, release impact, and handoff steps. |
| [pyproject.toml](../pyproject.toml) | Declares package metadata, dependencies, command entry points, and tooling configuration. |
| [uv.lock](../uv.lock) | Locks dependency resolution for reproducible installation with uv. |
| [google_category_list.txt](../google_category_list.txt) | Supplies canonical Google category IDs and paths to the taxonomy builder. |
| [CATEGORY_MATCHING_IMPLEMENTATION_PLAN.md](../CATEGORY_MATCHING_IMPLEMENTATION_PLAN.md) | Preserves the original architecture proposal as historical context rather than the current contract. |
| [CATEGORY_MATCHING_SPRINT_PLAN.md](../CATEGORY_MATCHING_SPRINT_PLAN.md) | Preserves historical sprint scope and sequencing, including work that was later removed. |

## Configuration

| File | Purpose |
|---|---|
| [config/model_pins.json](../config/model_pins.json) | Pins immutable embedding/ranking model revisions and download metadata. |
| [config/onnx_model_pins.json](../config/onnx_model_pins.json) | Pins the proposed multilingual INT8 repositories and exact file sizes/SHA-256 checksums. |
| [config/category_ranking.json](../config/category_ranking.json) | Selects the NLI ranking model, pair templates, candidate budget, and rank-fusion settings. |

## Matching and supporting modules

| File | Purpose |
|---|---|
| [src/category_matcher/__init__.py](../src/category_matcher/__init__.py) | Marks the category-matcher package and describes its provisional-result scope. |
| [src/category_matcher/__main__.py](../src/category_matcher/__main__.py) | Runs the CLI when invoked with python -m category_matcher. |
| [src/category_matcher/artifacts.py](../src/category_matcher/artifacts.py) | Builds taxonomy/vector/BM25 bundles and checks source, implementation, dependency, and file integrity. |
| [src/category_matcher/benchmark.py](../src/category_matcher/benchmark.py) | Measures serial end-to-end inference timing, process memory, swap observations, and resource gates. |
| [src/category_matcher/bundles.py](../src/category_matcher/bundles.py) | Stages bundles and manages checked, immutable releases with atomic activation, rollback, and file leases. |
| [src/category_matcher/calibration.py](../src/category_matcher/calibration.py) | Fits draft decision thresholds and qualifies policies against reviewed independent holdout evidence. |
| [src/category_matcher/cli.py](../src/category_matcher/cli.py) | Defines build, match, evaluation, review, release, and runtime-maintenance commands. |
| [src/category_matcher/datasets.py](../src/category_matcher/datasets.py) | Loads JSONL datasets and checks review provenance, split overlap, and leakage constraints. |
| [src/category_matcher/decision.py](../src/category_matcher/decision.py) | Validates version-bound policies and applies acceptance, abstention, or draft-preview rules. |
| [src/category_matcher/domain.py](../src/category_matcher/domain.py) | Defines framework-independent category, hit, match-result, and reranker-interface records. |
| [src/category_matcher/encoder.py](../src/category_matcher/encoder.py) | Loads pinned local BGE snapshots and encodes category documents and matching queries. |
| [src/category_matcher/evaluation.py](../src/category_matcher/evaluation.py) | Compares retrieval methods on labeled inputs while preserving provenance and per-example evidence. |
| [src/category_matcher/matcher.py](../src/category_matcher/matcher.py) | Orchestrates safe lookups, context inheritance, retrieval, reranking, safeguards, and decision policy. |
| [src/category_matcher/memory.py](../src/category_matcher/memory.py) | Collects native process memory observations with platform-aware accounting. |
| [src/category_matcher/migrations.py](../src/category_matcher/migrations.py) | Applies checksum-verified runtime schema migrations while retaining legacy database compatibility. |
| [src/category_matcher/normalization.py](../src/category_matcher/normalization.py) | Separates conservative exact-match keys from broader retrieval token normalization. |
| [src/category_matcher/quantization.py](../src/category_matcher/quantization.py) | Applies explicit float32 or QNNPACK int8 CPU profiles and accounts for model tensors. |
| [src/category_matcher/query.py](../src/category_matcher/query.py) | Parses inline context and source ancestry and defines generic-label, audience, broad-category, and promotion safeguards. |
| [src/category_matcher/reranker.py](../src/category_matcher/reranker.py) | Loads local cross-encoders, validates ranking profiles, aligns CPU weights, and fuses NLI evidence. |
| [src/category_matcher/retrieval.py](../src/category_matcher/retrieval.py) | Implements BM25, fuzzy/spelling retrieval, weighted reciprocal rank fusion, and parent expansion. |
| [src/category_matcher/review.py](../src/category_matcher/review.py) | Captures unlabeled inputs, exports recorded pilot requests, and applies source-bound human dataset reviews. |
| [src/category_matcher/runtime.py](../src/category_matcher/runtime.py) | Persists match events using short-lived SQLite connections and supports verified backup/restore. |
| [src/category_matcher/storage.py](../src/category_matcher/storage.py) | Owns taxonomy SQLite schema, hierarchy scope queries, embedding storage, and sqlite-vec similarity search. |
| [src/category_matcher/taxonomy.py](../src/category_matcher/taxonomy.py) | Parses and audits Google-format taxonomy text into canonical category and hierarchy records. |
| [src/category_matcher/verification.py](../src/category_matcher/verification.py) | Checks canonical lookups and vector similarity/scoping against independent numerical calculations. |

## HTTP application

| File | Purpose |
|---|---|
| [src/category_matcher/api/__init__.py](../src/category_matcher/api/__init__.py) | Exports the create_app application factory. |
| [src/category_matcher/api/app.py](../src/category_matcher/api/app.py) | Composes routes, request/body boundaries, error handlers, health, and canonical lookup endpoints. |
| [src/category_matcher/api/errors.py](../src/category_matcher/api/errors.py) | Defines service exceptions and stable JSON error envelopes with request IDs. |
| [src/category_matcher/api/schemas.py](../src/category_matcher/api/schemas.py) | Validates HTTP inputs and enforces accepted-versus-provisional response invariants. |
| [src/category_matcher/api/server.py](../src/category_matcher/api/server.py) | Launches one loopback Uvicorn worker with validated command-line configuration. |
| [src/category_matcher/api/service.py](../src/category_matcher/api/service.py) | Owns startup, the bounded request queue, inference-thread execution, event persistence, and shutdown. |
| [src/category_matcher/api/settings.py](../src/category_matcher/api/settings.py) | Validates service settings and maps supported environment variables to typed configuration. |

## Scripts

| File | Purpose |
|---|---|
| [scripts/api_client.py](../scripts/api_client.py) | Sends a category to the local match endpoint and prints its JSON response. |
| [scripts/benchmark_api.py](../scripts/benchmark_api.py) | Starts an isolated API and compares a fresh CLI reference while measuring HTTP latency and overload behavior. |
| [scripts/experiment_quantized_reranker.py](../scripts/experiment_quantized_reranker.py) | Evaluates or benchmarks an explicit reranker CPU profile without changing the active bundle. |
| [scripts/prepare_models.py](../scripts/prepare_models.py) | Prepares exactly the configured model revisions from cache or an explicitly requested download. |
| [scripts/prepare_onnx_models.py](../scripts/prepare_onnx_models.py) | Downloads missing multilingual ONNX assets on request and verifies pinned local files without changing the active API. |
| [scripts/run_user_examples.py](../scripts/run_user_examples.py) | Compares retrieval-only and reranked results for the original five user category examples. |
| [scripts/smoke_quantization.py](../scripts/smoke_quantization.py) | Checks pinned model/profile inference compatibility and output drift in isolation. |

## Data and review files

| File | Purpose |
|---|---|
| [data/google/aliases.jsonl](../data/google/aliases.jsonl) | Stores approved and pending alternative category names used as inputs when building bundles. |
| [data/google/evaluation/README.md](../data/google/evaluation/README.md) | Explains dataset origin, draft-label limitations, splits, and review requirements. |
| [data/google/evaluation/development.jsonl](../data/google/evaluation/development.jsonl) | Contains the original 200 AI-authored development examples for diagnostic evaluation. |
| [data/google/evaluation/development.audited.draft.jsonl](../data/google/evaluation/development.audited.draft.jsonl) | Preserves an unreviewed derivative with proposed corrections from a historical failure audit. |
| [data/google/evaluation/calibration.draft.jsonl](../data/google/evaluation/calibration.draft.jsonl) | Contains 200 AI-authored calibration proposals for review and draft-policy fitting. |
| [data/google/evaluation/test.draft.jsonl](../data/google/evaluation/test.draft.jsonl) | Contains 200 AI-authored test proposals that do not constitute an independent reviewed holdout. |
| [data/google/evaluation/user_examples.jsonl](../data/google/evaluation/user_examples.jsonl) | Records the original five user inputs with confirmed and proposed mapping evidence distinguished. |
| [data/google/evaluation/user_examples_2.jsonl](../data/google/evaluation/user_examples_2.jsonl) | Records five later real store titles and legacy model outputs with proposed labels awaiting review. |
| [data/google/reviews/README.md](../data/google/reviews/README.md) | Documents how to record explicit human decisions and apply source-bound review packets. |
| [data/google/reviews/development.pending.jsonl](../data/google/reviews/development.pending.jsonl) | Contains pending source-bound human review decisions for the development dataset. |
| [data/google/reviews/calibration.pending.jsonl](../data/google/reviews/calibration.pending.jsonl) | Contains pending source-bound human review decisions for calibration proposals. |
| [data/google/reviews/user_batch_2.pending.jsonl](../data/google/reviews/user_batch_2.pending.jsonl) | Contains pending source-bound review decisions for the five later user-supplied titles. |
| [data/google/reviews/development_audit.json](../data/google/reviews/development_audit.json) | Preserves historical AI audit proposals without promoting them to human ground truth. |
| [data/google/reviews/development_failure_notes.jsonl](../data/google/reviews/development_failure_notes.jsonl) | Preserves an earlier failure-review queue as historical notes rather than an importable review packet. |
| [data/google/reviews/development_followups.md](../data/google/reviews/development_followups.md) | Lists historical draft failure cases, alternatives, and unresolved reviewer decisions. |

## Tests

| File | Purpose |
|---|---|
| [tests/conftest.py](../tests/conftest.py) | Builds synthetic taxonomy/vector bundles shared by the regression tests. |
| [tests/test_api.py](../tests/test_api.py) | Covers HTTP contracts, removed feedback routes, bounded scheduling, deadlines, readiness, and inference failures. |
| [tests/test_benchmark.py](../tests/test_benchmark.py) | Checks resource-report accounting, swap observations, and qualification limits for short measurements. |
| [tests/test_boundaries.py](../tests/test_boundaries.py) | Enforces domain independence and prevents direct SQL execution inside API modules. |
| [tests/test_bundles.py](../tests/test_bundles.py) | Checks release/model integrity, staged publication, activation locks, rollback, and atomic pointer safety. |
| [tests/test_calibration.py](../tests/test_calibration.py) | Checks calibration abstention, qualification requirements, and rejection of reused holdout evidence. |
| [tests/test_cli.py](../tests/test_cli.py) | Covers CLI matching, experimental policy isolation, runtime backup/restore, and input export. |
| [tests/test_datasets.py](../tests/test_datasets.py) | Checks review provenance, split separation, and dataset leakage validation. |
| [tests/test_decision.py](../tests/test_decision.py) | Checks policy identity, threshold validation, assignment rules, and draft isolation. |
| [tests/test_default_ranking.py](../tests/test_default_ranking.py) | Ensures ranking models load lazily and default reranking can be explicitly disabled. |
| [tests/test_evaluation.py](../tests/test_evaluation.py) | Checks evaluation metrics, draft-label handling, candidate budgets, and per-example diagnostics. |
| [tests/test_generic_hierarchy.py](../tests/test_generic_hierarchy.py) | Checks placeholder abstention and provisional inheritance from grounded, compatible store ancestors. |
| [tests/test_matching.py](../tests/test_matching.py) | Covers canonical/alias lookup, context scope, retrieval branches, fusion, and generic-apparel safeguards. |
| [tests/test_memory.py](../tests/test_memory.py) | Checks parsing and accounting of native process memory observations. |
| [tests/test_quantization.py](../tests/test_quantization.py) | Checks CPU profile validation, forward-output contracts, and packed tensor accounting. |
| [tests/test_reranker.py](../tests/test_reranker.py) | Checks NLI scoring, rank fusion, revision-bound profiles, and CPU parameter alignment. |
| [tests/test_review.py](../tests/test_review.py) | Checks source-bound human review import, tamper rejection, and unlabeled input capture. |
| [tests/test_runtime.py](../tests/test_runtime.py) | Checks runtime migrations, corruption rejection, event preservation, and safe backup/restore. |
| [tests/test_source_ancestors.py](../tests/test_source_ancestors.py) | Checks ancestry validation, matching scope, persisted inputs, and review-binding compatibility. |
| [tests/test_specificity.py](../tests/test_specificity.py) | Checks parent-only NLI support and preservation of product-specific retrieval evidence. |
| [tests/test_spelling_and_broad.py](../tests/test_spelling_and_broad.py) | Checks conservative spelling interpretations, general-category matching, and unapproved audit labels. |
| [tests/test_storage.py](../tests/test_storage.py) | Checks bundle/database integrity, canonical scope, and sqlite-vec search equivalence. |
| [tests/test_store_labels.py](../tests/test_store_labels.py) | Checks merchandising titles, product descriptions, coordinated category labels, and weak-promotion safeguards. |
| [tests/test_taxonomy.py](../tests/test_taxonomy.py) | Checks source parsing, hierarchy integrity, malformed input rejection, and normalization collisions. |
| [tests/test_unmapped_contract.py](../tests/test_unmapped_contract.py) | Ensures uncertain/nonproduct titles never acquire an accepted category through permissive policies or serialization. |

## Documentation

| File | Purpose |
|---|---|
| [doc/README.md](../doc/README.md) | Provides reading order, current implementation status, resumption checklist, and outstanding work. |
| [doc/ARCHITECTURE.md](../doc/ARCHITECTURE.md) | Explains runtime ownership, matching stages, storage boundaries, and Mermaid architecture/build diagrams. |
| [doc/DECISIONS.md](../doc/DECISIONS.md) | Records implemented architecture decisions, constraints, and explicitly deferred platform work. |
| [doc/API.md](../doc/API.md) | Defines HTTP routes, request/response semantics, error behavior, and the WooCommerce contract. |
| [doc/OPERATIONS.md](../doc/OPERATIONS.md) | Documents setup, settings, rebuilds, activation, recovery, data review, diagnostics, and troubleshooting. |
| [doc/FILE_REFERENCE.md](../doc/FILE_REFERENCE.md) | Maps each maintained repository file to a one-line purpose and lists generated artifact roles. |
| [doc/SECURITY.md](../doc/SECURITY.md) | Defines trust boundaries, input validation, safe SQL/process/path handling, model trust, and sensitive-data rules. |
| [doc/LIMITATIONS_AND_ROADMAP.md](../doc/LIMITATIONS_AND_ROADMAP.md) | Records current limits, prioritized enhancements, scaling stages, and evidence required for future changes. |

## Generated files and directories

These are local outputs rather than maintained source files; names in angle
brackets vary by installation or release.

| Path | Purpose |
|---|---|
| `models/embedding_model.json` | Records the pinned embedding revision, dimension, and local snapshot path. |
| `models/nli_reranker_model.json` | Records the pinned NLI model revision, scoring profile, and local snapshot path. |
| `models/hub/` | Caches downloaded model/tokenizer snapshots and their backing files. |
| `artifacts/google/active.json` | Selects the artifact, immutable release, and previous release. |
| `artifacts/google/.service.lock` | Coordinates API residency with activation/rollback through file leases. |
| `artifacts/google/.staging.lock` | Serializes publication of staged bundle directories. |
| `artifacts/google/.releases/<release_id>.json` | Pins manifest/model hashes and an optional policy snapshot. |
| `artifacts/google/.releases/policy-<hash>.json` | Retains immutable qualified policy bytes selected by a release. |
| `artifacts/google/<version>/manifest.json` | Records source/configuration/implementation identity and bundle checksums. |
| `artifacts/google/<version>/audit.json` | Records taxonomy parsing counts, collisions, and source audit information. |
| `artifacts/google/<version>/aliases.json` | Contains compiled approved aliases read directly by the matcher. |
| `artifacts/google/<version>/taxonomy.sqlite3` | Stores canonical categories, hierarchy closure, and label/path vectors. |
| `artifacts/google/<version>/bm25/category_ids.json` | Maps BM25 document positions to canonical IDs. |
| `artifacts/google/<version>/bm25/params.index.json` | Stores BM25 index parameters. |
| `artifacts/google/<version>/bm25/vocab.index.json` | Stores the BM25 vocabulary mapping. |
| `artifacts/google/<version>/bm25/data.csc.index.npy` | Stores sparse BM25 matrix values. |
| `artifacts/google/<version>/bm25/indices.csc.index.npy` | Stores sparse BM25 matrix row indices. |
| `artifacts/google/<version>/bm25/indptr.csc.index.npy` | Stores sparse BM25 matrix column offsets. |
| `var/category_matcher.sqlite3` | Stores match-event requests/responses and runtime migration history. |
| `var/*.sqlite3-wal`, `var/*.sqlite3-shm` | Support live SQLite WAL transactions and coordination. |
| `var/backups/*.sqlite3` and `.sha256` sidecars | Preserve consistent runtime snapshots and their integrity hashes. |
| `reports/` | Holds regenerated evaluation, resource, HTTP, and experimental-policy diagnostics. |
| `reports/sprint_2/holdout_access.jsonl` | Records persistent holdout access and policy identity for reuse checks. |
| `.venv/`, `.cache/`, tool caches, `__pycache__/` | Hold installed dependencies and regenerable tool/interpreter caches. |

Historical optional model metadata may remain under `models/`; inspect retained
release manifests before removing snapshots. The active runtime does not read
pending review files or draft evaluation labels to make a match.
