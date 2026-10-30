# Category matcher

Match store category names to canonical Google product categories using exact
lookup, reviewed aliases, BM25, fuzzy matching, multilingual E5 embeddings, and NLI
reranking. Python 3.11, FastAPI, ONNX Runtime, and SQLite with `sqlite-vec` form the
current stack. Pinned INT8 models run locally on the CPU with two inference threads.

## Start the application

Run from the repository root using Python 3.11 and uv:

```bash
uv sync --frozen
uv run python scripts/prepare_onnx_models.py --download
uv run category-matcher build --activate
uv run category-matcher-api
```

For an existing compatible installation, start with `uv run category-matcher-api`.
Model preparation requires network access only when downloading weights. Bundles,
model weights, and runtime databases are generated locally and ignored by Git.

API documentation: `http://127.0.0.1:8000/docs`.
Readiness: `http://127.0.0.1:8000/health/ready`.
Stop the server with **Ctrl+C**; Ctrl+Z keeps its port and artifact lock occupied.

## Prepare multilingual ONNX models

Developers can automatically download and prepare the multilingual model files:

```bash
uv run --frozen python scripts/prepare_onnx_models.py --download
```

The command fetches only the five required files per model into
`models/multilingual-e5-small/` and `models/multilingual-nli/`. It verifies pinned
revisions, sizes, and SHA-256 checksums from `config/onnx_model_pins.json`, reuses
verified local files, and refuses to overwrite mismatched files. Omit `--download`
for offline file verification. Weights remain Git-ignored; only the pins and script
belong in Git. See [operations](doc/OPERATIONS.md#multilingual-onnx-model-preparation).

Preparation also writes `models/onnx_embedding_model.json` and
`models/onnx_nli_reranker_model.json`. Build and activate a new bundle before
starting an upgraded API; old embeddings and policies are incompatible.

The default embedding model is `Xenova/multilingual-e5-small`; reranking uses
`onnx-community/multilingual-MiniLMv2-L6-mnli-xnli-ONNX`. Input language defaults
to `auto`; explicit enabled hints include `ar`, `bn`, `de`, `en`, `es`, `fr`, `hi`,
`id`, `it`, `ja`, `ko`, `pt`, `ru`, `tr`, and `zh`, including regional variants.
Google IDs and output labels remain English. Language support and quantized ranking
have not been independently accuracy-qualified; semantic results remain provisional.

For a lean deployment, use `uv sync --frozen --no-dev`. PyTorch and
SentenceTransformers are optional: `uv sync --frozen --extra legacy` retains access
to legacy workflows. Do not copy old weights, caches, or development environments
into a storage-constrained deployment. Peak resource use still needs measurement.

## Match a category

```bash
uv run category-matcher match 'Hoodies & Sweatshirts|Clothing'

curl -X POST http://127.0.0.1:8000/v1/categories/match \
  -H 'Content-Type: application/json' \
  -d '{"taxonomy_key":"google","category":"Collections","source_ancestors":["Clothing","Men","Tops","Hoodies & Sweatshirts|Clothing"]}'
```

Send the store category name separately from its ancestors, ordered root to parent.
Only `matched` assigns `category`. A `provisional` result can offer
`provisional_category` with `requires_review: true`. Unclear inputs remain unmapped.
The service currently has no qualified policy for automatic semantic acceptance.

## Folder structure

```text
.
├── README.md                              # Quick start and documentation links
├── AGENTS.md                              # Codex-native instructions shared by agents
├── CLAUDE.md                              # Claude import of the shared instructions
├── GEMINI.md                              # Gemini import of the shared instructions
├── WORKFLOW.md                            # Contributor and agent development workflow
├── doc/                                   # Developer documentation
│   ├── README.md                          # Handoff guide and reading order
│   ├── ARCHITECTURE.md                     # Components, matching, and Mermaid diagrams
│   ├── DECISIONS.md                        # Architecture decisions and consequences
│   ├── API.md                             # HTTP contract and integration examples
│   ├── OPERATIONS.md                      # Builds, recovery, review, and troubleshooting
│   ├── SECURITY.md                        # Security, validation, and sensitive-data rules
│   ├── LIMITATIONS_AND_ROADMAP.md          # Limitations, future work, and scaling stages
│   └── FILE_REFERENCE.md                  # One-line description of each maintained file
├── src/category_matcher/                  # Matching engine and supporting workflows
│   └── api/                               # FastAPI routes, contracts, settings, and queue
├── config/                                # Pinned models and ranking profile
├── data/google/                           # Build inputs and development datasets
│   ├── aliases.jsonl                      # Approved and pending name mappings
│   ├── evaluation/                        # Draft evaluation/calibration/test inputs
│   └── reviews/                           # Human review packets and historical evidence
├── scripts/                               # Preparation, client, and measurement tools
├── tests/                                 # Regression tests and synthetic fixtures
├── google_category_list.txt               # Source taxonomy
├── pyproject.toml                         # Package, entry points, dependencies, and tools
├── uv.lock                                # Locked dependency resolution
├── CATEGORY_MATCHING_IMPLEMENTATION_PLAN.md # Historical architecture plan
├── CATEGORY_MATCHING_SPRINT_PLAN.md        # Historical sprint plan
├── models/                                # Generated local model files; Git-ignored
├── artifacts/google/                      # Generated bundles/releases; Git-ignored
├── var/                                   # Runtime match-event database; Git-ignored
└── reports/                               # Generated diagnostics; Git-ignored
```

Generated directories appear as their workflows create them. Hidden repository
configuration files are covered in the file reference.

## Developer documentation

Start with [the handoff guide](doc/README.md).

| Document | Contents |
|---|---|
| [Architecture](doc/ARCHITECTURE.md) | Runtime topology, matching pipeline, storage, and build flow diagrams |
| [Architecture decisions](doc/DECISIONS.md) | Why the current design works this way and its tradeoffs |
| [API](doc/API.md) | Routes, inputs, outputs, error behavior, and WooCommerce integration |
| [Operations](doc/OPERATIONS.md) | Setup, updates, configuration, backups, diagnostics, and troubleshooting |
| [File reference](doc/FILE_REFERENCE.md) | One-line purpose and ownership of every maintained project file |
| [Limitations and scaling](doc/LIMITATIONS_AND_ROADMAP.md) | Current constraints, enhancement priorities, and staged scaling path |
| [Security](doc/SECURITY.md) | Trust boundaries, validation, safe storage/process execution, and data handling |
| [Development workflow](WORKFLOW.md) | Contributor steps, verification, and shared agent entry points |

The historical plans describe earlier scope; these developer docs describe the
current implementation. The feedback endpoint and feedback-to-alias workflow have
been removed. Dataset review and manually maintained approved aliases remain.

## Coding agents

[AGENTS.md](AGENTS.md) is Codex's native project instruction file and the shared
source of project rules. Codex reads it directly; `CLAUDE.md` and `GEMINI.md`
import it for their respective tools. Start in this repository and
follow [WORKFLOW.md](WORKFLOW.md); if an agent does not load instructions
automatically, ask it to read `AGENTS.md` first. These files define development
expectations; security and quality controls still need code and appropriate checks.

## Development checks

Run the checks appropriate to your change:

```bash
uv run category-matcher inspect
uv run category-matcher verify
uv run category-matcher check-data
uv run --extra legacy pytest
uv run ruff check src tests scripts
```

Human review and independent accuracy qualification remain pending. Draft dataset
scores are diagnostic evidence, not a real-world accuracy claim. See the
[validation status and handoff priorities](doc/README.md#current-status).
