# Category matcher

Match store category names to canonical Google product categories using exact
lookup, reviewed aliases, BM25, fuzzy matching, BGE embeddings, and NLI reranking.
Python 3.11, FastAPI, PyTorch, and SQLite with `sqlite-vec` form the current stack.
Models run locally on the CPU.

## Start the application

Run from the repository root using Python 3.11 and uv:

```bash
uv sync --frozen
uv run python scripts/prepare_models.py --download
uv run category-matcher build --activate
uv run category-matcher-api
```

For an existing compatible installation, start with `uv run category-matcher-api`.
Model preparation requires network access only when downloading weights. Bundles,
model weights, and runtime databases are generated locally and ignored by Git.

API documentation: `http://127.0.0.1:8000/docs`.
Readiness: `http://127.0.0.1:8000/health/ready`.
Stop the server with **Ctrl+C**; Ctrl+Z keeps its port and artifact lock occupied.

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
├── doc/                                   # Developer documentation
│   ├── README.md                          # Handoff guide and reading order
│   ├── ARCHITECTURE.md                     # Components, matching, and Mermaid diagrams
│   ├── DECISIONS.md                        # Architecture decisions and consequences
│   ├── API.md                             # HTTP contract and integration examples
│   ├── OPERATIONS.md                      # Builds, recovery, review, and troubleshooting
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

The historical plans describe earlier scope; these developer docs describe the
current implementation. The feedback endpoint and feedback-to-alias workflow have
been removed. Dataset review and manually maintained approved aliases remain.

## Development checks

Run the checks appropriate to your change:

```bash
uv run category-matcher inspect
uv run category-matcher verify
uv run category-matcher check-data
uv run pytest
uv run ruff check src tests scripts
```

Human review and independent accuracy qualification remain pending. Draft dataset
scores are diagnostic evidence, not a real-world accuracy claim. See the
[validation status and handoff priorities](doc/README.md#current-status).
