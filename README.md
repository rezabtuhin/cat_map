# Category matcher

Local Google product-category matching using exact lookup, reviewed aliases,
BM25, fuzzy matching, and BGE semantic retrieval stored in SQLite with `sqlite-vec`.
An optional MiniLM reranker is available as an early Sprint 2 experiment.

Current [Sprint 2 status](reports/sprint_2/STATUS.md) and
[findings](reports/sprint_2/FINDINGS.md) include unresolved ranking failures.

**Non-exact results are provisional candidates.** Similarity and reranker scores
are ranking signals, not correctness probabilities. Sprint 2 implements calibration,
clarification/no-match policy rules, and qualification checks. The fitted draft
policy is not validated and can only preview decisions. An API and production
qualification remain future sprint work.

See [the sprint plan](CATEGORY_MATCHING_SPRINT_PLAN.md) and
[implementation plan](CATEGORY_MATCHING_IMPLEMENTATION_PLAN.md).

## Setup

Run these commands from the project root. Python 3.11 and uv are required.

```bash
uv sync --frozen
uv run python -m category_matcher.encoder
uv run category-matcher build
uv run category-matcher verify
```

The explicit model-download command requires network access. Build and inference
use the downloaded model locally. Model weights are cached under `models/`;
generated taxonomy/vector bundles live under `artifacts/google/`. Both are ignored
by Git. A source, alias, dependency, or matching-code change requires rebuilding;
the manifest pins revisions, configuration, file checksums, and implementation hashes.

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

Without a qualified policy, only canonical lookups and approved aliases return `matched`.
The `Tops` alias was
approved from the user's explicit confirmation of ID `212`. Other draft aliases
are inactive until reviewed. Rebuild after changing aliases.

## Optional reranking experiment

```bash
uv run python -m category_matcher.reranker
uv run category-matcher match 'Smart Phone' --rerank
uv run category-matcher match 'Hoodies & Sweatshirts|Clothing' --rerank
uv run python scripts/run_user_examples.py
```

This downloads the pinned local MiniLM model on the first command. Reranked
non-exact outputs remain provisional. The comparison script writes
`reports/sprint_2/user_examples.json` and does not change the decision policy.
The observed Apple Silicon CPU runtime requires copying misaligned checkpoint
parameters into aligned storage; the adapter applies and tests that workaround.
The draft benchmark does not justify enabling this reranker by default.

## Evaluate and verify

```bash
uv run category-matcher evaluate --allow-draft
uv run category-matcher evaluate --allow-draft --rerank \
  --output reports/sprint_2/retrieval_comparison.json
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
  --output reports/sprint_2/draft_policy.json
uv run category-matcher match 'Smart Phone' \
  --experimental-policy reports/sprint_2/draft_policy.json
uv run category-matcher benchmark --requests 200
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

`benchmark` measures 200 serial warm requests including query encoding and retrieval,
with model warmup reported separately. It reports process memory and system swap
observations; it does not establish production concurrency or matching accuracy.

## Current scope

- Google English taxonomy only; 5,582 categories and two vectors per category.
- CLI prototype, local CPU inference, one command process at a time.
- Small modules separate domain records, parsing, storage, retrieval, and orchestration.
- No remote API calls during inference, web service, frontend, or deployed site.
- Additional vendors and broader scaling work remain deferred until the matching
  approach has been validated, as requested.
