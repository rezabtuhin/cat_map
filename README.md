# Category matcher

Local Google product-category matching using exact lookup, reviewed aliases,
BM25, fuzzy matching, and BGE semantic retrieval stored in SQLite with `sqlite-vec`.
The default hybrid pipeline adds category entailment rank fusion with a pinned
local NLI model. Canonical lookups avoid model inference.

Current [Sprint 2 status](reports/sprint_2/STATUS.md) and
[latest findings](reports/sprint_2/improvement/FINDINGS.md) include unresolved
ranking failures. Top-1 ranking improved from 58.3% to 77.5% on reused draft
development inputs; this is not independently validated accuracy.

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
uv run python -m category_matcher.reranker \
  --model-name cross-encoder/nli-deberta-v3-small \
  --revision fa2804872c3b4bd748f38c0185cc85775361e735 \
  --output models/nli_reranker_model.json
uv run category-matcher build
uv run category-matcher verify
```

The explicit model-download commands require network access. Build and inference
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

Without a qualified policy, canonical lookups, approved aliases, and conservative
literal general-category lookups return `matched`.
The `Tops` alias was
approved from the user's explicit confirmation of ID `212`. Other draft aliases
are inactive until reviewed. Rebuild after changing aliases.

## Ranking configuration

```bash
uv run category-matcher match 'Smart Phone' --rerank
uv run category-matcher match 'Hoodies & Sweatshirts|Clothing' --no-rerank
uv run python scripts/run_user_examples.py \
  --output reports/sprint_2/improvement/user_examples.json
```

Hybrid matching uses the artifact's pinned NLI rank-fusion profile by default;
`--no-rerank` selects retrieval only. Other retrieval methods do not use it.
The first 100 candidates receive model scores, combined with retrieval ranks;
all remaining candidates stay available. Reranked non-exact outputs remain
provisional. The comparison script does not change the decision policy.
The observed Apple Silicon CPU runtime requires copying misaligned checkpoint
parameters into aligned storage; the adapter applies and tests that workaround.
The older MiniLM and BGE reranker experiments remain available through explicit
`--reranker-model` overrides; neither is the selected profile. The NLI pipeline
improves aggregate development ranking, but still regresses on the hoodie input:
it ranks Clothing first, while retrieval-only ranks Shirts & Tops first. Both
avoid the unsupported baby category. This error still needs resolution.

## Evaluate and verify

```bash
uv run category-matcher evaluate --allow-draft --rerank \
  --output reports/sprint_2/improvement/retrieval_comparison.json
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
  --output reports/sprint_2/improvement/draft_policy.json
uv run category-matcher match 'Smart Phone' \
  --experimental-policy reports/sprint_2/improvement/draft_policy.json
uv run category-matcher benchmark --requests 200 \
  --output reports/sprint_2/improvement/resource_report.json
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
actual model-path inputs rather than canonical/general-category shortcuts. The
latest run measured 779 ms p95 and 1.18 GiB peak process memory, but observed system
swap activity, so the combined resource gate has not passed. It does not establish
production concurrency or matching accuracy.

## Current scope

- Google English taxonomy only; 5,582 categories and two vectors per category.
- CLI prototype, local CPU inference, one command process at a time.
- Small modules separate domain records, parsing, storage, retrieval, and orchestration.
- No remote API calls during inference, web service, frontend, or deployed site.
- Additional vendors and broader scaling work remain deferred until the matching
  approach has been validated, as requested.
