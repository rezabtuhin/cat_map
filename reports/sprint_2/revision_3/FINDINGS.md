# Sprint 2: store-label handling, promotion guard, and review workflow

The new user inputs exposed a distinction between product categories and store
sections. This revision keeps product-type evidence broad where appropriate and
asks for context when a title does not identify products. It also guards weak
model promotions and adds a usable collection/review workflow.

Artifact: `eed281a45624da69-a0a29f98cc37`. **Decision: revise.** Automatic semantic
acceptance remains disabled. The original taxonomy and the three original data
splits are unchanged; test proposals remain unscored.

The user explicitly confirmed that uncertain inputs must stay unmapped. Only
accepted `matched` results may populate the assigned `category`; every other
result keeps it null. Suggestions and experimental previews are never automatic
assignments. For the three store-section titles, no mapped category is produced.

## The five new real store inputs

These strings and their LAYA outputs were supplied by the user. Intended IDs and
uncertainty labels in the regression file are still AI proposals, not human
approvals. The inputs were used during development and cannot be a holdout.

| Input | Reported LAYA output | Current default behavior |
|---|---|---|
| Jackets | 1128 — Life Jackets | 5598 — Coats & Jackets first, provisional |
| Collections | 203 — Outerwear | Needs clarification; no product category assigned |
| Fitness Equipment | 4669 — Exercise Equipment Mats | 990 — Exercise & Fitness first, provisional; Fitness & Nutrition retained as another broad interpretation |
| New Luma Yoga Collection | 3640 — Yoga & Pilates Mats | Needs clarification; no product category assigned |
| Erin Recommends | 100 — Backpacks | Needs clarification; no product category assigned |

Collection/recommendation titles do not justify inventing a product type. The yoga
title identifies an activity but does not establish clothing, equipment, or a
mixed assortment. Product descriptions are needed to resolve it. The matcher can
use supplied unresolved product context in place of a store-section title; it
keeps any resolved parent constraint. Canonical lookups still run first.

For Jackets, a whole alternative in the canonical label Coats & Jackets is
literal evidence. Fitness Equipment supports a broad Fitness interpretation,
rather than the extra noun Mats. The rule matches whole coordinated-label
alternatives after light retrieval normalization and an optional generic group
suffix. It never strips a specific noun such as Mats or a qualifier such as Life.
It works within the selected parent scope and retains competing broad labels.
This evidence gives ranking priority, not an approved alias or automatic match.

See [the five-input report](user_batch_2.json),
[input records](../../../data/google/evaluation/user_examples_2.jsonl), and
[pending review packet](../../../data/google/reviews/user_batch_2.pending.jsonl).

## Ranking changes and development evidence

The weak-promotion guard compares the retrieval winner with the model's promoted
candidate. If both have semantic evidence, the retrieval winner's similarity is
more than 0.03 higher, and the promoted candidate's raw entailment margin is below
1.0, it retains the retrieval winner first. Original model ranks/scores remain
available. It records a `promotion_conflict` and requires clarification under a
decision policy. Missing semantic scores do not establish a support gap.

The thresholds were selected using a small
[development-only comparison](promotion_guard_experiment.json) on saved NLI
predictions and [retrieval evidence](retrieval_evidence.json). They are ranking
heuristics, not calibrated correctness probabilities. The earlier parent-only
specificity safeguard remains active. Neither safeguard authorizes acceptance of
a retrieval result simply because it overrules the model.

The [rebuilt end-to-end evaluation](retrieval_comparison.json) uses the unchanged
original 200 draft examples. The comparable non-exact resolvable cohort is 120:

| Pipeline | Top-1 | Top-3 | Final-shortlist recall |
|---|---:|---:|---:|
| Original hybrid pipeline | 70/120, 58.3% | 76.7% | 98.3% |
| Previous specificity revision | 94/120, 78.3% | 108/120, 90.0% | 98.3% |
| Current revision | 96/120, 80.0% | 109/120, 90.8% | 98.3% |

The bird-cage and lawn-mower cases now rank their original proposed targets
first. DVD movies now ranks Film & Television DVDs first, replacing Electronics
> Video. The original label is the broader DVDs & Videos, so the primary top-1
metric continues to count this as an error. The more specific DVD target was
already one of the six earlier audit proposals; no new label change was invented
for this run. [Audit sensitivity metrics](audit_rescore.json) use those same
unapproved proposals separately: 98/116 (84.5%) top-1 and 109/116 (94.0%) top-three.
The original 120-case comparison remains primary. There are 24 remaining
original-label top-1 failures.

The hoodie query retains Shirts & Tops first and still exposes its parent-only
model disagreement. See [the original five-input report](user_examples.json).

The [new draft calibration policy](draft_policy.json) accepts 73/140 resolvable
fitted proposals (52.1% coverage), up from 67/140 (47.9%). There are zero errors
on those same fitted, unreviewed labels; seven acceptances use the model path.
The one-sided 95% precision lower bound is 96.0%. These figures do not establish
independent accuracy. Coverage remains below 80%, and independent qualification
has not passed. The policy permits previews only and is bound to this artifact.

## Label review and real-input collection

Three new CLI commands work without model weights or a vector artifact:

- `capture-inputs`: store real queries and available product context without
  inventing labels. Unlabeled records have a null expected status and cannot be
  evaluated as though they were labeled.
- `review-export`: generate editable pending decisions bound to the dataset,
  taxonomy, example cohort, and original row contents. Existing proposals do not
  become approvals.
- `review-apply`: validate explicitly supplied human decisions and write a new
  dataset. It preserves source/split metadata, rejects stale or invalid decisions,
  and refuses to overwrite an existing output. It records the supplied human
  declaration rather than authenticating a reviewer identity.

There are ready-to-edit packets for 200 development proposals, 200 calibration
proposals, and the five new real store inputs. No decisions were approved or
applied by this work. Test proposals were not exported. Read the
[review instructions](../../../data/google/reviews/README.md) and the
[remaining-error queue](LABEL_REVIEW.md).

## Memory profiling

The benchmark now records this process's native resident, private/internal,
compressed, and physical-footprint memory before model load, after warmup, every
20 requests, after inference, and after Python garbage collection. It uses the
macOS task-info interface for the current process only, with the layout checked
against Apple's installed SDK. It also reports model parameter byte counts;
these exclude activations, framework overhead, and allocator caches. Unsupported
platforms/counters are reported as unavailable rather than fabricated.

This supplements the earlier system-wide swap observations. It can show whether
the matcher's own footprint grows during the run, rather than attributing every
system swap event to it. The strict resource gate remains unchanged. See the
[resource report](resource_report.json) for the recorded run.

The 200 serial model-path requests measured p50 845 ms, p95 1,429 ms, maximum
2,078 ms, and 1.07 requests/second. Process peak resident memory was 1.27 GiB.
Latency and process-memory limits pass. The system swapout counter increased by
113,112 pages (8,772 during model load/warmup and 104,340 during timed inference);
the five-second idle baseline observed no increase. The conservative
no-observed-system-swap check fails, so resource qualification remains unpassed.

This process's physical footprint was 1.04 GiB after warmup and 1.03 GiB after
inference, with window samples between approximately 1.03 and 1.06 GiB. It did
not show sustained footprint growth in this run. Its compressed-memory counter
was 0.67 GiB after warmup and 0.73 GiB after inference. Python garbage collection
left the footprint unchanged while touching/decompressing some resident pages.
This is evidence of a compressed working set under memory pressure, not proof
of a Python leak or sole responsibility for system swaps. It does not establish
long-running leak freedom or concurrent capacity.

The live parameters account for about 418 MiB in the embedding model and 541 MiB
in the reranker, before framework overhead and activations. The next memory
optimization should therefore measure a smaller/quantized inference profile or
isolated serving with sufficient memory headroom, with a full ranking comparison
before adoption. Per-request garbage collection is not justified by these data.

## Verification and next gate

Regression tests cover collection titles with and without product context,
coordinated-label ranking, scope/qualifier preservation, model disagreement,
source-bound human review, unlabeled capture, ID-only examples, and the native
memory layout. Original data checksums and historical JSON reports are preserved.
The six additional unmapped-contract tests verify both CLI serialization and
abstention with a deliberately permissive test policy.
All 99 tests, Ruff lint/format, and diff whitespace checks pass. The
[final validation](validation.json) verifies evidence-version consistency, the
live unmapped behavior, unchanged source/split checksums, documentation links,
and 405 pending review decisions with zero new approvals.
All 16,746 canonical round trips and 16 scoped SQLite/NumPy checks pass on the
rebuilt 5,582-category, 11,164-vector artifact. Quality, calibration, resource,
and both user-example reports identify the same artifact.

Human review, independently collected qualification inputs, the remaining ranking
failures, and resource qualification still require work. Start with the five-row
real-input review packet; provide product context for the three store-section
titles rather than approving arbitrary product IDs. The API remains deferred.

Reproduce the current run from the project root:

```bash
uv run category-matcher build
uv run category-matcher evaluate --allow-draft \
  --output reports/sprint_2/revision_3/retrieval_comparison.json
uv run category-matcher calibrate --allow-draft \
  --output reports/sprint_2/revision_3/draft_policy.json
uv run category-matcher benchmark --requests 200 --idle-baseline-seconds 5 \
  --output reports/sprint_2/revision_3/resource_report.json
uv run python scripts/run_user_examples.py \
  --output reports/sprint_2/revision_3/user_examples.json
uv run python scripts/run_user_batch_2.py
uv run category-matcher verify
uv run python scripts/summarize_specificity_run.py \
  --reports reports/sprint_2/revision_3 --previous reports/sprint_2/specificity
```
