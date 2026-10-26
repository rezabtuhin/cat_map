# Sprint 2 specificity correction

This revision adds an explicit model/retrieval disagreement safeguard and changes
the NLI pair wording to category statements. The original taxonomy and all
development/calibration/test labels remain unchanged. The test proposals have
not been scored. **The release decision remains revise.**

Artifact: `eed281a45624da69-3cf82ad83e4d`.

## Implemented behavior

The NLI profile now compares:

```text
The product category is <input>.
The product category is <candidate label>.
```

The pinned model revision, 100-candidate scoring prefix, single inference pass,
and retrieval/model rank weights are unchanged. Pair templates are validated in
[the configuration loader](../../../src/category_matcher/reranker.py) and saved
in each artifact. Downloaded model metadata alone does not control the runtime
wording. Changing templates invalidates prior artifact-bound decision policies.

When the model's first candidate is the supplied parent, that parent has positive
entailment support, and every scored descendant has nonpositive support, the
matcher preserves retrieval order and records a `specificity_conflict`. There
must be at least two scored descendants; unscored candidates are not treated as
rejected. This rule applies to an explicit parent ID and to resolved pipe/context
constraints. It contains no category IDs or hoodie-specific vocabulary.

This is a disagreement fallback, not proof that the retrieval winner is correct.
The result stays provisional and includes clarification text. A decision policy
must request clarification for this flag even if similarity and margin would
otherwise permit acceptance. Original model ranks, model scores, and retrieval
evidence remain available in the candidates. Exact matches and attribute-only
context handling keep their existing behavior.

## Experiments and development evidence

The [representation experiment](representation_comparison.json) scores the same
frozen development candidate lists using category wording, immediate-parent
context, and alternative-label wording. A separate
[failure-only pilot](representation_pilot.json) also tests reversed entailment.
The pilot is not an aggregate accuracy comparison. Reversing entailment often
promotes unsupported child categories and was rejected.

[Selection results](representation_selection.json) compare the complete
development cohort and disclose changed predictions. The two-representation
category/context ensemble and the selected single-pass category profile both
reach 94/120 top-1 in this diagnostic; the ensemble adds inference work for a
top-three gain. The single-pass profile keeps the current resource budget.
All model selection uses reused unreviewed development data, not independent
quality evidence. The live rebuilt pipeline is evaluated separately below.

The [end-to-end rebuilt-pipeline report](retrieval_comparison.json) confirms:

| Pipeline | Top-1 on 120 non-exact resolvable drafts | Top-3 | Final shortlist |
|---|---:|---:|---:|
| Original hybrid | 70/120, 58.3% | 76.7% | 98.3% |
| Previous product-worded NLI pipeline | 93/120, 77.5% | 87.5% | 98.3% |
| Current category-worded NLI pipeline | 94/120, 78.3% | 90.0% | 98.3% |

All three rows use the unchanged original draft labels. The current pipeline
still has 26 original-label top-1 failures. Top-1 counts by slice are 44/60
paraphrases, 27/32 typos, and 23/28 hierarchy examples. The 40 canonical exact
examples pass and are excluded from the 120-example denominator above. Ranking
success is separate from automatic acceptance and real-world accuracy.

Four previous top-1 failures improve: phone cases, dog food, cat toys, and fish
food. The lawn-mower paraphrase regresses from first to third. DVD movies regresses
to Electronics > Video, a related but incorrect product type. Printed books moves
from Books to Print Books, matching the earlier audit proposal, but still counts
as a regression against the unchanged original label. These three losses produce
the net gain of one original-label success. The primary report keeps counting
all three as errors. The same six earlier unapproved audit proposals give
95/116 (81.9%) top-1 and 107/116 (92.2%) top-three in a
[separate sensitivity rescore](audit_rescore.json), versus 91/116 top-1 in the
previous pipeline. This is not new ground truth or independent validation.

The [live user examples](user_examples.json) now rank:

| Input | First canonical category | Status |
|---|---|---|
| Clothing | 1604 — Clothing | Matched canonical label |
| Tops | 212 — Shirts & Tops | Matched user-approved alias |
| Hoodies & Sweatshirts\|Clothing | 212 — Shirts & Tops | Provisional; specificity conflict requires clarification |
| Smart Phone | 267 — Mobile Phones | Provisional |
| Eco Friendly\|Clothing | 1604 — Clothing | Provisional attribute/context fallback |

The hoodie result preserves retrieval evidence, records the original model order,
and excludes the unsupported baby-tops category. Its best-candidate ranking is
restored; the semantic mapping has not been automatically accepted or independently
validated. The [draft-policy preview](hoodie_preview.json) also requests clarification.

The [new draft policy](draft_policy.json) remains preview-only. On the same
unreviewed calibration examples used for fitting, it accepts 67/140 resolvable
inputs (47.9% coverage), with zero fitted errors and only one model-path
acceptance. Coverage has not improved; it is below the 80% pilot target. The
one-sided 95% precision lower bound is 95.6%. These figures do not establish
independent precision or justify automatic semantic acceptance. The new policy
is bound to this artifact, and earlier policies cannot be reused with it.

## Local resource diagnostics

The benchmark now separates system swap observations during an optional idle
baseline, model load/warmup, and timed inference. It also records each window of
20 requests and the platform. Use `--idle-baseline-seconds 5` to capture the idle
observation. These are system-wide counters and cannot by themselves establish
which process caused swapping. The strict existing resource checks remain in
place; idle activity is not used to excuse a failed check.

The [200-request run](resource_report.json) measured:

- Warm p50 884 ms, p95 1,211 ms, maximum 1,487 ms; 1.08 serial requests/second.
- Process peak resident memory 1.14 GiB; load/warmup 7.13 seconds, untimed.
- Zero observed swapout-page increase during the five-second idle baseline and
  model load/warmup. Timed inference observed 57,944 additional system swapout
  pages, concentrated in three of ten 20-request windows. The other seven windows
  recorded zero increases.

Latency and process-memory limits pass. The no-observed-system-swap check fails,
so the combined resource gate remains unpassed. The phase measurements narrow
when swapping occurred, but cannot attribute it solely to this process. The new
run is slower than the preceding 779 ms p95 observation; different system memory
conditions may contribute, and a causal explanation is not established. A
controlled memory-profile run is still needed. Concurrent capacity is untested.

## Verification and remaining work

All 71 tests pass, including the scope-only fallback, missing/positive child
evidence, preservation of retrieval evidence, policy clarification, template
validation, and separation of swap phases. Ruff lint/format and diff whitespace
checks pass. No dependency or model download is needed for this revision.
The rebuilt real artifact also passes 16,746 canonical round trips and 16 scoped
SQLite/NumPy checks, retaining all 5,582 categories and 11,164 normalized vectors.
All generated policy, quality, user-example, and resource evidence identifies
the same current artifact. See the [updated decision](proof_of_concept_decision.json).

Human review and independent input collection remain required. The generated
[label-review queue](LABEL_REVIEW.md) presents original draft targets, current alternatives,
and earlier audit proposals without approving them. It covers remaining errors
and proposed corrections, rather than replacing review of every label. After
review and the remaining ranking corrections, freeze the pipeline, recalibrate,
and qualify on independently collected inputs. The API remains deferred until
the quality/resource gates pass.

Reproduce the frozen development and resource runs from the project root:

```bash
uv run category-matcher build
uv run category-matcher evaluate --allow-draft \
  --output reports/sprint_2/specificity/retrieval_comparison.json
uv run category-matcher calibrate --allow-draft \
  --output reports/sprint_2/specificity/draft_policy.json
uv run category-matcher benchmark --requests 200 --idle-baseline-seconds 5 \
  --output reports/sprint_2/specificity/resource_report.json
uv run python scripts/run_user_examples.py \
  --output reports/sprint_2/specificity/user_examples.json
uv run category-matcher verify
uv run python scripts/summarize_specificity_run.py
```

The comparison experiments use saved previous candidate pools and the already
downloaded model. Run `scripts/experiment_specificity.py` with `--pilot` to reproduce
the failure-only pilot, or without it and with `--variants category context alternatives`
to reproduce the complete representation comparison. Always provide a new output
path when preserving an earlier experimental report.
