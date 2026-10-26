# Sprint 2 ranking improvement

The updated default pipeline improves top-1 ranking from **70/120 (58.3%) to
93/120 (77.5%)** on the original non-exact, resolvable development proposals.
These are reused, unreviewed draft examples. This measures development progress,
not independent accuracy. **The proof-of-concept decision remains revise.**

Current artifact: `eed281a45624da69-e7bf07b4426d`.

## Implemented configuration

- BGE-base embeddings, SQLite/sqlite-vec, BM25, fuzzy retrieval, and existing
  hierarchy constraints remain the candidate-generation stages.
- Whole-label spelling evidence considers one edit against canonical labels and
  approved aliases. Only unique corrections of at least five characters receive
  ranking priority. Known taxonomy words are preserved; ambiguous alternatives
  are exposed. Spelling priority does not authorize an automatic match.
- Explicit general-category wrappers can resolve a literal canonical label.
  Full-label matching precedes stripping generic suffixes, preserving categories
  such as General Office Supplies and Handheld Devices. Other normalizations
  remain provisional; the system never invents a common-ancestor answer.
- The pinned `cross-encoder/nli-deberta-v3-small` model scores the first 100
  candidates as category entailment pairs. Its ranking is fused with the original
  retrieval ranking: `1/(10 + retrieval_rank) + 3/(10 + entailment_rank)`.
  Unscored tail candidates retain only their retrieval contribution. The unique
  eligible spelling candidate remains protected after fusion.
- [Ranking configuration](../../../config/category_ranking.json) is bound into
  each artifact. Changing it requires rebuilding and recalibrating. Models load
  lazily; exact lookups and literal general-category lookups avoid model inference.
  `--no-rerank` runs the retrieval-only pipeline.

Raw entailment margins, fused ranks, and semantic similarities are ranking signals.
They are not probabilities that a category is correct. Model results stay
provisional without an independently qualified decision policy.

## Development comparison

All rows below use the original labels and the same 120 resolvable non-exact inputs.
The unchanged original report remains in the parent directory.

| Pipeline | Top-1 | Top-3 | Final-shortlist recall |
|---|---:|---:|---:|
| Previous hybrid pipeline | 70/120, 58.3% | 76.7% | 98.3% |
| Updated spelling/general-category rules, retrieval only | 85/120, 70.8% | 84.2% | 98.3% |
| Updated rules plus selected entailment rank fusion | 93/120, 77.5% | 87.5% | 98.3% |

The model's incremental gain over updated rules is eight examples. Model and
fusion settings were chosen using development data. These are pipeline results;
the rule contribution must not be attributed entirely to the neural model.
See [predictions](retrieval_comparison.json) and
[fusion selection](entailment_fusion_ablation.json).

Two replacement models were also tested by sorting their scores alone over the
same frozen previous candidate pool, scoring a prefix of 100: BGE-reranker-base
reached 53/120 top-1 and NLI alone reached 65/120. Neither justified replacing
retrieval order directly. These diagnostics differ from the final rule/fusion
pipeline. The earlier MiniLM experiment used a different scoring budget and
remains historical evidence, rather than a controlled model-only comparison.

## Label audit and unresolved errors

All 50 previous top-1 failures were inspected. Six label changes are proposed in
[the audit](label_audit.json) and a separate
[draft dataset](../../../data/google/evaluation/development.audited.draft.jsonl).
No human approval was added, and the original file was preserved.

| Example | Original draft label | Proposed interpretation |
|---|---|---|
| Printed reading books | Books, 784 | Print Books, 543543 |
| Movies on DVD | DVDs & Videos, 839 | Film & Television DVDs, 543527 |
| Adult oriented products | Mature, 772 | Needs clarification |
| Soas | Sofas, 460 | Needs clarification: sofas/soaps and other readings compete |
| Shes | Shoes, 187 | Needs clarification: insufficient spelling/context evidence |
| Tos | Toys, 1253 | Needs clarification: Toys and the approved Tops alias compete |

Rescoring saved predictions against those proposals gives 91/116 (78.4%) top-1
for the selected pipeline. This changes both labels and denominator; it is a
separate sensitivity diagnostic, not the primary improvement claim or trusted
ground truth. See [audit rescore](audit_rescore.json).

There are still [27 original-label top-1 failures](remaining_failures.json).
Examples include notebook computers versus computers, pet food versus general
food, camera lenses versus broader optics, and broad or related-product answers.
Some failures reflect uncertain draft labels; others clearly need better ranking.

The five user inputs currently produce:

| Input | First category | Status |
|---|---|---|
| Clothing | 1604 — Clothing | Matched canonical label |
| Tops | 212 — Shirts & Tops | Matched user-approved alias |
| Hoodies & Sweatshirts\|Clothing | 1604 — Clothing | Provisional; still too broad |
| Smart Phone | 267 — Mobile Phones | Provisional |
| Eco Friendly\|Clothing | 1604 — Clothing | Provisional attribute/context fallback |

The hoodie query no longer selects baby tops, but retrieval-only ranks Shirts &
Tops first while the selected model pushes it to fifth. This is a recorded
regression on a user example and remains required work before qualification.
Aggregate development improvement does not resolve every input.
See [full user results](user_examples.json).

## Calibration and resources

The newly fitted draft policy accepts 67/140 resolvable calibration proposals
(47.9% coverage), with no errors on those same fitted, unreviewed examples.
Only one acceptance uses the model path; the rest use deterministic shortcuts.
The one-sided 95% precision lower bound is 95.6%. The coverage target and independent
qualification gate fail. This policy supports isolated previews only.
Calibration/test splits were not used to select the ranking model or fusion
weights. Test proposals remain unscored; independently collected reviewed inputs
are still required. See [policy](draft_policy.json).

The [resource run](resource_report.json) measures 200 serial, warm requests that
actually execute the full model path. It excludes shortcut requests from its
sampling pool. Results on this machine:

- p50 730 ms, p95 779 ms, maximum 1,004 ms; 1.37 serial requests/second.
- Process peak resident memory 1.18 GiB; model load/warmup 5.74 seconds, untimed.
- System swapout counter increased by 21,152 pages. Other applications can
  contribute to this system-wide observation; this report cannot attribute it
  solely to the matcher. The no-observed-swap target **fails**, so the combined
  resource gate fails despite passing latency and process-memory targets.

The old 77 ms retrieval-only benchmark is a different pipeline. This run does not
establish concurrent capacity. Further memory profiling and a recorded repeat
under controlled system conditions are required before resource qualification.

## Verification and next gate

All 63 tests pass. Ruff lint and format checks pass, the frozen offline environment
sync succeeds, and the three original data splits have no detected conflicts.
All 600 split records still await human review.

The real artifact passes 16,746 canonical round trips and 16 independent scoped
SQLite/NumPy comparisons. It contains the unchanged 5,582 source categories and
11,164 normalized vectors. The original taxonomy checksum is preserved.

The next work is to review labels, collect independent inputs, fix the remaining
specificity/related-product errors and hoodie regression, and resolve the swap
observation. After that, recalibrate and qualify a frozen pipeline. Sprint 3 API
work remains deferred until the quality and resource gates pass. See the
[machine-readable decision](proof_of_concept_decision.json).
