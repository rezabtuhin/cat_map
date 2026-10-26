# Sprint 2 findings

**Decision: revise. The local runtime target passes; matching quality is not qualified.**

All accuracy figures below use unreviewed AI-authored labels. Development figures informed implementation choices; calibration figures are fitted on the same draft data used to choose thresholds. Neither is independent accuracy evidence.

Active artifact: `eed281a45624da69-8f86c745339e`.

## Retrieval changes and development comparison

BM25 now indexes category labels with retrieval-only plural folding and content-word normalization. BGE uses plain queries. Fusion weights semantic ranks by five and lexical/fuzzy ranks by one, with rank constant ten. Candidate limits are 100/100/50, with up to 200 fused results plus five parents. These are development-selected settings.

| Method | Sprint 1 top-1 | Sprint 2 top-1 | Sprint 2 top-3 | Sprint 2 recall at 20 | Sprint 2 final recall |
|---|---:|---:|---:|---:|---:|
| fuzzy | 55.0% | 57.5% | 65.0% | 79.2% | 86.7% |
| bm25 | 26.7% | 45.8% | 55.0% | 70.8% | 73.3% |
| semantic | 49.2% | 56.7% | 75.8% | 88.3% | 94.2% |
| hybrid | 54.2% | 58.3% | 76.7% | 90.8% | 98.3% |

The same 120 resolvable non-exact development examples are compared. Wider shortlist limits contribute to final-recall gains; recall at 20 keeps a common budget.

Optional MiniLM reranking reaches 34.2% top-1 on these examples and remains disabled by default.

The two absent draft targets are `adult oriented products` → Mature and `Tos` → Toys. Both require label/ambiguity review; a three-letter input can be underspecified. No alias was added merely to force these benchmark cases to pass.

## User inputs under the default matcher

| Input | First category | Outcome |
|---|---|---|
| Clothing | 1604 — Clothing | matched |
| Tops | 212 — Shirts & Tops | matched |
| Hoodies & Sweatshirts\|Clothing | 212 — Shirts & Tops | provisional |
| Smart Phone | 267 — Mobile Phones | provisional |
| Eco Friendly\|Clothing | 1604 — Clothing | provisional |

Only Clothing and Tops have user-confirmed positive labels. The remaining positive labels are proposals; the user explicitly rejected baby tops for hoodies. The hoodie result excludes unsupported baby clothing. Eco Friendly keeps the broad Clothing context. Model-based candidates remain provisional.

## Policy fit and data provenance

There are 200 development, 200 calibration proposals, and 200 test proposals. Cross-split IDs, query keys, declared families, and category targets do not overlap. All 600 labels still await human review. Test proposals have not been scored.

The fitted draft policy would accept 56/140 resolvable calibration cases (40.0% coverage, target 80%). It has 0 fitted-label errors among those acceptances. This is a calibration fit on draft labels, not a precision claim. Its one-sided 95% lower precision bound is 94.8%, before considering the label-quality and data-reuse limitations.

The implemented policy can preview matched, clarification, and no-match outcomes. `--experimental-policy` preserves provisional status and puts its proposed decision in `decision_preview`. `--policy` rejects an unqualified policy. Qualification also checks review status, independent holdout provenance, split isolation, minimum data sizes, recall, coverage, precision, and non-exact evidence. It records holdout access to prevent evaluating changed policies on previously inspected inputs.

## Local performance

- Serial timed requests: 200 over 156 distinct available non-exact development inputs.
- Warm end-to-end p95: 77.0 ms.
- Process peak resident memory: 705.7 MiB.
- Encoder load/warmup: 3.14 seconds.
- Observed system swapout page change: 0.
- This run uses default retrieval without reranking or a decision policy; it is not a production concurrency test.

## Acceptance and next work

Sprint 2's implementation and diagnostic runs are delivered. Full acceptance is pending trusted labels and a passing independent quality gate. The required next work is label adjudication and further ranking improvement. API integration remains deferred. See `label_review_queue.jsonl` and `proof_of_concept_decision.json` for the actionable records.
