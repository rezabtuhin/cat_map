# Sprint 1 findings

**Outcome: executable prototype; quality not qualified.**

The evaluation labels are AI-authored development drafts awaiting human review. These numbers diagnose retrieval behavior; they do not establish real-world accuracy.

Artifact: `eed281a45624da69-ecbf2e1335a8`.
Dataset SHA-256: `2821da626f6d9fb302279e6e8df16f1979f36715ab6cb5fda358e31cff112d45`.

## Integrity verification

- 5,582 canonical categories.
- 16,746 successful ID/label/path checks.
- 11,164 normalized embeddings in SQLite.
- 16 SQLite/NumPy numerical comparisons.

## Non-exact retrieval diagnostics

Exact shortcuts are excluded from this table. There are 120 draft resolvable non-exact examples. Recall at 20 uses the same candidate budget; final shortlist sizes can vary by method and parent expansion.

| Method | Top-1 correct | Top-3 recall | Recall at 20 | Final shortlist recall |
|---|---:|---:|---:|---:|
| exact | 0/120 (0.0%) | 0.0% | 0.0% | 0.0% |
| fuzzy | 66/120 (55.0%) | 64.2% | 71.7% | 75.8% |
| bm25 | 32/120 (26.7%) | 35.8% | 54.2% | 60.8% |
| semantic | 59/120 (49.2%) | 75.8% | 87.5% | 89.2% |
| hybrid | 65/120 (54.2%) | 66.7% | 87.5% | 95.8% |
| hybrid_rerank | 41/120 (34.2%) | 55.8% | 82.5% | 95.8% |

Hybrid retrieval improves candidate coverage over the simpler branches, but its first result is not reliably correct. The selected MiniLM passage reranker reduces top-1 accuracy on this draft set and remains disabled by default. Neither pipeline meets the planned production target.

## User-provided inputs

| Input | Default retrieval first result | Experimental reranked first result | Status |
|---|---|---|---|
| Clothing | 1604 — Clothing | 1604 — Clothing | matched |
| Tops | 212 — Shirts & Tops | 212 — Shirts & Tops | matched |
| Hoodies & Sweatshirts\|Clothing | 212 — Shirts & Tops | 5552 — Hunting & Tactical Pants | provisional |
| Smart Phone | 264 — Mobile Phone Accessories | 267 — Mobile Phones | provisional |
| Eco Friendly\|Clothing | 1604 — Clothing | 1604 — Clothing | provisional |

Clothing and Tops were confirmed by the user. Tops now has an approved alias; it is a training/regression example, not a held-out success. Positive labels for the other three inputs are drafts. The user explicitly rejected baby/toddler tops for the hoodie query.

The default pipeline fixes that reported demographic error through pipe-context scoping and an unsupported-infant-apparel filter. Resolved parent words are not repeated as retrieval query terms, which previously overwhelmed the leaf concept. There is no hoodie/sweatshirt-specific entry in the supplied taxonomy.

The phone example remains an important failure: default hybrid retrieval ranks accessories above phones, while the optional reranker corrects this example but hurts the broader draft benchmark. Do not select a different method per example and claim that cherry-picked combination as one evaluated pipeline.

## Runtime observations

- Process peak resident memory: 795.7 MiB.
- Model loading and warmup: 4.38 seconds.
- Five inputs do not establish p95 latency, capacity, or the Sprint 2 resource gate.
- The local reranker initially crashed in CPU linear inference. Its checkpoint parameters included addresses offset by two bytes. Copying misaligned parameters into aligned PyTorch storage resolved the observed failure; value preservation has a regression test.

## Next work

1. Review/correct development labels, preserving real user examples separately.
2. Investigate categories missing from the shortlist before tuning acceptance.
3. Improve ranking on phone-versus-accessory and unsupported-specialization errors; reconsider the generic MiniLM reranker using development comparisons.
4. Collect distinct calibration and untouched-test inputs, calibrate abstention, and measure serial end-to-end inference before the Sprint 2 gate.
5. Keep API and larger architecture expansion deferred until the quality gate passes.
