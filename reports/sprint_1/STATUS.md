# Implementation status

Sprint 1's executable components are implemented. Acceptance of Sprint 1 remains
pending human review of its 200 development labels; the benchmark is explicitly
diagnostic. The first matching-quality gate has not passed.

| Item | Status | Evidence |
|---|---|---|
| S1-01: isolated environment | Implemented and verified | Python 3.11, `uv.lock`, frozen offline sync |
| S1-02: sqlite-vec integration | Implemented and verified | SQLite 3.53.1 / sqlite-vec 0.1.9; vector tests |
| S1-03: taxonomy ingestion | Implemented and verified | 5,582 categories; source audit and parent checks |
| S1-04: SQLite and manifests | Implemented and verified | Versioned snapshots, checksums, source/model/code hashes |
| S1-05: exact lookup and aliases | Implemented | User-confirmed Tops → 212 approved; three other draft aliases inactive |
| S1-06: lexical and fuzzy retrieval | Implemented and evaluated diagnostically | BM25S and RapidFuzz baselines |
| S1-07: embeddings | Implemented and built | 11,164 normalized 768-dimensional BGE vectors |
| S1-08: semantic retrieval and fusion | Implemented and verified | sqlite-vec/NumPy agreement; scoped retrieval and RRF |
| S1-09: CLI and evaluation harness | Implemented and exercised | Real taxonomy verification and draft comparison reports |
| S1-10: trusted development labels | Partial | 200 AI-authored drafts plus five separately stored user inputs; full review outstanding |
| Early S2-02 experiment | Implemented, opt-in | MiniLM reranking improves one user example but worsens aggregate draft ranking |
| Remaining Sprint 2 work | Pending | Calibration/test data, calibrated decision policy, qualification and resource evidence |
| Sprints 3–5 | Deferred | No API, production release, or deployment |

See [FINDINGS.md](FINDINGS.md) for measured results and unresolved failures.
The [README](../../README.md) contains commands to try the prototype.

The user's `label|parent` examples led to explicit context parsing and a narrow
guard against inferring baby/toddler clothing without an infant-related cue.
`Eco Friendly|Clothing` produces Clothing as a provisional broad candidate. These
rules do not create an unmeasured guarantee for other demographic qualifiers.

All non-exact/model-based outputs remain provisional. There is no calibrated
semantic acceptance policy, and neither the draft results nor confirmed alias
regressions establish production accuracy.
