# Sprint 2 implementation status

**Integration update:** the user authorized Sprint 3 local development API work
while deferring review. See [Sprint 3 findings](../sprint_3/FINDINGS.md). This
changes the development order; Sprint 2 qualification and the resource swap gate
remain pending, and automatic semantic acceptance remains disabled.

**Current update:** see [the latest findings](revision_3/FINDINGS.md) and
[updated decision](revision_3/proof_of_concept_decision.json). The current
pipeline reaches 80.0% draft non-exact top-1 versus 78.3% in the previous
revision and 58.3% originally. Store-section titles remain unmapped, literal
group evidence stays provisional, and model disagreements require clarification.
Draft calibration coverage is 52.1%, below target. Native memory profiling finds
a stable process footprint with substantial compression; the observed system
swap check still fails. Human review and independent qualification remain pending.
Automatic semantic acceptance remains disabled. Only accepted `matched` results
may contain an assigned category; candidates and previews must not be assigned.
All 102 tests, lint/format, artifact integrity, and evidence-consistency checks pass.
There are 405 pending review decisions prepared, with no new approvals applied.

The [quantization experiment](quantization/FINDINGS.md) is complete. Float32
remains selected: full int8 regresses on ranking; embedding-only int8 preserves
development top-1 outcomes but increases process peak and physical footprint.
Neither experiment enables automatic semantic assignment or changes the active
artifact. Review and independent qualification remain pending.

The snapshot below describes the earlier retrieval-only default and is retained
as historical evidence. Intermediate findings remain in `improvement/` and `specificity/`.

**Decision: revise.** Local resource targets pass. The matching-quality gate has
not passed, and trusted calibration/test evidence is still missing. Automatic
semantic acceptance remains disabled.

Verification: 52 tests pass, Ruff checks/formatting pass, frozen offline dependency
sync succeeds, and the original taxonomy checksum is unchanged. The real artifact
also passes all 16,746 canonical round trips and 16 SQLite/NumPy numerical checks.

| Item | Status | Evidence |
|---|---|---|
| S2-01: 600 reviewed examples | Partial | 600 draft proposals prepared in isolated 200/200/200 splits; labels await review |
| S2-02: reranking comparison | Implemented and measured | MiniLM remains opt-in because it worsens aggregate development ranking |
| S2-03: hierarchy/context checks | Implemented | Pipe context, no unsupported infant apparel, broad attribute fallback; corrected phone ranking |
| S2-04: decision outcomes | Implemented with preview isolation | Matched/clarification/no-match policy rules; draft policies cannot auto-accept |
| S2-05: calibration | Diagnostic fit complete | 200 unreviewed proposals; fitted coverage is below the target |
| S2-06: untouched-test evaluation | Pending | No test predictions were run on the unreviewed proposals; independent reviewed data required |
| S2-07: local resource test | Complete for the serial default matcher | 200 warm requests; p95 latency, memory, and swap observations recorded |
| S2-08: proof-of-concept decision | Complete | Revise; resources are feasible, quality and evidence are insufficient |
| Sprint 3 | Deferred | Quality gate has not passed |

See [FINDINGS.md](FINDINGS.md), [resource_report.json](resource_report.json),
[draft_policy.json](draft_policy.json), and
[proof_of_concept_decision.json](proof_of_concept_decision.json).

The [README](../../README.md) includes the new `check-data`, `calibrate`, `qualify`,
and `benchmark` commands and draft-policy preview examples. Historical Sprint 1
reports remain in their original directory.
