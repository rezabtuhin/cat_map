# Sprint 2 implementation status

**Current update:** see [the improvement findings](improvement/FINDINGS.md) and
[updated decision](improvement/proof_of_concept_decision.json). The selected
default now uses NLI rank fusion, reaching 77.5% draft non-exact top-1 versus
58.3% previously. Quality qualification is pending; the latest full-model run
passes latency/process-memory targets but fails the no-observed-swap target.
Automatic semantic acceptance remains disabled. The snapshot below describes
the earlier retrieval-only default and is retained as historical evidence.

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
