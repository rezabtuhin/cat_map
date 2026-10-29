# Developer handoff

This guide describes the current source tree after removal of the feedback feature.
It is the starting point for a developer resuming implementation. Coding agents
start with [AGENTS.md](../AGENTS.md) and [WORKFLOW.md](../WORKFLOW.md).

For a fresh developer machine, use the [Docker setup](DOCKER.md) to prepare the
backend without installing Python, uv, or SQLite on the host.

## Reading order

1. [Architecture](ARCHITECTURE.md): trace a match and understand the storage boundaries.
2. [Decisions](DECISIONS.md): understand the constraints before changing behavior.
3. [API](API.md): inspect the client contract and provisional-selection rules.
4. [Operations](OPERATIONS.md): build, run, update, diagnose, and recover the service.
5. [File reference](FILE_REFERENCE.md): find the implementation or test to change.
6. [Security](SECURITY.md): preserve validation and data-handling boundaries.
7. [Limitations and roadmap](LIMITATIONS_AND_ROADMAP.md): distinguish current constraints from proposed scaling work.

## Current status

| Area | Current implementation |
|---|---|
| Taxonomy | Google English; the inspected local bundle contains 5,582 records and 11,164 embedding rows |
| Matching | Canonical/alias fast paths, scoped hybrid retrieval, local NLI reranking, and specificity safeguards |
| Serving | Loopback FastAPI service, one inference thread, bounded pending queue, persistent match events |
| Containers | Compose development service with CPU dependencies, source mounts, and separate persistent volumes |
| Review | Source-bound dataset review tools and manually maintained approved aliases |
| Feedback | Endpoint, handlers, storage accessors, and feedback-to-alias commands removed |
| Database compatibility | Historical runtime migration 2 retained; runtime schema remains version 2 |
| Accuracy | Human review and independent semantic acceptance qualification remain pending |
| Scaling | One local inference worker; multi-host deployment and additional vendors are deferred |

Counts describe the inspected bundle, not a fixed requirement for future taxonomy
updates. Use `category-matcher inspect` for the selected installation.

## Resume checklist

1. Read the working tree with `git status`; preserve changes already in progress.
2. Use `uv sync --frozen` and inspect the active release before rebuilding anything.
3. Check `/health/ready` and the startup log; confirm which artifact and policy are active.
4. Reproduce a reported category with its original `category`, `context`, and ancestry.
5. Follow the matching-stage ownership map below and make the smallest relevant change.
6. Update affected regression coverage and run checks when authorized by the task.
7. Rebuild when a fingerprinted matching file, build input, model, or pinned dependency changes.
8. Update the API/docs if the observable contract or operational procedure changes.

## Where to make changes

| Change | Start here |
|---|---|
| Input syntax, generic labels, source ancestry | `query.py`, then `matcher.py` |
| Candidate retrieval, branch weights, spelling | `retrieval.py`, `normalization.py`, `storage.py` |
| Reranking, model templates, specificity | `reranker.py`, `config/category_ranking.json`, `matcher.py` |
| Acceptance, clarification, calibration | `decision.py`, `calibration.py`, `datasets.py` |
| HTTP contract, scheduling, service settings | `api/schemas.py`, `api/app.py`, `api/service.py`, `api/settings.py` |
| Artifact identity, activation, rollback | `artifacts.py`, `bundles.py` |
| Runtime events or schema evolution | `runtime.py`, append-only `migrations.py` |
| Evaluation input collection and review | `review.py`, `data/google/reviews/README.md` |

Paths without a prefix above are relative to `src/category_matcher/`.

## Work that remains

See the [limitations, enhancements, and scaling roadmap](LIMITATIONS_AND_ROADMAP.md)
for proposed stages and completion evidence.

- Review development/calibration labels and collect an independent reviewed holdout
  before enabling semantic auto-acceptance; existing AI drafts do not meet that requirement.
- Re-measure resource use and concurrent HTTP behavior for the current release
  before claiming deployment capacity; old generated sprint reports were removed.
- Decide the vendor/version registry and ingestion adapters before adding another
  taxonomy; the current parser, routes, configuration, and matching identity use Google.
- The last full-repository lint check found two existing overlong strings in
  `matcher.py`; re-check the current tree before treating this as outstanding.

The previous user deferred automated execution and planned manual matching tests.
No new runtime-test result is implied by these documents. Existing tests remain
available for subsequent authorized verification.

## Documentation maintenance

Keep architectural diagrams aligned with source ownership and actual data flow.
Keep `FILE_REFERENCE.md` synchronized when adding, moving, or deleting maintained
files. Store generated diagnostics under `reports/`; the historical holdout access
ledger is persistent evidence and must survive diagnostic cleanup. See
[operations](OPERATIONS.md#evaluation-and-qualification).
