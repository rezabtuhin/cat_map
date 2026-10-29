# Sprint 3 — Local development API

Date: 26 October 2026. Google artifact: `eed281a45624da69-a0a29f98cc37`.

## Scope and qualification

The user authorized Sprint 3 implementation while deferring human review and
further matching tests. This advances development integration ahead of the
original Sprint 2 API entry gate. It does not qualify matching accuracy or change
the earlier resource decision. Float32 and the existing hybrid/NLI ranking profile
remain selected; no matching rules, aliases, model revisions, or policy thresholds
were changed for this sprint.

The default API does not load a decision policy. Only canonical lookups, approved
aliases, and the matcher's existing conservative literal general-category lookups
can assign a category. All other outcomes keep `category: null`. Alternatives are
suggestions requiring review. Draft or unqualified policies fail readiness.

## Implementation

| Work item | Implemented behavior | Evidence |
|---|---|---|
| S3-01 | Strict request/response models; uncertain statuses cannot carry assignments | `src/category_matcher/api/schemas.py`, `tests/test_api.py` |
| S3-02 | Match, canonical lookup, and Google taxonomy discovery routes | `app.py`, [OpenAPI snapshot](openapi.json) |
| S3-03 | Background startup validates the bundle and warms resident models before readiness; failure remains unready | `service.py`, lifecycle tests |
| S3-04 | Dedicated owner thread initializes, runs, and closes the matcher/SQLite connection; canonical GET uses an immutable snapshot | Thread-ownership and responsiveness tests |
| S3-05 | One active inference, eight pending by default, 503 overload, bounded body and request deadline | Queue, timeout, cancellation, and overload tests; [HTTP concurrency report](concurrency_report.json) |
| S3-06 | UUID event IDs, `X-Request-ID`, JSON startup/HTTP/matching logs, queue/inference/total timings | Log assertions, response timings, `server.log` |
| S3-07 | API tests, runnable caller, documented startup and configuration | `tests/test_api.py`, `scripts/api_client.py`, [README](../../README.md#run-the-local-api) |

The application package is version `0.3.0`. New dependencies are FastAPI,
Uvicorn, Pydantic v2, and HTTPX for development checks. The existing artifact loads
without rebuilding: quality-affecting implementation and model-library versions
remain compatible. API source checksums and installed package versions are in
[environment.json](environment.json).

Routes:

- `POST /v1/categories/match`
- `GET /v1/taxonomies`
- `GET /v1/taxonomies/google/categories/{category_id}`
- `GET /health/live`
- `GET /health/ready`

The launcher binds to loopback with one worker. Models stay resident. Inference
is kept off the event loop, and queue admission does not put unbounded jobs into
the executor. A timed-out or cancelled application request cannot release the
active inference slot before its native model call finishes. Queued application
cancellations remove their jobs. Shutdown drains active work and closes SQLite
on the owner thread. The 30-second default deadline covers body reading, queue
waiting, and inference; a native call cannot be forcibly interrupted by it.

## CLI equivalence and supplied examples

[cli_equivalence.json](cli_equivalence.json) compares all 200 frozen development
inputs against the existing CLI report, using the same artifact and dataset
checksum. All 200 statuses, accepted IDs, methods, reason codes, and displayed
alternatives agree. This is behavior equivalence against prior predictions;
unreviewed labels do not establish accuracy. No test-set predictions were run.

[user_examples.json](user_examples.json) records the ten supplied category strings:

| Input | API status | Assigned ID |
|---|---|---|
| Clothing | `matched` | `1604` |
| Tops | `matched` | `212` |
| Hoodies & Sweatshirts\|Clothing | `provisional` | None |
| Smart Phone | `provisional` | None |
| Eco Friendly\|Clothing | `provisional` | None |
| Jackets | `provisional` | None |
| Collections | `needs_clarification` | None |
| Fitness Equipment | `provisional` | None |
| New Luma Yoga Collection | `needs_clarification` | None |
| Erin Recommends | `needs_clarification` | None |

The `no_match` response contract is exercised with a controlled backend fixture.
The default unqualified pipeline does not claim calibrated out-of-taxonomy
rejection. Invalid input, overload, and inference failures use HTTP error envelopes
and cannot supply an assigned category.

## Real HTTP measurements

The benchmark started a real Uvicorn server on an ephemeral loopback port on the
existing M1 Mac. It checked CLI equivalence and user inputs first, then measured
200 serial warm requests drawn from 137 actual model-path inputs. Canonical and
general-category shortcuts were excluded from the timed set. Models remained
resident in the server, with no model process running in the client. API source
hashes were checked before and after measurement.

| Measurement | Observed result |
|---|---|
| Startup through readiness, including process start | 8.92 seconds; initial readiness returned 503 |
| Warm HTTP latency p50 / p95 / maximum | 798.9 / 1,055.9 / 1,318.8 ms |
| Matching duration p95 | 1,052.3 ms |
| Queue wait p95 during serial run | 0.107 ms |
| HTTP/framework overhead p50 / p95 | 1.93 / 6.48 ms |
| Process peak resident memory | 1,263,697,920 bytes, about 1.18 GiB |
| Process physical footprint before / after serial run | 1,125,225,536 / 1,125,684,288 bytes |
| Process compressed memory after serial run | 811,253,760 bytes |
| System-wide swapout increase during serial run | 63,064 pages; not attributable to this process alone |
| Ten-request burst | Nine HTTP 200 responses; one HTTP 503 `queue_full` |
| Maximum observed active / pending requests | 1 / 8 |
| Liveness / readiness p95 during burst | 4.04 / 2.79 ms; all probes HTTP 200 |

Sources: [resource report](resource_report.json) and
[concurrency report](concurrency_report.json). Overhead is measured client wall
time minus server queue and matching duration; it includes local HTTP, validation,
and serialization. Memory comes from native macOS counters in the server process.
This warm execution check establishes neither production capacity nor matching
accuracy. It does not isolate service-induced swap from other applications and
cannot pass the outstanding swap gate.

The benchmark stopped its child server after measurement. The log records
application shutdown completion; Uvicorn subsequently exits via the requested
SIGTERM (`server_exit_code: -15`). Python's resource tracker emitted a warning
about one semaphore being cleaned up at process exit. No inference error or
readiness failure occurred during the measured requests. Application lifecycle
tests separately verify that active work drains and the SQLite owner closes.

## Verification

All 129 tests pass, including 27 API cases. The five warnings are the existing
PyTorch quantization deprecation warnings. Ruff lint and formatting checks pass,
as does `git diff --check`. The installed `category-matcher-api --help` launcher
works, and the frozen offline dependency sync succeeded after installation.

API tests cover strict validation, all five semantic statuses, canonical lookup,
thread ownership, nonblocking startup, failed startup, rejection of draft policy,
bounded admission, health/lookup responsiveness, body limits, deadlines, application
task cancellation, shutdown, and failure without assignment on backend errors,
invalid canonical output, or status/category inconsistency.

[verification.json](verification.json) records the checks. The original taxonomy
and development/calibration/test checksums are unchanged. All recorded category
and alternative IDs, labels, and paths agree with the source taxonomy. Artifact
integrity passes, and the API files still match the benchmarked hashes. The lock
adds API dependencies and updates this package to `0.3.0`; existing model-library
versions are unchanged. All 405 review decisions remain pending. The unreviewed
test holdout was not scored.

Sprint 3's local integration deliverables are complete. This is separate from
passing the pending Sprint 2 and release qualification gates.

## Remaining work

- All 405 prepared review decisions remain pending; no new approvals were applied.
- Development/calibration labels and an independent reviewed holdout are still
  needed before semantic acceptance can be qualified.
- The earlier ranking/coverage deficits and unpassed resource swap gate remain
  open. API agreement does not resolve them.
- Feedback capture, migration/recovery workflows, and pilot reproducibility are
  Sprint 4 work. Additional taxonomies, multilingual inference, public hosting,
  authentication, and distributed scaling remain outside this sprint.

Start locally with `uv run category-matcher-api`, then use `/docs` or
`uv run python scripts/api_client.py 'Smart Phone'`. See README for the complete
request contract and settings. The HTTP benchmark starts and stops its own server;
it does not leave a service running.
