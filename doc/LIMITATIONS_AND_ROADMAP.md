# Limitations, enhancements, and scaling

The first section describes the implemented system. The roadmap is proposed future
work, not functionality enabled by configuration today. Complete the stages needed
for the actual workload; a local test installation does not need every future component.

## Current limitations

| Area | Limitation and practical effect | Source or evidence |
|---|---|---|
| Mapping accuracy | Reviewed calibration and independent holdout qualification remain pending; approximate rankings are not validated accuracy claims. | `data/google/`, `calibration.py`, `decision.py` |
| Product meaning | Names such as Collections or recommendations need product context; placeholder labels stay unmapped without explicit useful context. | `query.py`, `matcher.py` |
| Context safeguards | Rules address observed spelling, merchandising, ancestry, infant-apparel, and promotion failures; they are not a complete understanding of every product qualifier. | `query.py`, related regression tests |
| Vendor/language coverage | Parser, executable routing, and matching identity support Google English; another vendor is not a drop-in text-file addition. | `taxonomy.py`, `api/schemas.py`, `matcher.py` |
| Inference throughput | One active inference and eight pending requests by default; queue size is configurable, but more queue capacity does not add compute capacity. | `api/service.py`, `api/settings.py` |
| Resources | BGE embeddings and NLI reranking require resident CPU model memory; adding processes duplicates model/runtime state. | `encoder.py`, `reranker.py`, `api/server.py` |
| Search size | Cosine distances are computed in SQL over scoped label/path vectors; there is no approximate nearest-neighbor index in the current implementation. | `storage.py` |
| Caching/bulk work | Ancestor lookup caches exist, but there is no general result cache, batch-match HTTP contract, or durable bulk-job system. | `matcher.py`, `api/app.py` |
| Timeouts | HTTP deadlines cannot cancel an active native model call; its slot remains occupied until completion. | `api/service.py` |
| Updates | Taxonomy/model/matching identity changes require a compatible bundle; the API must stop for activation/rollback. | `artifacts.py`, `bundles.py` |
| Portability | Model metadata contains local absolute snapshot paths and leases use POSIX `fcntl`; copied bundles need installation compatibility. | `encoder.py`, `bundles.py`, model metadata |
| Persistence/analytics | Successful requests are retained locally without a built-in retention scheduler, aggregation dashboard, or multi-host event service. | `runtime.py`, `api/service.py` |
| Network/security | Launcher permits loopback only; application auth, tenant controls, and a public admin API are not implemented. | `api/server.py`, [security guide](SECURITY.md) |
| Validation evidence | Historical generated reports were removed; benchmark and regression results must be reproduced for the release being assessed. | [handoff](README.md), [operations](OPERATIONS.md) |

A specific taxonomy result may be absent, ambiguous, or unsupported by available
context. Returning a flagged suggestion or clarification is expected behavior;
raising acceptance thresholds or adding a model does not create missing product facts.

## Enhancement roadmap

| Priority | Work | Completion evidence |
|---|---|---|
| P0: establish quality | Review labels, expand real ambiguous/negative/hierarchy cases, and collect an independent holdout; qualify semantic acceptance only when gates pass. | Reviewed provenance, slice-level errors, accepted precision/coverage, and qualification tied to the release |
| P1: make performance measurable | Re-run current-release resource/HTTP benchmarks and add aggregate observability and event retention design. | Cold/warm timing, queue wait, memory, error/timeout rates, and status proportions by release |
| P1: improve local efficiency | Evaluate bounded result caching, reduced rerank budgets, smaller/distilled models, or quantization independently. | Lower resource use with unchanged or reviewed quality/regression evidence; no assumed int8 benefit |
| P2: improve data lifecycle | Add background build jobs, taxonomy diffs, reusable unchanged embeddings, portable manifests, and controlled activation status. | Reproducible import/build/rollback and visible failures without mutating the selected bundle |
| P2: support operators | Build an authenticated admin UI for usage metrics, taxonomy browsing/import, release management, and existing dataset review. | Authorized actions, audit records, bounded uploads/jobs, and clear draft-versus-approved state |
| P3: support vendors | Introduce a vendor/version registry, parsers/adapters, scoped aliases/lookup/policies, and per-vendor evaluation. | Overlapping Google/Meta IDs remain isolated; unknown vendors never fall back to Google |
| P3: support concurrent stores | Add measured replication, centralized operational state when required, and store-aware access/quotas. | Load/overload/recovery checks, isolation checks, consistent release selection, and explicit capacity targets |

Priorities describe dependencies, not a requirement to block independent product
work while human review is deferred. Keep semantic auto-acceptance disabled until
its quality gates are met. Removed feedback capture/promotion is not part of this
roadmap unless the user explicitly requests it again.

## Scaling path

### Stage 1: measure the current single instance

Measure separately for canonical fast paths, contextual inheritance, and actual
neural inference. Capture p50/p95/p99 latency, queue wait, model service time, cold
startup, process memory, 503/504 rates, and the input/status mix. Retain the artifact,
model, configuration, and dataset identities with each comparison.

For a serial model workload with mean service time `S` seconds, the ideal service
ceiling is approximately `1 / S` model requests per second before other overhead.
Arrival rate near that ceiling leaves little room for bursts. This is a planning
estimate, not measured capacity; use workload-specific latency and memory evidence.

### Stage 2: improve efficiency without changing deployment topology

- Cache bounded, immutable matching payloads using a key that includes release/
  policy identity and every meaningful input: category, context, ordered ancestry,
  canonical ID/parent scope, and language. Include output options if caching a
  serialized response. Preserve exact semantics rather than using a lossy global key.
- Generate a fresh request ID/timings and event record for each call; a cached
  result must not reuse another request's identity or bypass accounting.
- Reduce candidate/rerank budgets only after checking recall and difficult slices.
  Compare smaller models, quantization, or batching under the same quality gates.
- If batch inference is added, keep model scheduling and SQLite ownership explicit;
  do not introduce concurrent access to one matcher/connection as a shortcut.

A larger queue smooths short bursts but increases wait time. Respect the HTTP
budget and reject excess work predictably rather than allowing unbounded memory growth.

### Stage 3: replicate isolated workers when one instance reaches capacity

Use supervised processes/replicas with independent matchers, model instances,
connections, and bounded queues. Each gets a local immutable bundle and verified
model files. Route traffic only to ready workers and make overload/backpressure
visible to the routing layer. Provision memory from measured per-worker footprint.

The current launcher deliberately uses one loopback worker. A remote fleet needs
an implemented private deployment interface or colocated proxy, protected transport,
and authentication/authorization as appropriate. Merely increasing a worker count
is not a complete deployment design or a guarantee of linear throughput.

Keep the release identity consistent across traffic-serving workers. During rollout,
prepare and warm a new worker, validate its release, route compatible requests to
it, and drain the previous worker. This avoids editing resident bundles and provides
a route to reduced-downtime updates. Fleet routing/rollout is not implemented today.

### Stage 4: separate writable operational state when hosts must share it

SQLite WAL requires processes sharing a database to be on the same host and does
not support WAL over a network filesystem. Do not mount a writable runtime SQLite
file across hosts. See the [SQLite WAL documentation](https://www.sqlite.org/wal.html).

Proposed design: keep taxonomy bundles local/read-only, and introduce one authoritative
event writer/service for shared operational records. If cross-host queries or write
contention require a shared transactional database, migrate operational events/jobs/
users to a suitable managed or self-hosted transactional store through an explicit
repository interface. Preserve event identities, retention, backup/restore, and tenant
boundaries. A vector database is not required merely because the API has replicas.

For substantially larger taxonomies, benchmark filtered exact vector search first.
Adopt an indexed/approximate retrieval store only when measured search cost justifies
it and vendor/version filters and candidate recall can be preserved.

### Stage 5: separate management/build work from serving

Use a management API and authenticated UI to submit validated taxonomy uploads and
build jobs. An isolated builder validates format, computes source changes, builds
vectors/indexes, evaluates the result, and publishes an immutable candidate release.
The serving path only reads selected verified releases. Track build job state,
resource limits, failure details, and operator actions independently of match queues.

For a previously supported vendor format, uploading a new list should become a
data-and-release operation rather than an application deployment. A new vendor
format may still require a parser/adapter implementation and release once. Define
activation/rollback policy separately from upload success; do not let an uploaded
file select model code or silently auto-activate an unevaluated release.

## Acceptance gates for future changes

| Change | Gate before relying on it |
|---|---|
| Cache | Scoped/versioned keys, bounded memory, correct request/event accounting, and no stale policy decisions |
| New model/profile | Rebuilt identity, canonical/provisional invariants, difficult-slice quality, and measured memory/latency |
| Extra worker | No shared connection/model mutation, bounded queues, ready/drain behavior, and load/recovery evidence |
| New vendor | Vendor/version isolation across IDs, aliases, ancestry, retrieval, releases, and evaluation |
| Admin import/build | Authenticated authorized action, content/size/path validation, isolated job budgets, and safe publication |
| Central events | Transactional durability, retention/privacy policy, backup recovery, and store isolation |

Update [architecture decisions](DECISIONS.md) and [operations](OPERATIONS.md) when a
stage is implemented. Keep future controls distinct from current guarantees.
