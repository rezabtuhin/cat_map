# Architecture decisions

These decisions describe the implemented design, not a proposal for a larger platform.

| ID | Decision | Reason and consequence | Implementation |
|---|---|---|---|
| ADR-01 | One modular Python application with HTTP and CLI entry points | Both interfaces use one matching engine; changing behavior requires no duplicate implementation. | `matcher.py`, `cli.py`, `api/` |
| ADR-02 | Return canonical taxonomy records | IDs and paths come from a validated bundle; clients can resolve the selected category consistently. | `domain.py`, `taxonomy.py`, `storage.py` |
| ADR-03 | Combine exact, lexical, fuzzy, embedding, and NLI evidence | Exact names are cheap; approximate inputs need complementary retrieval signals and bounded reranking. Scores alone do not authorize acceptance. | `normalization.py`, `retrieval.py`, `encoder.py`, `reranker.py` |
| ADR-04 | Separate ranking from acceptance | An attractive candidate can still be wrong; accepted assignments and reviewable suggestions have different response fields. Semantic acceptance needs independent qualification. | `decision.py`, `api/schemas.py` |
| ADR-05 | Treat store ancestry as contextual evidence | Store merchandising hierarchies are not Google taxonomy trees; only an explicit canonical parent ID creates that hard constraint. | `query.py`, `matcher.py` |
| ADR-06 | Use immutable SQLite taxonomy bundles and separate mutable runtime state | Requests cannot mutate category/vector artifacts; runtime WAL writes, migrations, and backups have an independent lifecycle. | `storage.py`, `runtime.py`, `migrations.py` |
| ADR-07 | Pin source identity, implementation, dependencies, models, and policy | Code/data/model drift can change predictions; incompatible or modified bundles fail validation rather than mixing versions. Rebuild when affected identities change. | `artifacts.py`, `bundles.py` |
| ADR-08 | Process one inference job at a time with a bounded queue | Keeps SQLite ownership and resident-model memory predictable; throughput is limited to one active inference and overload returns 503. | `api/service.py` |
| ADR-09 | Keep inference local and model acquisition explicit | Runtime uses downloaded CPU snapshots; preparing a new machine requires acquiring pinned weights and rebuilding installation-specific artifacts. | `scripts/prepare_models.py`, `encoder.py`, `reranker.py` |
| ADR-10 | Separate draft evaluation from reviewed holdout qualification | AI-authored labels and inspected real examples are useful development data but cannot establish independent accuracy; draft policies stay previews. | `datasets.py`, `evaluation.py`, `calibration.py`, `review.py` |
| ADR-11 | Use atomic activation plus an API-held file lease | A resident matcher must not serve one release while activation selects another; API downtime is required for switching releases. Locking uses POSIX `fcntl`. | `bundles.py`, `api/service.py` |
| ADR-12 | Retire feedback capture and promotion | The feature was removed at the user's request; old migrations retain checksum compatibility, while matching, event logging, and dataset review continue. | `api/app.py`, `runtime.py`, `migrations.py` |
| ADR-13 | Provide one isolated Docker development service | A locked Linux CPU environment avoids host Python/SQLite setup; editable source mounts and persistent volumes retain development state. First startup prepares models and a release; later identity changes require explicit rebuilding. Host publishing remains loopback-only. | `Dockerfile`, `compose.yaml`, `docker/entrypoint.py`, `doc/DOCKER.md` |

## Constraints to preserve

- Exact normalization is conservative; retrieval normalization can be broader.
  Do not turn retrieval synonyms or spelling guesses into approved aliases implicitly.
- A placeholder/default store label cannot justify a product category by itself.
  Source-derived selections remain provisional and conflicts remain explicit.
- Runtime persistence must stay outside `artifacts/`. Never mutate active bundle
  files, release descriptors, model snapshots, or migration checksums in place.
- HTTP timeouts bound waiting; they cannot interrupt a native PyTorch call already
  running. Releasing its execution slot early would violate the single-owner design.
- API defaults use the artifact's ranking profile and a release-pinned qualified
  policy when present. Experimental CLI overrides must not silently alter API behavior.

## Deferred architecture

The current executable routes and parser support Google English. Multi-vendor
support needs explicit ingestion adapters, vendor/version selection, and identity
isolation across storage, lookup, aliasing, policy, and evaluation. A Meta text file
cannot simply be dropped into the Google build and safely become another vendor.

Multi-process inference duplicates model memory; multi-host service introduces
release distribution and writable-event ownership concerns. These require measured
capacity and an explicit design change. No admin dashboard, authentication layer,
or public-network deployment is implemented in this repository.

Future implementation stages and scaling gates are recorded in
[limitations and roadmap](LIMITATIONS_AND_ROADMAP.md); cross-cutting development
rules are in [AGENTS.md](../AGENTS.md) and [security guidance](SECURITY.md).

## Changing a decision

Record the concrete requirement, source modules affected, compatibility/rebuild
impact, and verification evidence in the relevant decision entry. Update the
architecture diagrams and API contract if their observable behavior changes.
Historical root plans remain references for prior scope, not current feature promises.
