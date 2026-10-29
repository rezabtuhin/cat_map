# Multi-Vendor Category Matching: Architecture, System Audit, and Implementation Plan

Historical planning document. See [README.md](README.md) for current usage.
Generated sprint outputs were removed during cleanup; earlier evidence remains in Git history.
The feedback API and feedback-to-alias workflow described below have since been removed.

Prepared: 26 October 2026 (Asia/Dhaka)

Project: `/Users/linno/Documents/code/RAG`

## 1. Implementation decision

Build a **local Python service that selects a category from the supplied taxonomy using exact lookup, verified aliases, hybrid retrieval, cross-encoder reranking, and an explicit acceptance policy**.

The matching engine is vendor-independent. Google is the first planned source; future sources such as a vendor-supplied `meta_category_list` use the same engine through source adapters and taxonomy-scoped artifacts. A Meta file has not been supplied or inspected, so its format, identifiers, and hierarchy are not assumed here.

The implementation stack is Python 3.11, uv, FastAPI, Pydantic, SQLite with `sqlite-vec`, NumPy, BM25S, RapidFuzz, Sentence Transformers, and PyTorch. Use `BAAI/bge-base-en-v1.5` for embeddings and the pinned `cross-encoder/nli-deberta-v3-small` category-entailment rank-fusion profile for the current development matcher. FastAPI/Pydantic integration is implemented as a local development service in Sprint 3 (historical), following the user's authorization to advance while qualification remains pending. Run one application worker on the existing Mac initially.

Build a **modular monolith with explicit interfaces and dependency injection**. Future features, retrieval methods, models, and storage providers must be addable through focused modules with stable contracts. The initial single-process deployment is a resource setting, not a restriction on the architecture. Section 4.6 defines the required module boundaries, change workflows, and capacity-growth path.

This is the chosen architecture for this project and its observed hardware. Model quality on this taxonomy must be established through evaluation; no model can truthfully be declared the most accurate for these inputs before labeled examples exist.

The system must optimize for **correct automatic matches while maintaining useful coverage**. It must return a clarification or no match when the input does not justify a reliable decision. Requiring one category for every possible input conflicts with minimizing incorrect assignments.

### Product contract

- Every request explicitly selects a registered taxonomy, such as `google` or a future `meta` taxonomy. The selected taxonomy's active version defines the allowed output vocabulary.
- Each successful result contains exactly one existing category ID and its original full path.
- Category identity is `(taxonomy_key, taxonomy_version, category_id)`. IDs are opaque strings in the domain/API; the Google source adapter validates its numeric source IDs and preserves them as strings.
- No request may return a category, alias target, or parent reference from another taxonomy or version.
- Broad categories are valid outputs when the input supports only that level.
- The service never invents a category, rewrites an official path, or assigns an unsupported child category.
- Inputs that could mean materially different categories receive alternatives and a clarification message.
- Inputs outside the taxonomy receive no match.
- Version 1 supports English category names, English paraphrases, and English spelling errors. This scope is an implementation assumption because input-language requirements have not been supplied. Bangla and other languages require their own labeled evaluation and a multilingual model release before they are supported.
- Optional parent-category context is accepted. A parent category explicitly supplied as a constraint must be respected.

## 2. Verified project and machine audit

These findings come from inspecting the workspace, parsing the entire category file, reading local runtime metadata, and checking the hardware. They are observations, not performance estimates.

### 2.1 Project contents

At the initial inspection, the directory contained only `google_category_list.txt`; application code, environment, dependencies, tests, database, index, and Git repository were absent. No applicable `AGENTS.md` was found in the checked ancestor directories. This is the original audit, not the current inventory. The CLI, local development API, model bundles, and tests now exist; see [README](README.md) and the sprint reports for current behavior and outstanding qualification work.

### 2.2 Category data

| Check | Observed result |
|---|---|
| UTF-8 file size | 481,108 bytes |
| Parsed category records | 5,582 |
| Blank records | 0 |
| Malformed records | 0 |
| Duplicate category IDs | 0 |
| Duplicate full paths | 0 |
| Top-level categories | 21 |
| Maximum hierarchy depth | 7 |
| Terminal categories | 4,709 |
| Categories with children | 873 |
| Missing immediate parent paths | 0 |
| Ambiguous normalized terminal labels | 0 under the audited normalization below |
| Longest full path | 214 characters |
| Rows containing non-ASCII characters | 9 |

The terminal-label collision check used Unicode NFKC normalization, case folding, `&` to `and`, and whitespace collapsing. More aggressive punctuation normalization must have its own collision check.

Hierarchy depth distribution:

| Depth | Categories |
|---|---:|
| 1 | 21 |
| 2 | 192 |
| 3 | 1,349 |
| 4 | 2,198 |
| 5 | 1,377 |
| 6 | 397 |
| 7 | 48 |

File SHA-256:

```text
eed281a45624da690a1e0fcbbf64529ad331c75488816ab51d641db3d11459fa
```

There are 5,582 records even though a newline-counting utility reports 5,581: the last record has no trailing newline.

The file is structurally ready for ingestion. It does not include a taxonomy version header or provenance metadata. Google's published English taxonomy endpoint returned a `2021-09-21` version header when checked. That header alone does not establish whether the local file matches Google's published content. Preserve the supplied file as the active authority; compare and review upstream changes separately. [Google taxonomy source](https://www.google.com/basepages/producttype/taxonomy-with-ids.en-US.txt)

This file is a **product taxonomy**. Inputs such as business services or arbitrary subjects may have no suitable entry. Google's documentation describes the taxonomy as a product categorization tree. [Google product category documentation](https://support.google.com/merchants/answer/6324436?hl=en)

### 2.3 Local machine and runtime

| Item | Observed state | Implementation consequence |
|---|---|---|
| Computer | MacBook Air, Apple M1, 8 CPU cores | Run native Apple Silicon Python |
| Memory | 8 GB unified memory | Use compact models, bounded inference concurrency, and one worker |
| Operating system | macOS 26.2, arm64 | Build and verify locally on macOS |
| Available disk at inspection | Approximately 18 GiB | Keep one virtual environment and the required model artifacts |
| Default `python3` | Python 3.9.6 | Do not use it for this application |
| uv | Installed, version 0.11.26 | Use it to manage the project and lock dependencies |
| uv-managed Python | Python 3.11.15 already installed | Use the existing 3.11 runtime |
| SQLite in Python 3.11 | 3.53.1 | Available through the Python standard library |
| SQLite CLI on PATH | `/usr/bin/sqlite3`, version 3.51.0 | CLI and Python use separate SQLite runtimes |
| `sqlite-vec` | User reports an existing installation; its path/version was not established by this inspection | Use it as the vector engine and verify it in the application environment |
| Python 3.11 extension loading | Successfully enabled and disabled on an in-memory connection | The selected runtime supports loading `sqlite-vec` |
| Git executable | Available | Initialize repository tracking during implementation |
| Docker executable | Not found on PATH | Run the initial service directly with uv |

The Python 3.11 interpreter has none of the required third-party application packages installed. PyTorch 2.8.0 and NumPy 2.0.2 were visible in the default Python 3.9 environment; those installations do not supply the planned isolated Python 3.11 environment.

The `sqlite-vec` Python distribution was not found in the inspected Python 3.11 interpreter. This does not contradict the existing installation: a CLI extension or another environment is separate from this interpreter. The application must declare its own `sqlite-vec` dependency and check `SELECT vec_version()` on its actual connections. [sqlite-vec Python integration](https://alexgarcia.xyz/sqlite-vec/python.html)

No application dependencies or model weights were installed as part of this audit. Model loading, inference latency, peak memory, and available PyTorch acceleration remain unmeasured.

### 2.4 What is missing

| Missing item | Required work | Why it matters |
|---|---|---|
| Isolated application environment | Create a Python 3.11 uv project and lockfile | Reproducible installation |
| Application-level vector integration | Load and verify `sqlite-vec` in the selected Python environment | An existing machine installation does not wire the extension into application connections |
| Extension-ready architecture | Implement domain contracts, application services, adapters, and dependency injection | Future feature changes stay localized |
| Database migrations | Version schema changes for taxonomy artifacts and runtime state | Add fields and features without manual database edits |
| Taxonomy loader | Parse, validate, and build parent/child relationships | Correct IDs, paths, and hierarchy |
| Taxonomy registry and source adapters | Register vendor sources and normalize their formats into canonical records | Add a vendor without changing matching logic |
| Version manifest | Record source checksum, model revisions, and index configuration | Prevent incompatible artifacts from being mixed |
| Verified aliases | Create a reviewed synonym-to-category mapping | Reliable recurring vocabulary matches |
| Search indexes | Build exact, BM25, fuzzy, and embedding representations | Handle wording differences and errors |
| Reranking component | Score shortlisted input/category pairs | Distinguish closely related candidates |
| Decision policy | Calibrate acceptance, clarification, and no-match thresholds | Avoid forced wrong answers |
| Labeled input examples | Collect and adjudicate actual user inputs | Measure correctness instead of guessing |
| API and response schema | Implement validated endpoints and stable statuses | Integrate the feature into a consuming application |
| Feedback workflow | Record confirmations and review alias changes | Improve without turning errors into permanent rules |
| Operational measurements | Measure warm latency, memory, queueing, and failure behavior | Establish what this machine can serve |
| Product integration details | Establish calling application, expected traffic, and required languages | Validate deployment capacity and supported scope |

The largest accuracy gap is **missing labeled input/output data**, not missing GPU hardware or a vector database.

## 3. Fixed technology stack

| Layer | Selected technology | Responsibility |
|---|---|---|
| Runtime | Python 3.11 | Application and model execution |
| Environment | uv, `.venv`, `pyproject.toml`, `uv.lock` | Dependency resolution and reproducibility |
| API | FastAPI, Uvicorn, Pydantic v2 | Request validation, response contracts, service lifecycle |
| Canonical storage | SQLite through `sqlite3` | Taxonomy metadata, aliases, reviewable feedback |
| Schema evolution | Numbered SQL migrations with a migration runner | Reproducible database changes with recorded checksums |
| Module contracts | Python dataclasses and `typing.Protocol` | Keep matching rules independent of frameworks, models, and databases |
| Dependency wiring | Explicit composition root | Select concrete implementations and inject them into use cases |
| Exact lookup | In-memory Python dictionaries | Resolve canonical IDs, paths, labels, and approved aliases |
| Lexical retrieval | BM25S | Retrieve candidates by relevant words |
| Fuzzy retrieval | RapidFuzz | Recover candidates from spelling errors |
| Embedding model | `BAAI/bge-base-en-v1.5` through Sentence Transformers | Retrieve candidates with similar meaning |
| Vector storage and search | SQLite + `sqlite-vec`, normalized float32 BLOBs | Persist vectors and execute exact cosine-distance retrieval |
| Numeric conversion | NumPy | Normalize and serialize model output and validate numerical equivalence |
| Reranker | `cross-encoder/nli-deberta-v3-small` through Sentence Transformers | Category entailment ranks fused with retrieval ranks |
| Model execution | PyTorch, initially CPU float32 | Predictable baseline on the observed machine |
| Verification | pytest, HTTPX, SciPy | Functional tests, API tests, precision confidence bounds |
| Quality checks | Ruff | Formatting and static linting |

Resolve compatible package versions during environment creation, commit the generated lockfile, and use frozen installs thereafter. Pin both model repositories to immutable revisions in the artifact manifest after the first verified download.

BM25S provides local BM25 retrieval, and RapidFuzz supplies spelling-distance and fuzzy-string matching. [BM25S documentation](https://bm25s.github.io/), [RapidFuzz documentation](https://rapidfuzz.github.io/RapidFuzz/)

The embedding model produces 768-dimensional vectors. Sprint 2 development comparisons selected plain query encoding, omitting the initial retrieval instruction; stored category text also has no instruction. The active artifact records this choice. [BGE model card](https://huggingface.co/BAAI/bge-base-en-v1.5)

The CPU execution profile remains float32 after the measured
quantization comparison (historical). Full int8
regressed on development rankings, and word-embedding-only int8 did not improve
the observed process-memory peak or footprint. Experimental packed-weight byte
counts must not be treated as measurements of runtime memory.

The initial MiniLM passage reranker worsened development ranking. The current
development configuration instead uses the NLI model with category-entailment
pair formatting and rank fusion. It scores the first 100 candidates and combines
model/retrieval ranks with weights 3:1 and rank constant 10. The selected profile
and immutable revision are stored in `config/category_ranking.json` and the
artifact. Raw entailment margins and fused scores are ranking signals, not
correctness probabilities. This improves draft development ranking but is not
independently qualified. See measured findings (historical)
and the [NLI model card](https://huggingface.co/cross-encoder/nli-deberta-v3-small).

Store and search vectors through the SQLite adapter using `sqlite-vec`. Keep NumPy for model-output conversion and correctness checks. Exact vector search is appropriate for this corpus size; it does not require an additional database service. Sentence Transformers documents direct similarity search for small corpora. [Semantic search documentation](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)

## 4. Architecture and matching behavior

```text
OFFLINE BUILD
category file + approved aliases
    -> vendor source adapter + validation + taxonomy graph + normalized keys
    -> SQLite taxonomy/vector snapshot + BM25 index + manifest

ONLINE REQUEST
taxonomy key + input category + optional context
    -> resolve selected version and capture its bundle
    -> validation + normalization
    -> exact ID/path/label lookup or approved alias
    -> if unresolved: BM25 + fuzzy + semantic candidate retrieval
    -> deduplicate by category ID + reciprocal rank fusion
    -> include relevant immediate parents
    -> selected category-entailment rank fusion, or explicit retrieval-only mode
    -> ambiguity, context, and specificity checks
    -> calibrated decision
    -> matched | needs_clarification | no_match | unsupported_language
```

### 4.1 Canonical records

Each parsed category record contains:

```json
{
  "taxonomy_key": "google",
  "taxonomy_version": "<source-checksum-version>",
  "category_id": "2353",
  "label": "Mobile Phone Cases",
  "full_path": "Electronics > Communications > Telephony > Mobile Phone Accessories > Mobile Phone Cases",
  "parent_id": "264",
  "depth": 5,
  "is_leaf": true
}
```

Preserve original labels and paths separately from normalized search keys. The Google adapter derives parent IDs from complete parent paths; adapters for other sources can read explicit parent links when supplied. Never infer hierarchy from numerical ID relationships. The canonical graph must support multiple parents if a future source requires them; use a `category_edges` relation as the authority and treat the single `parent_id` in this Google example as a convenience field. Do not flatten a vendor's graph into an incorrect tree.

### 4.2 Normalization and exact lookup

Maintain two normalization levels:

1. **Conservative keys for automatic exact matching:** Unicode NFKC, case folding, whitespace collapsing, and `&`/`and` equivalence. Preserve meaningful punctuation and hierarchy boundaries.
2. **Retrieval text:** additionally standardize hyphens and punctuation for tokenization and fuzzy matching. This broader normalization never creates an automatic exact match by itself.

Do not globally remove words such as `accessories`, `replacement`, `food`, `non-prescription`, or `prescription`. Do not apply unrestricted singularization or substring-based exact matching.

Exact lookup order:

1. Explicit `category_id`, if supplied and present in the active taxonomy.
2. Complete normalized canonical path.
3. Unique normalized canonical label.
4. A reviewed, unambiguous alias valid for the supplied context and taxonomy version.

Reject inconsistent combinations of explicit ID, category text, and parent constraint. Do not silently favor one field when they contradict each other.

The current labels have no collisions under conservative normalization. The loader must still detect collisions on every rebuild and disable that fast path for any conflicting key. Exact canonical category terms identify taxonomy nodes; broad real-world phrases and aliases still need ambiguity review.

### 4.3 Candidate retrieval

Current Sprint 2 development configuration, superseding the initial Sprint 1 limits:

| Branch | Representation | Candidate count |
|---|---|---:|
| BM25 | Canonical label only; retrieval-only content-word and light plural normalization | 100 |
| Semantic | Maximum similarity over label and full-path embedding views | 100 distinct category IDs |
| Fuzzy | Normalized canonical labels and approved aliases, deduplicated by category ID | 50 |

For semantic retrieval, precompute two normalized vectors per category: one for its label and one for its full path. Score all vectors, take the maximum score per category, then select the best category IDs. The two views contribute one semantic ranking, not two independent fusion votes.

Store vector rows with an explicit category-ID/view mapping in `category_embeddings`; never assume file line numbers equal category IDs.

Use `sqlite-vec` cosine-distance functions over the persisted vectors. Group both views by category ID and take the minimum distance, equivalent to maximum cosine similarity. Apply parent-scope filtering before ranking and limiting. For the current file, two 768-dimensional float32 views require approximately 32.7 MiB of raw vector storage, excluding SQLite overhead. Model weights, Python, indexes, and inference buffers use additional memory and must be measured separately.

Use plain BGE queries in the active Sprint 2 configuration. The prefixed and plain
forms were compared on development examples; this change has not been validated
on an independent reviewed holdout. Exact-match keys retain conservative normalization.

Fuzzy matching supplies candidates; a high fuzzy score alone never authorizes a match. Drop zero-evidence BM25 hits, and exclude fuzzy hits below the configured retrieval floor. An empty branch contributes no votes.

Merge the three branch rankings using reciprocal rank fusion:

```text
rrf(category) = sum(branch_weight / (10 + rank_in_branch))
branch_weights = {semantic: 5, bm25: 1, fuzzy: 1}
```

Ranks start at 1. Deduplicate each branch before fusion within the selected taxonomy/version. Sort ties deterministically by native string category ID. Keep up to 200 fused candidates and add at most five distinct immediate parents of the top five, where absent. Prioritize the higher-ranked child's parents, then native parent ID for ties. At most 205 candidates enter optional reranking. The wider pool improves diagnostic recall at a higher potential reranking cost; serial runtime measurements and independent quality validation remain required.

Do not discard entire top-level branches before retrieval; an incorrect early branch decision would make the correct category unreachable. Apply an explicitly supplied parent constraint consistently to all branches instead.

RRF combines ranks without equating incompatible raw scores. [RRF definition](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)

### 4.4 Reranking and hierarchy handling

For each of the first 100 candidates, the selected profile constructs a pair:

```text
premise: The product category is <retrieval query>.
hypothesis: The product category is <canonical label>.
```

Batch the pairs through the cross-encoder and rank by entailment logit minus
log-sum-exp of the two other logits. Fuse those ranks with retrieval ranks:
`1/(10 + retrieval_rank) + 3/(10 + entailment_rank)`. Candidates outside the
scored prefix retain their retrieval term. Apply the unique eligible whole-label
spelling priority without converting it into a verified alias. Record raw model
margins, fused scores, retrieval ranks, and parent/child relationships. Preserve
the full shortlist; return alternatives according to the response limit.

If only the supplied parent receives positive NLI support and every scored
descendant is rejected, preserve retrieval order and record a specificity
conflict. Retain model scores/ranks and require clarification even under a
qualified policy. This is a conservative disagreement fallback, not acceptance
of the retrieval winner. Missing scores do not imply a rejected descendant.

The current weak-promotion guard also protects the retrieval winner when both
semantic scores exist, its semantic support is more than 0.03 higher, and the
promoted candidate's entailment margin is below 1.0. These development-selected
values are ranking heuristics and are pinned in the artifact. The conflict must
abstain under a policy. Whole coordinated-label alternatives give provisional
group evidence; collection/recommendation titles without product information stay
unmapped. See the current findings for regressions, test evidence, and limits.

Canonical ID/path/label lookup and approved aliases run first. Conservative
literal general-category wrappers can also resolve a canonical label directly.
The current adapter preserves parent constraints and the existing infant-apparel
exclusion, but remaining broad-parent/related-product ranking failures still need
work before qualification. Retrieve-only mode is available through `--no-rerank`.

Choose a category at the specificity supported by the input. For example, `phone accessories` supports Mobile Phone Accessories; it does not justify Mobile Phone Cases. An unclear competition between sibling or parent/child categories produces clarification unless a reviewed rule or validated decision threshold resolves it.

Do not automatically return the common ancestor of unrelated candidates. A shared ancestor is not proof that it correctly describes the input.

Retrieve-and-rerank is the selected two-stage design; its quality improvement for this specific taxonomy must be demonstrated against the retrieval-only baseline. [Sentence Transformers retrieve-and-rerank documentation](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)

### 4.5 Acceptance policy

Implement a versioned policy with these outcomes:

Only an accepted `matched` result may populate the assigned `category` field.
Provisional, clarification, no-match, and unsupported-language results must keep
`category: null`. Consumers must never automatically assign the first ranked
candidate or an experimental decision preview. Unsupported store-section titles
stay unmapped; an unsupported specific category must not become a forced broad
fallback. Human review can leave the input unmapped rather than inventing an ID.

| Outcome | Required behavior |
|---|---|
| `matched` | Return the accepted category ID and original path |
| `needs_clarification` | Return no selected category, up to three alternatives, and a precise context request |
| `no_match` | Return no selected category because the input lacks adequate support in the taxonomy |
| `unsupported_language` | Return no selected category when the declared language is outside the English release |

The deterministic exact/alias path and the model path must both appear in evaluation reports. Alias approval is not a blanket exemption from ambiguity or context checks.

For the model path, tune a rule using the top reranker score, margin over the second result, semantic support, and explicit context/specificity checks. Maintain distinct thresholds for acceptance and for distinguishing clarification from no match. Store selected values in `decision_policy.json`.

Determine thresholds on the calibration split only. Do not invent values such as `0.90 similarity = 90% confidence`. Do not display a probability field in version 1. Return method and reason codes instead.

If calibration cannot separate a difficult input slice reliably, that slice remains on clarification. The uncertainty path is an essential part of meeting the accuracy target.

### 4.6 Required architecture for easy feature changes

#### Dependency boundaries

Organize the application into four layers plus an explicit composition root:

| Layer | Owns | Dependency rule |
|---|---|---|
| `domain` | Category, query, candidate, result, scope, and policy contracts | Depends only on the Python standard library |
| `application` | Matching, index-build, feedback, and evaluation use cases | Depends on domain types and interfaces |
| `infrastructure` | SQLite/`sqlite-vec`, migrations, BM25S, RapidFuzz, and model adapters | Implements application-facing interfaces |
| `presentation` | HTTP request/response mapping and endpoint definitions | Calls application use cases; contains no SQL or ranking logic |
| `bootstrap.py` | Configuration, adapter registry, lifecycle, and dependency construction | Wires concrete adapters into the use cases |

API schemas remain in `presentation`; the core uses framework-independent dataclasses. Neither FastAPI, SQLite, nor Sentence Transformers may be imported by domain modules. Application modules must not construct concrete database connections or model objects.

#### Stable interfaces

Define small `typing.Protocol` contracts around actual replaceable boundaries:

| Contract | Responsibility |
|---|---|
| `CategoryRepository` | Look up canonical records and hierarchy relationships |
| `AliasRepository` | Resolve approved aliases and expose conflicts |
| `QueryEncoder` | Return normalized vectors and an embedding-space identifier |
| `VectorStore` | Retrieve scoped category candidates from vectors in a declared embedding space |
| `CandidateRetriever` | Return ranked, deduplicated candidates for a query, scope, and limit |
| `CandidateReranker` | Score a supplied candidate list |
| `DecisionPolicy` | Produce a semantic outcome from ranking evidence and constraints |
| `FeedbackRepository` | Store reviewable confirmations and corrections |
| `ArtifactRepository` | Resolve and validate immutable artifact versions |
| `TaxonomyRegistry` | Resolve a taxonomy key and supported version to its bundle and configuration |
| `TaxonomySourceAdapter` | Parse a vendor format into canonical categories, source metadata, and parent edges |

`Candidate` includes taxonomy key/version, category ID, source name, rank, raw score, score meaning, and artifact version. A cosine distance is marked lower-is-better; a similarity or reranker score is marked higher-is-better. Fusion consumes ranks rather than assuming score comparability.

The semantic retriever depends on `QueryEncoder` and a vector-store interface. The concrete SQLite vector-store adapter owns SQL and vector serialization. That keeps model replacement separate from storage replacement.

#### Pipeline composition and configuration

Register the BM25, fuzzy, and semantic retrievers by stable names at startup and inject the configured list into the matching use case. Fusion iterates over this list. Adding a retriever must not require adding a new conditional branch throughout the matching code.

Validate configuration through a settings model: enabled retrievers, per-branch limits, model revisions, embedding dimension, normalization version, fusion parameters, queue limits, and policy/artifact versions. Reject unknown adapter names and incompatible combinations at startup.

Defaults live in version-controlled configuration. Secrets and deployment paths come from the environment. Record all quality-affecting configuration in the artifact manifest. A changed model, candidate limit, retriever set, or acceptance policy requires evaluation before activation.

The composition root is the only place that selects concrete adapters. Do not implement arbitrary code loading or a separate plugin platform for these internal extension points.

#### Change workflows

| Future change | Required change surface | Release evidence |
|---|---|---|
| Add a vendor taxonomy | Source adapter if its format is new, registry entry, scoped artifacts, aliases, and policy | Source integrity, taxonomy isolation, and vendor-specific qualification |
| Add a new retrieval method | New `CandidateRetriever` adapter, registration, and configuration | Adapter contract tests, candidate recall, and recalibration |
| Replace an embedding model | Encoder adapter/configuration and a newly built vector bundle | New embedding-space ID, rebuilt vectors, evaluation, and recalibration |
| Replace a reranker | Reranker adapter/configuration | Ranking comparison and recalibrated policy |
| Add a language | Language-specific routing, qualified models/aliases, and labeled data | Per-language accuracy and coverage |
| Add a batch endpoint | Batch application use case and HTTP mapping | Per-item equivalence, bounded batch size, and capacity tests |
| Add authentication | Presentation middleware/dependency and deployment configuration | Access-control and integration tests |
| Add a field or review workflow | Additive SQL migration and relevant repository/use-case changes | Migration tests and backward-compatible API behavior |
| Change storage engine | New repository/vector-store adapters and a data migration | Shared contract tests and retrieval/result equivalence |

Approved synonyms are data changes with review and versioning; they must not require editing matching source code. New features must not grow into one large `matcher.py` or mix SQL, inference, and HTTP concerns.

#### Database and API evolution

Use ordered, checksum-verified SQL migrations with a `schema_migrations` table in each database. Apply migrations to staging taxonomy bundles during build and to the runtime database through a controlled migration command before service startup. Startup checks compatibility and never silently mutates the production schema.

Prefer additive schema changes. For a rename or removal, add the replacement, migrate data, deploy compatible readers/writers, then remove obsolete fields in a later release. Back up runtime state before destructive migrations. Artifact rollback switches to a compatible prior bundle; runtime-database recovery restores a verified backup when forward compatibility is unavailable.

Keep the public `/v1` semantics stable. Add optional fields compatibly; use a new API version for breaking request or response changes. Test adapter contracts, migrations, and API compatibility in addition to domain behavior.

#### Growth in traffic and data

Easy feature changes are the immediate requirement. Capacity growth must also have a defined path:

1. **Initial deployment:** one process on this Mac, one inference job at a time, bounded queue, immutable taxonomy/vector bundle, separate writable feedback database.
2. **Higher inference demand:** move model execution behind the existing encoder/reranker interfaces to dedicated serving processes or hosts. Keep matching/API nodes stateless with respect to individual requests and distribute the same versioned bundles to each node. Add request timeouts and inference-service failure handling at the adapter boundary.
3. **Multiple application replicas:** each replica reads its own local immutable taxonomy/vector bundle. Centralize feedback/events behind the feedback repository so all replicas write through one authoritative store. Use a single-writer service while it meets capacity; move that operational store to PostgreSQL when cross-host shared writes or write contention require it.
4. **Larger taxonomy:** benchmark the vector-store adapter and replace exact scan with an indexed backend when vector retrieval exceeds its latency budget. Revalidate recall before switching. `sqlite-vec` exact search is not a promise of unlimited approximate-index capacity.

Keep a bounded queue and measure throughput, p95 latency, queue wait, model memory, and SQLite busy/timeout events. A sustained failure of the qualified traffic/latency target triggers capacity work. A new backend requires adapter implementation, data migration, and validation; an interface makes that work localized rather than automatic.

SQLite serializes writers. WAL improves read/write concurrency but does not allow multiple simultaneous writers and is unsuitable for sharing one live database across hosts through a network filesystem. Replicate immutable read bundles; do not share the writable SQLite file between machines. [SQLite deployment guidance](https://www.sqlite.org/whentouse.html), [SQLite WAL documentation](https://www.sqlite.org/wal.html)

### 4.7 Multiple vendor taxonomies

The engine must support multiple registered taxonomies from the first release, even though only the Google dataset is currently available. Vendor support is an architectural capability; the accuracy of an unprovided taxonomy is not established.

#### Isolation and routing

- Require `taxonomy_key` in match requests. An unknown taxonomy is a validation error, not a reason to fall back to Google.
- Resolve the active version for that taxonomy once at request entry. An optional explicit version can select a retained, supported bundle. Return the resolved version in every semantic response.
- Scope repositories, exact dictionaries, aliases, hierarchy constraints, vectors, policies, feedback, and evaluation datasets to that identity.
- Scope any future result cache by taxonomy, resolved bundle version, normalized input, context, parent constraint, and language.
- Keep one immutable artifact directory per taxonomy/version. A Google update must not activate a Meta bundle or invalidate unrelated vendor aliases.
- Preserve native identifiers as strings, including significant leading zeros. A Google ID `"267"` and a future vendor ID `"267"` are different entities.

An English encoder or reranker can be shared between taxonomies when the model revision and encoding configuration match. Category embeddings and calibrated policies remain taxonomy-specific. Compute one query embedding for a shared embedding space when processing a multi-taxonomy application request; do not reuse vectors across incompatible model spaces. Multiple different models resident at once must pass the existing memory gate.

#### Adding a future Meta list

1. Obtain its actual source format, language, native IDs, hierarchy rules, version, and update policy.
2. Reuse an existing source adapter when its format contract fits; otherwise implement a new `TaxonomySourceAdapter`. Canonical matching, fusion, and API orchestration remain vendor-independent.
3. Register `meta` with its adapter, language/model configuration, and source metadata.
4. Ingest and validate source records, edges, cycles, normalized-label collisions, and native IDs. If a source lacks IDs, define stable, vendor-scoped identifiers and an update mapping before activation; do not invent unstable line-number IDs.
5. Build the Meta exact lookup, hierarchy closure, BM25/fuzzy data, SQLite vectors, and manifest in an isolated bundle.
6. Create reviewed Meta aliases and labeled Meta examples. Do not copy Google's category-ID mappings or confidence thresholds into Meta.
7. Calibrate and qualify Meta independently, then activate its registry entry.
8. Verify the same input routed to Google and Meta produces only categories from the requested taxonomy, including a fixture where both vendors reuse an ID.

File-only addition is sufficient only when an existing adapter can parse the file and the remaining build, validation, and qualification steps pass. Placing a file in the directory does not automatically establish correct matching.

#### Multi-taxonomy output and conversion

The core match use case selects one taxonomy. A future endpoint that requests Google and Meta results invokes that same use case separately for each taxonomy and returns separately namespaced results. It must not fuse candidates from different taxonomies into a single ranking.

Converting an existing Google category to a Meta category is a separate crosswalk feature. It requires reviewed mappings or a qualified crosswalk workflow, may be one-to-many or unmappable, and must not equate vendors through IDs or identical labels alone.

## 5. Data storage and artifact layout

Use these SQLite tables:

| Table | Essential fields |
|---|---|
| `taxonomies` | taxonomy key, vendor/source identity, language support, source-adapter key |
| `taxonomy_versions` | taxonomy key, version key, source SHA-256, source URL if verified, import timestamp, record count |
| `categories` | taxonomy key/version, native category ID as TEXT, label, canonical display path, normalized label/path, depth, leaf flag |
| `category_edges` | taxonomy key/version, parent ID, child ID; supports more than one parent |
| `category_closure` | taxonomy key/version, ancestor ID, descendant ID, minimum edge distance, including self links |
| `category_embeddings` | taxonomy key/version, category ID, view (`label` or `path`), float32 embedding BLOB |
| `aliases` | taxonomy key/version, normalized alias, category ID, optional parent scope, status, reviewer, review timestamp |
| `match_events` | event ID, taxonomy key/version, timestamp, input/context, outcome, selected ID, candidate IDs and scores, latency, artifact version |
| `feedback` | event ID, taxonomy key/version, confirmed category or no-match/clarification correction, review state, timestamp |
| `schema_migrations` | migration namespace/version, checksum, applied timestamp; present in each database |

Use `(taxonomy_key, taxonomy_version, category_id)` category keys and corresponding composite foreign keys, including edge endpoints, aliases, and embeddings. Validate feedback targets against the event's taxonomy/version. Enable SQLite foreign-key enforcement on every connection. Allow colliding aliases to remain reviewable records; only approved, unambiguous, correctly scoped aliases enter the automatic lookup.

### 5.1 SQLite and sqlite-vec integration

Use ordinary SQLite tables for embeddings and `sqlite-vec` scalar cosine-distance search in the first release. This permits exact aggregation over label/path views and arbitrary parent-scope filtering before the configured candidate limit. Keep all vector SQL inside the vector-store adapter. [sqlite-vec KNN methods](https://alexgarcia.xyz/sqlite-vec/features/knn.html)

The embedding table has a composite primary key `(taxonomy_key, taxonomy_version, category_id, view)`, a matching category foreign key, a `view` enum check, and an embedding BLOB check for float32 type, 768 dimensions, and 3,072 bytes for the selected initial model. Reject nonfinite or zero-norm model output in the adapter before insertion. A different dimensionality creates a new compatible schema/bundle; it must never be mixed into the active 768-dimensional space.

For an unscoped request, the semantic branch uses this query shape:

```sql
SELECT category_id,
       MIN(vec_distance_cosine(embedding, :query_vector)) AS distance
FROM category_embeddings
WHERE taxonomy_key = :taxonomy_key
  AND taxonomy_version = :taxonomy_version
GROUP BY category_id
ORDER BY distance ASC, category_id ASC
LIMIT :candidate_limit;
```

Bind `query_vector` as a normalized float32 BLOB. Apply a parent constraint with an `EXISTS` filter against `category_closure` in the `WHERE` clause before aggregation and limit. The closure table includes the selected parent itself and all descendants. Index the closure keys used by the filter. Store one embedding space per active bundle and validate its identifier on every query. The scalar functions support vector-distance computation over ordinary tables. [sqlite-vec API reference](https://alexgarcia.xyz/sqlite-vec/api-reference.html)

Every vector-capable connection is created by the SQLite connection factory. It loads the pinned extension through the `sqlite-vec` Python package, disables extension loading after load, checks `vec_version()`, enables foreign keys, and sets a bounded busy timeout. Never load an extension path supplied by an API caller. Verify extension loading, vector insertion, and a known nearest-neighbor result in a build/startup smoke check. [sqlite-vec Python integration](https://alexgarcia.xyz/sqlite-vec/python.html)

Open published taxonomy/vector bundles read-only. Use WAL for the local writable feedback/event database, keep transactions short, and give each execution thread its own connection or exclusive ownership of a pooled connection. Do not share one Python SQLite connection concurrently across inference threads. Before publishing a built bundle, close build connections and ensure all data is checkpointed into the artifact being copied.

### 5.2 Artifact publication and consistency

Separate the immutable taxonomy/index bundle from the writable feedback database. Build a new bundle in a staging directory, validate it, then switch the active manifest atomically. Keep the last working bundle for rollback.

Each request captures one bundle handle at entry and uses that version for exact lookup, all retrievers, reranking metadata, and decision policy. In-flight requests finish on their captured version; only new requests see an activated bundle. An artifact switch must not mix old vectors with new labels or policies.

On the 8 GB Mac, activate changes that require different resident models through a controlled drain and process restart. Do not assume two generations of model weights can coexist within the memory target. Compatible index/policy updates may use captured-handle activation without duplicating models.

The artifact manifest must record:

- Taxonomy key, source-adapter/configuration revision, source-file checksum, record count, and taxonomy version key.
- Embedding and reranker model names and immutable revisions.
- Embedding dimension, dtype, view order, and ID mapping checksum.
- Embedding-space identifier, SQLite schema version, `sqlite-vec` version, distance metric, and vector row count.
- Normalization/tokenization versions and query instruction.
- BM25, fuzzy, fusion, and candidate-count configuration.
- Approved-alias snapshot hash.
- Decision-policy version and hash.
- Relevant package versions and build timestamp.

Fail readiness if the database, vector table, aliases, extension, schema, or policy reference incompatible manifests. Rebuild relevant artifacts and recalibrate when their inputs change.

## 6. API contract

Implement:

| Endpoint | Purpose |
|---|---|
| `GET /v1/taxonomies` | List activated taxonomies, versions, and supported languages |
| `POST /v1/categories/match` | Match one category input |
| `GET /v1/taxonomies/{taxonomy_key}/categories/{category_id}` | Retrieve the canonical record; optional retained version query parameter |
| `POST /v1/categories/feedback` | Store a confirmation or correction for review |
| `GET /health/live` | Report that the process is running |
| `GET /health/ready` | Report that the active artifacts and models are loaded and compatible |

Request:

```json
{
  "taxonomy_key": "google",
  "taxonomy_version": null,
  "category": "protective covers for cellphones",
  "context": null,
  "parent_category_id": null,
  "language": "en"
}
```

Require a registered `taxonomy_key` and resolve its active version when `taxonomy_version` is null. Also accept an explicit string `category_id` field for canonical-ID lookup. `parent_category_id` is also an opaque string in the selected taxonomy/version. Require at least category text or an ID. Validate text after trimming: category 1–300 characters, optional context up to 1,000 characters, language from the supported enum. Reject unknown fields, unregistered taxonomy/version combinations, and invalid parent IDs. Whitespace-only input is invalid.

An accepted response has this shape; the example illustrates the contract, not a measured model prediction:

```json
{
  "status": "matched",
  "taxonomy_key": "google",
  "taxonomy_version": "<resolved-source-version>",
  "category": {
    "id": "2353",
    "path": "Electronics > Communications > Telephony > Mobile Phone Accessories > Mobile Phone Cases"
  },
  "method": "hybrid_rerank",
  "reason_code": "acceptance_policy_passed",
  "alternatives": [],
  "clarification": null,
  "artifact_version": "<active-manifest-version>",
  "event_id": "<event-id>"
}
```

For clarification, set `category` to `null`, return canonical alternatives, and explain the distinction that needs resolving. For `apple`, ask whether the user means the fruit or Apple-related electronics. For unsupported language or no match, return no assigned category.

Use HTTP 200 for valid semantic outcomes, 422 for invalid request data, and 503 when the service is unready or its bounded inference queue is full. A semantic no-match is not an application exception.

Do not truncate arbitrary long context silently. Enforce the input contract and verify tokenizer limits. The declared language is a caller contract; do not claim automatic detection of every non-English or transliterated input.

## 7. Step-by-step implementation plan

The commands and module paths below describe work to implement. They have not been executed or created by writing this plan.

### Step 1 — Establish the product rules and dataset version

1. Make the contract in Section 1 the executable specification.
2. Register the current source checksum as the first local taxonomy version under taxonomy key `google`. Create the vendor registry and source-adapter contract before ingestion.
3. Preserve the original file bytes.
4. Record English-only support, parent-category behavior, ambiguity handling, and release targets in configuration/documentation.
5. Record the expected calling application and actual traffic when integration begins. Local capacity remains unverified until measured.

**Completion condition:** every endpoint outcome and hierarchy decision has a defined meaning; the source vocabulary is fixed.

### Step 2 — Create the isolated Python project

From this project directory, run during implementation:

```bash
uv init --python 3.11 --package --name google-category-matcher
uv add fastapi 'uvicorn[standard]' 'pydantic>=2,<3' sqlite-vec numpy bm25s rapidfuzz sentence-transformers torch
uv add --dev pytest httpx ruff scipy
uv lock
uv sync --frozen
```

Use the existing uv-managed Python 3.11 interpreter. Verify the environment architecture is `arm64` and the installed PyTorch builds can execute both selected models. Do not alter the system Python installation.

Create the layer structure in Section 9 before implementing the matching flow. Define the domain types, interfaces, and composition root first. Configure Ruff import restrictions for domain/application boundaries and add a focused architecture test that fails if core modules import FastAPI, concrete database adapters, or model frameworks.

Verify extension integration inside the new environment with an in-memory smoke check:

```bash
uv run python - <<'PY'
import sqlite3
import sqlite_vec

connection = sqlite3.connect(":memory:")
connection.enable_load_extension(True)
try:
    sqlite_vec.load(connection)
finally:
    connection.enable_load_extension(False)
print(connection.execute("SELECT sqlite_version(), vec_version()").fetchone())
connection.close()
PY
```

Use this same loading sequence inside the application connection factory. Record the verified extension version; installing the SQLite CLI alone does not complete this check.

Configure model cache paths under a project-owned ignored directory. Exclude `.venv`, model caches, generated indexes, local databases, and query logs from Git. Track manifests, code, lockfiles, and reviewed alias source files. Initialize Git if uv has not already done so.

uv's project workflow manages the manifest, environment, and lockfile. [uv project documentation](https://docs.astral.sh/uv/guides/projects/)

**Completion condition:** a frozen sync, extension load/version query, and architecture-boundary checks succeed in `.venv`.

### Step 3 — Implement taxonomy ingestion

1. Read UTF-8 with optional BOM support.
2. Skip blank lines and upstream comment headers, while recording their metadata.
3. Implement the Google text source adapter: parse each record at the first ` - ` separator and split the hierarchy on ` > `.
4. In that adapter, validate numeric source IDs, nonempty hierarchy segments, unique IDs, and unique full paths, then preserve IDs as strings. Other adapters validate their own native ID contracts. Core matching must not assume numeric IDs or unique labels/paths.
5. Resolve Google's immediate parents by full path. Validate the canonical edge graph for missing endpoints and cycles; other source adapters may supply explicit multi-parent edges and a deterministic display path.
6. Calculate depth, leaf status, normalized keys, and collision sets.
7. Implement the checksum-verified migration runner and apply initial schema migrations to a staging database.
8. Import all categories transactionally into a versioned SQLite snapshot and build `category_closure`, including self links.
9. Emit a machine-readable audit report and fail the build on malformed or conflicting source records.

**Completion condition:** this source produces exactly 5,582 category rows with the audited counts and relationships.

### Step 4 — Build exact matching and the alias review path

1. Implement conservative normalization as a single versioned function.
2. Build ID, full-path, and collision-aware label dictionaries.
3. Create reviewed alias records with category IDs and optional parent scopes.
4. Seed clearly reviewed vocabulary such as `cellphones` to Mobile Phones (`267`) and `couches` to Sofas (`460`). Mark these as intentional mappings to the available taxonomy level.
5. Keep ambiguous phrases such as `apple` out of global automatic aliases.
6. Validate every alias target against the active taxonomy.
7. Keep aliases derived from development data separate from held-out evaluation inputs.

**Completion condition:** exact and approved-alias matches return canonical records, while conflicting keys cannot auto-match.

### Step 5 — Assemble and label the evaluation data

Create an initial 600 manually reviewed examples with these coverage quotas:

| Slice | Examples |
|---|---:|
| Exact labels, paths, and valid IDs | 100 |
| Synonyms and paraphrases | 140 |
| Typos and word-order changes | 100 |
| Broad versus specific hierarchy decisions | 100 |
| Ambiguous inputs | 80 |
| Out-of-taxonomy and unsupported-language inputs | 80 |

Each record includes taxonomy key/version, input, context, language, expected status, acceptable category IDs, input family, source, and adjudication notes. Use explicit ambiguity labels rather than arbitrarily choosing one ID. Record invalid API payloads in functional tests separately. These initial 600 examples qualify only the Google source; a future vendor needs its own labeled data and release evidence.

Split into 200 development, 200 calibration, and 200 untouched test examples. Group near-duplicate phrases, aliases, typo variants, and input families together so they cannot leak across splits. Maintain coverage across the 21 top-level branches, including difficult neighboring categories.

An independently reviewed subset must resolve labeling disagreements before evaluation. This initial set supports development; it is too small by itself to substantiate a strong production accuracy claim.

**Completion condition:** the benchmark has trusted expected outcomes and frozen split membership.

### Step 6 — Build reproducible retrieval artifacts

1. Download the embedding model and reranker once; record immutable revisions.
2. Encode label and path views without query instructions, in batches of 16 on CPU float32, with embedding normalization enabled.
3. Insert the float32 BLOBs and explicit category-ID/view keys into `category_embeddings` through the SQLite vector-store adapter. Expect exactly 11,164 embedding rows for this file.
4. Build the BM25 index and RapidFuzz choices using versioned normalization rules.
5. Implement semantic SQL distance aggregation by category ID, filtering before limit, and branch deduplication. Compare rankings against an independent NumPy cosine calculation within numerical tolerance, including parent-scoped queries.
6. Implement RRF and the candidate limits in Section 4.3.
7. Write the artifact manifest and verify round-trip reload from SQLite without separate embedding-array files.
8. Measure recall after the final candidate shortlist, including hierarchy expansion. Diagnose any missing gold categories before tuning acceptance.

**Completion condition:** artifacts rebuild deterministically within documented numerical tolerance; final candidate recall is reported by slice.

### Step 7 — Implement and verify reranking

1. Load one cross-encoder instance for the process.
2. Score the configured prefix of candidate pairs in batches of 8 under inference mode.
3. Fuse model and retrieval ranks; preserve raw scores, top-result margin, and retrieval evidence.
4. Check parent/child competition and explicit parent constraints.
5. Compare reranked top-1 performance with fused retrieval on development data.
6. Freeze candidate formatting and model revisions before calibration.

**Completion condition:** the report shows whether reranking improves the actual category task, and all returned IDs remain valid. If it harms a slice, fix the candidate text, reviewed rules, or ranking behavior before release; do not assume a generic passage model guarantees improvement.

### Step 8 — Calibrate the decision policy

1. Use the calibration split to choose score, margin, and support thresholds.
2. Select a configuration that prioritizes precision subject to the coverage target.
3. Check ambiguous, no-match, parent/child, and typo slices separately.
4. Save the policy and the calibration report as versioned artifacts.
5. Evaluate once on the untouched test set.
6. When changing the policy after inspecting test failures, treat those examples as development data and obtain a fresh holdout for the next release decision.

**Completion condition:** every non-exact assignment passes a calibrated policy, and uncertain outcomes remain unassigned.

### Step 9 — Implement the API and lifecycle

Sprint 3 implements this lifecycle in `src/category_matcher/api/`: `app.py` is the
composition root, `service.py` owns the inference thread and model lifecycle, and
`schemas.py` defines the HTTP contracts. The broader module layout below remains
an architecture target. Only Google and its active version are currently enabled;
feedback recording and recovery are implemented in Sprint 4 (historical). See [the runnable API contract](README.md#run-the-local-api)
for additional response fields, limits, error codes, and configuration.

1. Define Pydantic request/response schemas with the mutually consistent status fields.
2. Build the application through `bootstrap.py`: load compatible artifact handles, construct the configured adapters, and initialize both models during FastAPI startup.
3. Mark readiness only after load and a small inference check succeed.
4. Run CPU inference outside the async event loop through a bounded executor.
5. Allow one model inference job at a time initially; cap pending inference requests at eight and return 503 when full.
6. Reuse model instances and read-only bundle handles across requests. Keep SQLite connection ownership scoped to the executing thread and capture one artifact version per request.
7. Record event IDs, stage timings, outcomes, and artifact versions.
8. Implement feedback recording without automatic promotion to approved aliases.

Start the implemented service with:

```bash
uv run category-matcher-api --host 127.0.0.1 --port 8000
```

One process bounds duplicated model memory. FastAPI documents worker-based deployment and process management. [FastAPI workers documentation](https://fastapi.tiangolo.com/deployment/server-workers/)

**Completion condition:** documented endpoints work against the real loaded artifacts, including readiness and overload behavior.

### Step 10 — Run functional and quality verification

Test meaningful behavior:

- All 5,582 canonical IDs and full paths round-trip correctly.
- Unique canonical labels resolve to their own IDs under conservative normalization.
- BOMs, missing final newline, duplicate IDs, broken parents, and normalization collisions are handled correctly.
- Phone cases do not become phones solely because the terms overlap.
- Prescription and non-prescription categories stay distinct.
- Broad input does not automatically acquire an unsupported child qualifier.
- Aliases respect parent scope and cannot reference removed categories.
- Ambiguous and out-of-taxonomy examples remain unassigned as labeled.
- Invalid input, incompatible artifacts, startup failures, and queue saturation produce the defined responses.
- The exact, retrieval, and model paths all resolve category paths from the canonical store.
- Every retriever and repository passes its interface contract tests; a small test retriever can be registered without editing matching orchestration.
- Migration checksums, upgrades from the previous supported schema, and a simulated bundle rollback work correctly.
- SQLite vector distances and category grouping agree with independent cosine calculations, including parent scopes and ties.
- An in-flight request uses one captured bundle version during concurrent bundle activation.
- Architecture tests prevent domain/application imports of concrete frameworks and adapters.
- A second synthetic taxonomy with overlapping IDs and different categories remains isolated across requests, aliases, vectors, parent scopes, feedback, and bundle activation. An unknown taxonomy never falls back to Google.
- A synthetic multi-parent graph retains its edges, respects hierarchy scope, and keeps the reranking expansion cap.

Run after implementation:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python -m category_matcher.cli.evaluate --split test
```

**Completion condition:** functional checks pass and the evaluation report includes all metrics in Section 8.

### Step 11 — Benchmark on this M1 Mac

1. Measure cold model-load time separately from warm request latency.
2. Run warm exact/alias requests and warm model-path requests separately.
3. Measure at least 200 model-path requests for a first latency report.
4. Record p50/p95 latency, process peak resident memory, queue wait, and failures.
5. Test one active request, then the configured bounded queue.
6. Check memory pressure and swapping with both models resident.
7. Keep CPU float32 as the verified baseline. Changes to execution device, precision, or batch sizes require the same correctness and resource checks before activation.

Initial operational targets are warm model-path p95 at most 3 seconds with one active request, process peak resident memory below 3 GB, and no sustained swapping attributable to the service. These are engineering targets, not measured performance claims. Exact-search vector cost is small; model inference is expected to dominate and must be measured.

**Completion condition:** the machine meets the targets or the report explicitly identifies the remaining capacity gap. Production concurrency cannot be committed until real traffic requirements are known.

### Step 12 — Pilot, qualify, and maintain the release

1. Run a reviewable pilot in which human reviewers check accepted and uncertain outcomes.
2. Collect at least 1,000 additional independently labeled, representative inputs for release qualification, separate from policy tuning data.
3. Freeze the pipeline and thresholds before evaluating that qualification set.
4. Apply the release gates in Section 8.
5. Publish the resulting artifact bundle and retain the previous bundle.
6. Review feedback and recurring unknown vocabulary before adding aliases.
7. Compare upstream taxonomy updates by added/removed IDs, renamed paths, and parent moves. Rebuild and requalify before switching versions.

**Completion condition:** the release has measured accuracy, documented coverage, reproducible artifacts, and rollback behavior.

## 8. Accuracy metrics and release gates

### Metrics

```text
automatic-match precision = correct automatic matches / all automatic matches

resolvable-input coverage = automatically matched resolvable inputs / all resolvable inputs

overall automatic coverage = all automatic matches / all valid inputs

candidate recall = resolvable inputs with an acceptable category in the final
                   pre-reranking shortlist / all resolvable inputs

ambiguous-input false acceptance = ambiguous inputs auto-matched / ambiguous inputs

no-match false acceptance = unsupported/out-of-taxonomy inputs auto-matched /
                           unsupported/out-of-taxonomy inputs
```

Report metrics separately for each taxonomy/version and supported-language/unsupported-language slice. Report model-path results separately from exact/alias results so easy lookups cannot conceal weak semantic matching. Report top-1 accuracy before abstention, top-3 recall, and ancestor-versus-leaf mistakes as diagnostics. A strong Google result must not conceal a weak Meta result in an aggregate score.

### Required release gates

| Gate | Required result |
|---|---|
| Output integrity | 100% of returned IDs and paths exist in the active taxonomy |
| Canonical lookup | 100% correct canonical-ID/path round trips |
| Final candidate recall | At least 99.5% on resolvable qualification inputs |
| Automatic-match precision | At least 99%, with a one-sided 95% exact binomial lower confidence bound also at least 99% |
| Resolvable-input automatic coverage | At least 80% on representative English inputs |
| Ambiguity/no-match handling | False-acceptance rates reported separately and included as errors in precision |
| Resource behavior | The Step 11 targets met on the intended host |
| Reproducibility | Frozen dependencies, model revisions, source checksum, and policy recorded |
| Extension and schema compatibility | Verified `sqlite-vec` loading/version, vector dimensions, and migration history |
| Maintainability | Layer boundaries, adapter contracts, migration upgrades, and compatible API behavior pass checks |
| Vendor isolation | Overlapping IDs, aliases, hierarchy scopes, and active versions cannot leak between taxonomies |

These values define the desired release standard for each activated taxonomy; the current project has no measured accuracy and cannot yet claim to meet it. Vendor activation is independent, so an unqualified new vendor stays unavailable while qualified vendors continue operating.

Use an exact binomial confidence interval, for example through SciPy's `binomtest` result with a one-sided alternative and `proportion_ci(method="exact")`. At least 299 independent accepted examples with zero errors are needed for a one-sided 95% lower precision bound to reach 99%. Repeated copies of one query do not establish that evidence. More examples are required when errors occur, and the model path needs adequate independent coverage of its own. [SciPy binomial-test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)

If precision passes only by abstaining on nearly every difficult input, the coverage gate fails. If the gates fail, continue development and obtain a fresh qualification set after changes; do not report the target as achieved.

## 9. Planned source structure

```text
RAG/
├── google_category_list.txt
├── CATEGORY_MATCHING_IMPLEMENTATION_PLAN.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── data/
│   └── google/                     # future vendor gets a separate directory
│       ├── aliases.jsonl
│       └── evaluation/
│           ├── development.jsonl
│           ├── calibration.jsonl
│           ├── test.jsonl
│           └── qualification.jsonl
├── artifacts/                       # generated, ignored
│   └── <taxonomy-key>/
│       └── <bundle-version>/
│           ├── taxonomy.sqlite3    # categories, edges, closure, aliases, vectors
│           ├── bm25/
│           ├── decision_policy.json
│           └── manifest.json
├── models/                          # downloaded cache, ignored
├── var/                             # writable runtime state, ignored
│   └── feedback.sqlite3
├── reports/
├── config/
│   ├── defaults.toml
│   └── taxonomies.toml             # source adapters and active versions
├── src/category_matcher/
│   ├── __init__.py
│   ├── bootstrap.py                # adapter registry, configuration, wiring
│   ├── settings.py
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── models.py               # query, category, candidate, result types
│   │   ├── ports.py                # Protocol contracts
│   │   ├── normalization.py
│   │   └── policy.py               # pure acceptance/specificity rules
│   ├── application/
│   │   ├── __init__.py
│   │   ├── match_category.py        # orchestrates injected components
│   │   ├── build_artifacts.py
│   │   ├── record_feedback.py
│   │   ├── evaluate.py
│   │   └── fusion.py
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   ├── sqlite/
│   │   │   ├── __init__.py
│   │   │   ├── connection.py       # extension loading and connection ownership
│   │   │   ├── repositories.py
│   │   │   ├── vector_store.py
│   │   │   ├── migration_runner.py
│   │   │   └── migrations/
│   │   │       ├── taxonomy_001_initial.sql
│   │   │       └── runtime_001_initial.sql
│   │   ├── retrieval/
│   │   │   ├── __init__.py
│   │   │   ├── bm25.py
│   │   │   ├── fuzzy.py
│   │   │   └── semantic.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── bge_encoder.py
│   │   │   └── minilm_reranker.py
│   │   ├── taxonomy_sources/
│   │   │   ├── __init__.py
│   │   │   └── google_text.py
│   │   ├── taxonomy_registry.py
│   │   └── artifacts.py
│   ├── presentation/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   └── schemas.py
│   └── cli/
│       ├── __init__.py
│       ├── build_index.py
│       ├── migrate.py
│       ├── evaluate.py
│       └── benchmark.py
└── tests/
    ├── unit/                       # pure domain/application behavior
    ├── contracts/                  # shared adapter interface tests
    ├── integration/                # SQLite, sqlite-vec, migrations, artifacts
    ├── api/                        # endpoint and API-compatibility behavior
    └── test_architecture.py
```

Build the `cli.build_index`, `cli.migrate`, `cli.evaluate`, and `cli.benchmark` modules with documented command-line arguments. CLI entry points call the same application use cases as other entry points. Include migration SQL files as package data. Keep ingestion, retrieval, reranking, and decision logic independently callable so failures can be diagnosed by stage.

## 10. Expected behaviors to verify

These are acceptance examples based on real entries in the supplied file, not claims that the unbuilt model already produces them.

| Input | Required outcome |
|---|---|
| `Mobile Phones` | Match `267`: Electronics > Communications > Telephony > Mobile Phones |
| `cellphones` | Reviewed alias to `267` |
| `mobil phons` | Recover `267` as a candidate; accept only when the calibrated policy passes |
| `phone protective covers` | Match `2353`: Electronics > Communications > Telephony > Mobile Phone Accessories > Mobile Phone Cases, if policy passes |
| `phone accessories` | Match broader category `264`; do not force a particular accessory type |
| `couches` | Reviewed alias to `460`: Furniture > Sofas |
| `apple` | Clarification: fruit versus Apple-related electronics |
| `legal consultation services` | No match in this product taxonomy |
| English model input declared `bn` | Unsupported language; do not run the English model as though Bangla were supported |
| A nonexistent explicit category ID | Validation error |

## 11. Inputs still required to achieve the desired output

The implementation can proceed with the fixed decisions above. The following evidence must be supplied or collected before claiming the finished feature meets its objective:

1. **Real user vocabulary:** category inputs from the actual use case, with expected IDs or explicit ambiguity/no-match labels.
2. **Trusted review:** a category owner or reviewer who can resolve labeling disagreements and approve aliases.
3. **Language scope:** confirmation that English covers the intended users; additional languages require a separately qualified release.
4. **Traffic and integration requirements:** the application that will call the API, expected simultaneous requests, and acceptable response time.
5. **Taxonomy authority:** whether this supplied snapshot remains the intended vocabulary or a reviewed Google update should become the next version.
6. **Acceptance of uncertainty responses:** a product flow that can show alternatives or ask for context instead of always forcing a category.

This document specifies the intended full application. The initial CLI prototype and
local SQLite/vector integration have since been implemented; broader service work,
trusted evaluation, qualified calibration, migrations, and release qualification remain.
See Sprint 2 status (historical) for the current implementation and
latest findings (historical) for measured diagnostic
results, remaining ranking regressions, and phase-specific resource observations.
