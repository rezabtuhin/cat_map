# Category Matching: Sprint Execution Plan

Prepared: 26 October 2026 (Asia/Dhaka)

Reference: [Architecture and implementation plan](CATEGORY_MATCHING_IMPLEMENTATION_PLAN.md)

## 1. Immediate objective and execution order

**First prove that the Google category matcher works on real inputs.** The original execution order required completing Sprint 2 qualification before API integration. On 26 October 2026, the user explicitly requested Sprint 3 implementation while deferring human review and further matching tests. Sprint 3 therefore proceeds as a local development integration. This changes the development order; it does not pass the outstanding quality or resource gates. Multi-vendor support and larger architecture improvements remain later work.

This document sets the execution order for the current scope. The architecture document remains the technical reference; its full multi-vendor design is not a prerequisite for the first experiment.

The first validation must answer:

1. Can the system match category names that differ from the supplied list?
2. Does hybrid retrieval improve on simpler exact, fuzzy, and keyword baselines?
3. Does the selected cross-encoder improve category ranking?
4. Can the system avoid assigning categories to ambiguous or unsupported inputs?
5. Can the complete model path run acceptably on the existing M1 Mac with 8 GB RAM?

At planning time, the project contained only the category source and planning documents.
Implementation has now started. Track current work in
[Sprint 2 status](reports/sprint_2/STATUS.md) and
[latest measured diagnostic findings](reports/sprint_2/revision_3/FINDINGS.md).
The current Sprint 2 revision selects NLI entailment rank fusion after the initial
MiniLM comparison failed. The current revision reaches 80.0% top-1 on reused
draft non-exact inputs, guards weak promotions, and leaves unsupported store
section titles unmapped. Human-review export/apply and unlabeled-input capture
are implemented, with pending review packets ready.
Trusted label review, independent qualification, and remaining ranking regressions
still require work before qualifying the matcher for release. Resource evidence now separates idle/load/
inference swap observations and native process memory; the swap check remains
unpassed. See the current report for measurements and limits.

The [CPU quantization comparison](reports/sprint_2/quantization/FINDINGS.md) has
also completed. It does not support replacing float32: the dynamic-int8 profile
regresses on ranking, and embedding-only int8 does not lower the measured process
memory. Sprint 2 qualification remains pending; current integration work is tracked
in [Sprint 3 findings](reports/sprint_3/FINDINGS.md).
The sprint descriptions below define planned scope and acceptance criteria.

## 2. Sprint cadence and scope

Use five-working-day sprints. The allocation below assumes one developer and an available reviewer for category labels and ambiguous examples. These are planning estimates; environment issues, labeling availability, and failed quality gates can extend a sprint. There are no committed calendar dates.

| Sprint | Goal | Demonstrable output | Dependency |
|---|---|---|---|
| 1 | Build an executable retrieval prototype | CLI matcher over all 5,582 categories, with baseline evaluation | Existing source file |
| 2 | Validate matching quality and local feasibility | Reranking, uncertainty policy, untouched-test report, resource report | Sprint 1 |
| 3 | Expose the current matcher through a development API | Tested local API using the same matching service | Current frozen Sprint 2 bundle; user authorized early integration while qualification remains pending |
| 4 | Make builds and pilot operation repeatable | Versioned artifacts, migrations, feedback capture, recovery checks | Sprint 3 |
| 5 | Qualify the Google release | Independent qualification report and release decision | Sprint 4 and sufficient labeled pilot inputs |

**The first delivery milestone is the end of Sprint 2:** a runnable Google matcher plus evidence showing whether the selected approach meets the preliminary quality and hardware targets.

Keep these minimal foundations from the beginning: separate parsing, retrieval, reranking, and decision code; a canonical category record; string category IDs; a `google` taxonomy identifier; an artifact version; and deterministic configuration. Build only the interfaces needed by the current implementation.

Deferred work:

- Meta ingestion and other real vendor taxonomies.
- A generalized vendor registry, multi-parent source support, and cross-vendor conversion.
- Multilingual matching.
- Remote inference services, distributed queues, multi-host deployment, and database replacement.
- A frontend, accounts, authentication, and an administrative application.
- Custom model training or fine-tuning before measured errors justify it.

## 3. Sprint 1 — Executable prototype and retrieval evidence

### Goal

Run actual category queries against the complete Google list and compare retrieval methods. The initial interface is a command-line tool, so a user can test inputs without waiting for an API or frontend.

### Work items

| ID | Work | Verification |
|---|---|---|
| S1-01 | Create the uv project using the existing Python 3.11 runtime; lock dependencies | Frozen sync and required imports succeed |
| S1-02 | Load `sqlite-vec` through the application environment | Extension version query and known-vector nearest-neighbor check pass |
| S1-03 | Parse the source into canonical Google records and hierarchy links | Exactly 5,582 records, no duplicate IDs, no broken parents |
| S1-04 | Create the initial SQLite schema and source/model manifest | Source checksum and schema version are recorded |
| S1-05 | Implement conservative normalization, exact lookup, and a small reviewed alias set | Canonical IDs/paths round-trip; aliases resolve only to valid entries |
| S1-06 | Build BM25 and fuzzy candidate retrieval | Results include scores, source names, ranks, and valid IDs |
| S1-07 | Load `BAAI/bge-base-en-v1.5`; build label/path embeddings in SQLite | Exactly 11,164 normalized 768-dimensional embedding rows |
| S1-08 | Implement `sqlite-vec` semantic retrieval and RRF candidate fusion | Scoped SQL retrieval agrees with independent NumPy calculations |
| S1-09 | Build the CLI matcher and evaluation harness | One query command and one evaluation command produce reproducible results |
| S1-10 | Assemble and review 200 development examples | Expected categories/statuses are human-reviewed and versioned |

Use the candidate configuration from the architecture plan: BM25 50, semantic 50 distinct categories, fuzzy 20, fused shortlist 40, and at most five additional parents. Log shortlist recall after that final expansion.

For non-exact inputs, Sprint 1 returns **ranked candidates marked provisional**. It must not present uncalibrated similarity as a confirmed match or probability. Exact canonical lookups and approved aliases may resolve deterministically.

### Working-day allocation

| Day | Developer focus | Review/data focus |
|---|---|---|
| 1 | Environment, extension smoke check, parser, canonical records | Define labeling rules and gather real category inputs |
| 2 | SQLite import, normalization, exact/alias lookup | Review development labels and ambiguity cases |
| 3 | BM25/fuzzy retrieval and embedding build | Complete the 200 development examples |
| 4 | Semantic SQL retrieval, RRF, and CLI | Review ID/path and hierarchy examples |
| 5 | Run baseline evaluation, fix correctness defects, record findings | Adjudicate disputed development labels |

### Required artifacts

- Locked Python project and initial source code.
- Versioned SQLite taxonomy/vector snapshot and index manifest.
- CLI command that accepts a category and displays canonical candidate IDs and paths.
- `data/google/evaluation/development.jsonl` with 200 reviewed examples.
- `reports/sprint_1/retrieval_baseline.json` and a readable findings report.
- Focused ingestion, exact-lookup, vector-search, and retrieval-behavior tests.

The baseline report compares these paths on the same development examples:

1. Exact lookup plus reviewed aliases.
2. Exact/aliases plus fuzzy retrieval.
3. Exact/aliases plus BM25 retrieval.
4. Exact/aliases plus semantic retrieval.
5. Exact/aliases plus fused hybrid retrieval.

Report non-exact input results separately so canonical lookups cannot hide weak semantic retrieval. For ranking comparisons, use the same candidate-count budget and record candidate recall, top-1 accuracy, and top-3 recall.

### Sprint exit criteria

- A clean environment can ingest and query the complete source.
- All returned candidates exist in the supplied taxonomy, with original paths.
- All canonical ID/path round-trip checks pass.
- The 200-example development evaluation finishes and its errors are inspectable by stage.
- A user can enter a synonym, a typo, and a paraphrase and inspect the actual rankings.

Sprint 1 proves execution and establishes a baseline. It does not prove production accuracy. Low candidate recall must be investigated before acceptance thresholds are tuned.

## 4. Sprint 2 — Reranking, uncertainty, and proof-of-concept decision

### Goal

Determine whether the complete planned approach is accurate enough and practical enough to continue into application integration.

### Work items

| ID | Work | Verification |
|---|---|---|
| S2-01 | Complete the initial 600-example dataset | 200 development, 200 calibration, and 200 untouched test examples |
| S2-02 | Compare cross-encoder ranking; current profile uses `cross-encoder/nli-deberta-v3-small` rank fusion | Same candidate sets can be evaluated with and without reranking; keep the measured configuration |
| S2-03 | Add scope, parent/child, and ambiguity checks | Broad terms do not gain unsupported child qualifiers |
| S2-04 | Implement acceptance, clarification, no-match, and declared-language outcomes | Uncertain inputs remain unassigned |
| S2-05 | Calibrate score/margin/support thresholds on calibration data | Versioned policy records selected values and evidence |
| S2-06 | Freeze the pipeline and run the untouched test set | Report includes counts, slice metrics, and failure examples |
| S2-07 | Benchmark models and complete inference on the M1 Mac | Warm p50/p95, load time, peak process memory, and swapping are recorded |
| S2-08 | Produce the proof-of-concept decision report | Findings explicitly say proceed, revise, or locally infeasible |

### Dataset requirements

Across the initial 600 examples, preserve the architecture plan's coverage:

| Slice | Total examples |
|---|---:|
| Exact labels, full paths, and valid IDs | 100 |
| Synonyms and paraphrases | 140 |
| Typos and word-order changes | 100 |
| Broad versus specific hierarchy decisions | 100 |
| Ambiguous inputs | 80 |
| Out-of-taxonomy and unsupported-language inputs | 80 |

Sprint 1's 200 development records count toward these totals. Complete the other 400 in Sprint 2, adjusting development coverage only before the split is frozen. Keep near-duplicate phrases, alias families, and typo variants in one split. Include examples across all 21 top-level Google branches.

Do not place an alias in the implementation and then use a copy of that phrase as an independent held-out success. Prioritize actual intended-user vocabulary; mark manually authored examples separately. Model-generated labels cannot serve as ground truth without human review.

### Working-day allocation

| Day | Developer focus | Review/data focus |
|---|---|---|
| 1 | Complete evaluator and add reranker | Finish and freeze calibration/test labels and split membership |
| 2 | Compare hybrid retrieval with and without reranking on development data | Adjudicate remaining labeling disputes before evaluation |
| 3 | Implement specificity checks and calibrate uncertainty policy | Review ambiguity definitions; leave test outcomes unseen |
| 4 | Freeze configuration and evaluate untouched test data | Review resulting errors after the run |
| 5 | Benchmark hardware, document findings, and make the gate decision | Confirm interpretation of errors and user-facing outcomes |

If test results trigger implementation changes, those inspected examples are no longer an untouched test set. Obtain a fresh holdout before the next final decision; do not repeatedly tune against the same test report.

### Required artifacts

- CLI matcher returning a final status, canonical category or alternatives, reason code, and artifact version.
- Frozen development/calibration/test datasets totaling 600 reviewed examples.
- Versioned `decision_policy.json` and model revisions.
- `reports/sprint_2/quality_report.json` and readable error analysis.
- `reports/sprint_2/resource_report.json`.
- `reports/sprint_2/proof_of_concept_decision.md`.

### Preliminary gate for API work

These thresholds are **proof-of-concept targets on the small holdout**, not a production claim:

| Check | Required evidence to proceed |
|---|---|
| Canonical integrity | 100% valid output IDs/paths and correct canonical lookup |
| Candidate recall | At least 99.5% on resolvable holdout inputs, with raw counts shown |
| Automatic-match precision | At least 99% observed precision; report the confidence interval and sample size |
| Resolvable-input coverage | At least 80% automatically matched |
| Non-exact behavior | Separate synonym, typo, paraphrase, hierarchy, and model-path results; no reliance on exact matches alone |
| Ambiguity and no-match behavior | False acceptances counted as precision errors and reported by slice |
| Value of added model stages | Same-input comparison with simpler baselines; explain which errors hybrid retrieval and reranking fix or introduce |
| Local feasibility | Warm model-path p95 at most 3 seconds at one active request; process peak memory below 3 GB; no sustained service-induced swapping |

If a slice has no examples or no automatic matches, report that explicitly. Do not convert an undefined precision or coverage into a passing value.

A component that adds complexity without demonstrated quality benefit must be revised or excluded from the validated pilot configuration, with the reason recorded. Do not keep a harmful reranker simply because it was named in the initial architecture.

### Decision at sprint end

- **Proceed:** preliminary quality and resource gates pass. Begin Sprint 3.
- **Revise:** matching runs, but a quality target fails. Work on the failing stage using development/calibration data, then use a fresh holdout. Defer API and infrastructure expansion.
- **Locally infeasible:** the implementation cannot meet the resource target on this host after focused measurement and correction. Record the measured bottleneck and the required execution change before integration.

The user's subsequent authorization advances Sprint 3 development work despite
the original API entry gate. Preserve these thresholds for qualification; API
implementation or CLI/API agreement does not count as passing them. No draft
policy may enable automatic semantic mapping through the API.

Even a passing 200-example holdout cannot substantiate the full production precision claim. Sprint 5 provides independent qualification.

## 5. Sprint 3 — Local API and practical integration

### Goal

Expose the current frozen matcher through FastAPI while preserving its Sprint 2
behavior and its rule that uncertain inputs stay unmapped. This is development
integration authorized before the pending qualification work is complete.

**Implementation status:** S3-01 through S3-07 are complete for local development.
All 129 tests pass, including 27 API cases. Real HTTP checks agree with all 200
frozen CLI predictions and their alternatives; a ten-request burst admits nine
and rejects one with 503 while health remains responsive. See
[Sprint 3 findings and measurements](reports/sprint_3/FINDINGS.md). Qualification
remains pending; these checks establish integration behavior, not accuracy.

### Work items

| ID | Work | Verification |
|---|---|---|
| S3-01 | Wrap the matching service in validated request/response schemas | CLI and API produce equivalent semantic outcomes |
| S3-02 | Implement match and canonical-category lookup endpoints | Requests return canonical records from the selected Google bundle |
| S3-03 | Implement live/ready checks and startup artifact validation | Readiness fails for missing/incompatible artifacts or model-load failure |
| S3-04 | Reuse resident models; move inference off the async event loop | Health endpoints stay responsive during inference |
| S3-05 | Bound inference to one active job and eight pending requests | Queue saturation returns 503 without unbounded memory growth |
| S3-06 | Add structured outcome/stage timing logs and event IDs | A failed request can be traced to its stage and artifact version |
| S3-07 | Add API behavior tests and a minimal caller example | Invalid payloads and semantic outcomes follow the documented contract |

Preserve string IDs, `taxonomy_key: "google"`, and a resolved version in the API contract. Only Google is activated in this phase. Reject unimplemented taxonomy keys; a generalized registry and additional vendor adapters remain deferred.

Use the architecture plan's match, canonical lookup, and health routes. Feedback is added in Sprint 4. A minimal taxonomy-discovery response can list only the available Google version.

### Working-day allocation

- Days 1–2: schemas, endpoints, startup, and model lifecycle.
- Day 3: bounded executor, queue behavior, errors, and observability.
- Day 4: API equivalence and failure tests.
- Day 5: local caller demonstration and API resource benchmark.

### Exit criteria and deliverables

- Tested local API bound to `127.0.0.1`, running one worker.
- Example requests for a valid match, ambiguity, no match, bad input, and overload.
- CLI/API agreement on the frozen evaluation inputs.
- API overhead and concurrency measurements recorded in `reports/sprint_3/`.
- No unexplained quality regression relative to the frozen Sprint 2 configuration.

This is a local integration milestone. Public hosting, authentication, and internet exposure are outside this sprint.

## 6. Sprint 4 — Reproducible artifacts and reviewable pilot

### Goal

Make the working Google application reproducible, recoverable, and capable of collecting trustworthy feedback.

### Work items

| ID | Work | Verification |
|---|---|---|
| S4-01 | Add ordered, checksum-verified schema migrations | Initial creation and upgrade from the previous supported schema pass |
| S4-02 | Finalize bundle manifests and staging builds | Database, vectors, aliases, models, and policy cannot silently mismatch |
| S4-03 | Separate read-only category artifacts from writable runtime state | Feedback writes do not mutate the active taxonomy/vector bundle |
| S4-04 | Add feedback storage and the feedback endpoint | Confirmations/corrections link to the original input and bundle |
| S4-05 | Add alias review and controlled rebuild commands | Feedback never becomes an automatic alias without review |
| S4-06 | Verify bundle activation, drain/restart for model changes, and rollback | Requests never mix category/vector/policy generations |
| S4-07 | Add focused module-boundary and adapter checks | SQL/model execution remains outside API/domain rules |
| S4-08 | Run a reviewable pilot and prepare independent qualification inputs | Reviewer can inspect accepted and uncertain outcomes |

### Working-day allocation

- Days 1–2: migrations, artifact checks, reproducible builds, and runtime storage.
- Day 3: feedback endpoint and review workflow.
- Day 4: recovery, upgrade, and integrity verification.
- Day 5: pilot run and operational findings.

### Exit criteria and deliverables

- Documented build, run, migrate, review, and restore commands.
- A fresh checkout can reproduce the active configuration using the lockfile and pinned models.
- Bundle rollback and runtime-data backup/restore checks pass.
- Pilot feedback is captured with traceable IDs and resolved Google versions.
- Qualification data collection is underway, separate from policy-tuning data.
- `reports/sprint_4/` contains reproducibility and pilot-operation results.

Do not expand this sprint into a generic plugin system, distributed deployment, or Meta implementation.

## 7. Sprint 5 — Independent qualification and release decision

### Goal

Determine whether the Google feature meets the architecture plan's production quality and local-resource requirements.

### Entry requirement

At least 1,000 additional independently labeled, representative inputs must be available. These inputs are separate from the initial 600 and must not have been used to select aliases, thresholds, models, or candidate limits. Collection/review time may extend beyond a five-day sprint; do not replace this requirement with repeated or copied queries.

Development pilot data may inform improvements. Data designated for final qualification must stay held out. Freeze the pipeline before opening qualification outcomes.

### Work items

1. Review data quality and representative slice coverage before final evaluation.
2. Freeze the source checksum, aliases, model revisions, retrieval settings, policy, and dependencies.
3. Run qualification once and produce raw counts, slice metrics, and confidence bounds.
4. Recheck warm API latency, process memory, queue behavior, and inference failures on the intended host.
5. Verify the runbook and recovery procedure against the frozen release candidate.
6. Publish a release decision report stating which targets passed and failed.

### Final acceptance criteria

| Metric | Required result |
|---|---|
| Output ID/path integrity | 100% |
| Canonical ID/path lookup | 100% |
| Final candidate recall on resolvable inputs | At least 99.5% |
| Automatic-match precision | At least 99% observed, with a one-sided 95% exact binomial lower bound also at least 99% |
| Resolvable-input automatic coverage | At least 80% |
| Non-exact/model-path quality | Reported separately with adequate independent accepted samples |
| Warm model-path p95 | At most 3 seconds at one active request |
| Peak process resident memory | Below 3 GB, with no sustained service-induced swapping |
| Operational integrity | Rebuild, version checks, overload behavior, and rollback verified |

Precision includes incorrect acceptances of ambiguous and out-of-taxonomy inputs. Report absolute counts and denominators, not percentages alone. An overall precision result cannot conceal an inadequately evaluated model path.

The 99% confidence-bound requirement needs sufficient accepted samples: with zero errors, at least 299 independent accepted examples are necessary for the one-sided 95% lower bound to reach 99%. Errors increase the evidence required. A 1,000-input dataset does not automatically pass this requirement if too few inputs are accepted.

### Deliverables and sprint outcome

- `reports/sprint_5/qualification_report.json`.
- Readable precision/coverage, failure-slice, and resource analysis.
- Frozen release manifest and local runbook.
- An explicit **qualified** or **not qualified** decision for the tested Google configuration and traffic level.

If a target fails, return to the responsible stage, preserve the failure report, and obtain fresh qualification evidence after changes. Do not lower thresholds or relabel examples merely to obtain a passing report. This sprint does not automatically deploy or publish the service.

## 8. Stage-specific response to failures

| Observed failure | Next work |
|---|---|
| Correct category absent from the final shortlist | Fix ingestion, normalization, representation, or candidate retrieval before reranking |
| Correct category shortlisted but ranked below an incorrect one | Inspect candidate formatting, hierarchy evidence, and reranking |
| Correct top result rejected too often | Inspect calibration and specificity rules against the coverage requirement |
| Wrong or ambiguous result accepted | Tighten/rework decision evidence using calibration data; inspect unsupported qualifiers |
| Good aggregate results, weak synonym/typo slice | Improve that slice and gather representative labels; keep slice metrics visible |
| Model quality offers no gain over simpler retrieval | Record the comparison and simplify the validated configuration |
| Latency or memory exceeds the local target | Profile encoding, reranking, batch sizes, SQL retrieval, and queueing before changing execution hardware |
| Labels conflict or input lacks enough context | Adjudicate the labels or require clarification; do not force a category |

## 9. Reporting and definition of done

Each sprint ends with a runnable demonstration, a short results report, and a record of unresolved issues. Reports must identify the dataset split, source checksum, configuration/model revisions, commands run, sample counts, and actual pass/fail outcomes.

A sprint is complete when its listed artifacts exist and its exit criteria have been verified. Finishing the estimated number of days does not make a failing gate pass.

After Sprint 2, the user should be able to run the CLI on their own category inputs and inspect both its answer and measured limitations. Sprints 3–5 turn the validated experiment into an integrated, repeatable, independently qualified Google feature. Future vendor and architecture expansion resumes after this evidence exists.
