# CPU quantization experiment

**Decision: retain float32 as the default.** Full int8 quantization regresses on
development ranking. Word-embedding-only quantization preserves the observed
top-1 outcomes but does not reduce the measured process-memory peak or footprint.
Neither tested quantized profile passes the recorded resource gate.

The active artifact remains `eed281a45624da69-a0a29f98cc37`. Its model configuration,
stored vectors, draft policy, and acceptance behavior remain the selected baseline.
Uncertain inputs stay unmapped: candidates and experimental previews never become
automatic assignments. No labels or aliases were approved by this work.

## Implementation and controlled inputs

The [experimental converter](../../../src/category_matcher/quantization.py)
provides two QNNPACK CPU profiles using the installed, locked PyTorch 2.14.1:

- `int8_embeddings_v1`: row-wise 8-bit word-embedding weights. Linear layers and
  the directly accessed DeBERTa relative-position embedding remain float32.
- `int8_dynamic_v1`: the same word-embedding conversion plus dynamic int8 linear
  layers. Layer normalization and relative-position embeddings remain float32.

The APIs permit selecting modules and doing conversion in place. The embedding
configuration uses floating-point row scale/zero-point parameters.
[PyTorch quantize_dynamic documentation](https://docs.pytorch.org/docs/2.14/generated/torch.ao.quantization.quantize_dynamic.html),
[weight-only embedding qconfig](https://docs.pytorch.org/docs/2.14/generated/torch.ao.quantization.qconfig.float_qparams_weight_only_qconfig.html).
The installed API emits deprecation warnings; these are isolated experiments on
the existing locked environment, without a new production dependency or runtime
profile exposed by the matching CLI.

The full reranker conversion replaces 38 linear modules and one word embedding;
embedding-only replaces one word embedding. Compatibility tests also execute the
BGE encoder with quantized modules. The ranking experiment targets the reranker,
holding the float32 query encoder and stored document vectors fixed so candidate
retrieval is comparable. Encoder smoke tests do not qualify a quantized embedding
index or change the active vector space.

All ranking rules, candidate budgets, model revisions, templates, and source
labels are fixed. Each profile runs in a separate process. Ranking evaluation
uses the same 200 unreviewed development records and the ten existing real user
inputs. Calibration fitting and test predictions are excluded. Runtime overrides
are explicitly recorded in each experimental report; the baseline artifact ID
identifies the frozen store, not a deployed quantized bundle.

## Ranking and assignment results

| Reranker profile | Non-exact top-1 | Top-3 | Previously correct top-1 cases lost |
|---|---:|---:|---:|
| Float32 baseline | 96/120, 80.0% | 109/120, 90.8% | — |
| Word embeddings int8 | 96/120, 80.0% | 109/120, 90.8% | 0 |
| Word embeddings + dynamic int8 linear layers | 95/120, 79.2% | 107/120, 89.2% | 5 |

Full int8 improves four original-label cases but loses cat toys, bird cages,
children's toys, baby nappies, and fish food. This does not support replacing the
baseline for an accuracy-focused application. The word-only profile preserves
all previously correct top-1 outcomes and the aggregate shortlist metrics on
this reused development set. These are diagnostic model-selection results,
not independent accuracy evidence.

All ten user examples retain the same status, assigned category, and first
candidate. Collections, New Luma Yoga Collection, and Erin Recommends remain
unmapped. Jackets and Fitness Equipment remain provisional suggestions; they
are not automatically assigned. See [the decision and per-case changes](decision.json).

Quality reports: [word-only](embeddings/retrieval_comparison.json),
[full int8](dynamic/retrieval_comparison.json).
User checks: [word-only](embeddings/user_examples.json),
[full int8](dynamic/user_examples.json).

## Model size versus process memory

Counting only `model.parameters()` misses packed quantized weights. The new
diagnostic accounts for packed weights, bias tensors, and quantization metadata
as well as remaining float parameters. It estimates tensor storage, excluding
opaque kernel caches, framework overhead, and activations.

| Reranker profile | Estimated parameter tensor storage |
|---|---:|
| Float32 | 541.3 MiB |
| Word embeddings int8 | 260.8 MiB |
| Full int8 | 137.6 MiB |

That storage reduction is real, but it does not establish lower runtime memory.
The comparable fresh-process resource runs below execute the same 200 serial
model-path inputs, including model load/conversion in the cold measurements:

| Measurement | Float32 | Word embeddings int8 |
|---|---:|---:|
| Warm p50 | 760 ms | 758 ms |
| Warm p95 | 818 ms | 800 ms |
| Process peak resident memory | 1.08 GiB | 1.53 GiB |
| Physical footprint after warmup | 1.04 GiB | 1.50 GiB |
| Physical footprint after 200 requests | 1.03 GiB | 1.13 GiB |
| Model load/warmup | 7.43 s | 6.21 s |
| Observed system swapout-page increase | 21,280 | 10,260 |

Both pass latency and process-peak limits, but both fail the no-observed-swap
check. The smaller swap count is a system-wide observation across sequential
runs, without control over other applications. It does not prove a causal memory
improvement. The small latency difference also does not establish a repeatable
speedup. The embedding-only conversion raises the measured peak and final
footprint despite its smaller parameter tensors, so it does not meet the intended
memory-improvement objective.

The rejected full-int8 profile receives only a **30-request diagnostic** after
its ranking regression: p95 1,615 ms and peak resident memory 1.54 GiB. It is not
a 200-request resource qualification or a directly comparable overall p95.

Resource evidence: [float32 control](float32/resource_report.json),
[word-only, 200 requests](embeddings/resource_report.json),
[full int8, 30-request diagnostic](dynamic/resource_report.json).

## Conversion investigation and provenance

The initial module-compatibility tests included float inference before conversion
and diagnostic unpacking of quantized weights. Their memory counters are not the
fresh-process benchmark and must not be used as a profile comparison. Unpacking
diagnostics in the benchmark runner occur after timed measurements.

A follow-up [conversion probe](conversion_no_grad_probe.json) applies conversion
under `torch.no_grad()` and collects Python garbage before the first prediction.
It does not eliminate the elevated footprint. The original floating Parameter's
Python object is not retained, but that does not identify every native allocation
or prove all underlying storage was released. A macOS allocator-relief probe
returns zero bytes in its recorded run. No per-request garbage-collection change
or unsupported memory-leak explanation is shipped.

The [implementation snapshot](initial_implementation.json) captures the
pre-cleanup converter and runner after formatting. The initial runs did not
record a code hash at each process start. The final
experimental helper explicitly disables gradients during conversion; the
follow-up probe records that cleanup investigation separately. New runner
summaries record converter/script hashes, reject mid-run code changes, and
refuse to overwrite existing experiment
evidence. The converter is not part of the active matching runtime.

## Tests and remaining gate

Tests cover numerical forward compatibility, finite output, packed-weight byte
accounting, preservation of relative-position tensors, in-place conversion,
invalid profiles, and refusing a second conversion. The existing unmapped
contract is checked against all ten real inputs in both model profiles.
All 102 tests pass, with five deprecation warnings from the installed experimental
PyTorch API. Ruff lint/format and diff whitespace checks pass. The
[validation receipt](validation.json) records artifact integrity, unchanged source
and data checksums, 405 pending review decisions, and overwrite protection.

There is no model/default switch, artifact rebuild, policy refit, human approval,
or test-set evaluation required for this keep-baseline decision. Existing review
packets remain pending. Independent reviewed accuracy evidence and the baseline
resource qualification remain unresolved.

An eventual production quantization change needs a measured cold-load strategy
and a supported runtime implementation that improves the whole process, followed
by quality comparison and artifact/policy binding. Smaller checkpoint tensors
alone are insufficient evidence to adopt it.

## Reproduce without overwriting evidence

Use a new output directory for each experiment/action. For example:

```bash
uv run python scripts/experiment_quantized_reranker.py \
  --profile int8_embeddings_v1 --action evaluate \
  --output-dir reports/local/embedding-int8
uv run python scripts/experiment_quantized_reranker.py \
  --profile int8_embeddings_v1 --action benchmark --requests 200 \
  --output-dir reports/local/embedding-int8
```

The float32 control uses `--profile float32`; full int8 uses
`--profile int8_dynamic_v1`. The runner uses development data only and never
activates a quantized bundle or loads a decision policy. Review the existing
[input-review workflow](../../../data/google/reviews/README.md) before claiming
accuracy or enabling semantic auto-assignment.
