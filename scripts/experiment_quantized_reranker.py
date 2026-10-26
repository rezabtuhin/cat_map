"""Compare CPU profiles on development data without changing the active bundle."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from category_matcher.artifacts import digest, load_manifest, resolve_bundle
from category_matcher.benchmark import benchmark
from category_matcher.datasets import read_jsonl
from category_matcher.evaluation import evaluate
from category_matcher.matcher import Matcher
from category_matcher.memory import process_memory
from category_matcher.quantization import PROFILES, apply_profile, tensor_bytes
from category_matcher.reranker import CrossEncoderReranker


class LazyProfileReranker:
    def __init__(self, metadata, verbose=False):
        self.metadata = metadata
        self.adapter = None
        self.conversion = None
        self.conversion_seconds = None
        self.requests = 0
        self.verbose = verbose

    @property
    def model(self):
        return self.adapter.model if self.adapter is not None else None

    def rank(self, query, candidates):
        if self.adapter is None:
            self.adapter = CrossEncoderReranker(self.metadata)
            started = time.perf_counter()
            self.conversion = apply_profile(
                self.adapter.model.model, self.metadata["inference_profile"]
            )
            self.conversion_seconds = time.perf_counter() - started
        result = self.adapter.rank(query, candidates)
        self.requests += 1
        if self.verbose and self.requests % 20 == 0:
            print(f"Completed {self.requests} model-path requests", flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--action", choices=["evaluate", "benchmark"], required=True)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_name = (
        "retrieval_comparison.json" if args.action == "evaluate" else "resource_report.json"
    )
    if (args.output_dir / report_name).exists():
        raise SystemExit(
            "Refusing to overwrite existing experiment evidence; use a new output directory"
        )
    implementation_path = Path(__file__).parents[1] / "src/category_matcher/quantization.py"
    implementation_hash = digest(implementation_path)
    script_hash = digest(Path(__file__))
    bundle = resolve_bundle(Path("artifacts/google"))
    manifest = load_manifest(bundle)
    metadata = {**manifest["ranking_model"], "inference_profile": args.profile}
    reranker = LazyProfileReranker(metadata, verbose=args.action == "evaluate")
    matcher = Matcher(bundle, reranker=reranker, reranker_metadata=metadata)
    dataset = Path("data/google/evaluation/development.jsonl")
    if args.action == "evaluate":
        report = evaluate(matcher, dataset, allow_draft=True)
        (args.output_dir / "retrieval_comparison.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        users = []
        for path in ("user_examples.jsonl", "user_examples_2.jsonl"):
            for row in read_jsonl(Path("data/google/evaluation") / path):
                result = matcher.match(row["query"], context=row.get("context", ""))
                assert result.status == "matched" or result.category is None
                users.append(
                    {
                        "example_id": row["example_id"],
                        "query": row["query"],
                        "review_status": row["review_status"],
                        "result": asdict(result),
                    }
                )
        (args.output_dir / "user_examples.json").write_text(
            json.dumps(
                {
                    "baseline_artifact_version": manifest["artifact_version"],
                    "experimental_ranking_model": metadata,
                    "examples": users,
                },
                indent=2,
            )
            + "\n"
        )
        result_summary = report["methods"]["hybrid_rerank"]["non_exact"]
    else:
        report = benchmark(matcher, dataset, args.requests, idle_baseline_seconds=5)
        (args.output_dir / "resource_report.json").write_text(json.dumps(report, indent=2) + "\n")
        result_summary = {
            key: report[key]
            for key in (
                "end_to_end_warm_latency_ms",
                "process_peak_resident_bytes",
                "resource_targets_passed",
            )
        }
    summary = {
        "baseline_artifact_version": manifest["artifact_version"],
        "quantization_implementation_sha256": implementation_hash,
        "experiment_script_sha256": script_hash,
        "runtime_override": metadata,
        "dataset_sha256": digest(dataset),
        "action": args.action,
        "conversion": reranker.conversion,
        "conversion_seconds": reranker.conversion_seconds,
        "reranker_tensors": tensor_bytes(reranker.model.model) if reranker.model else None,
        "process_memory_after_tensor_diagnostics": process_memory(),
        "result": result_summary,
        "notes": [
            "Experimental runtime override; active artifact and default config are unchanged.",
            "Encoder and stored document embeddings remain float32 for this reranker experiment.",
            "No test examples, calibration fitting, or human approvals are used here.",
            "Tensor unpacking diagnostics occur after the benchmark, outside timed requests.",
        ],
    }
    if digest(implementation_path) != implementation_hash or digest(Path(__file__)) != script_hash:
        raise RuntimeError(
            "Experiment code changed during execution; repeat in a new output directory"
        )
    (args.output_dir / f"{args.action}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    matcher.close()


if __name__ == "__main__":
    main()
