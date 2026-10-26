"""Build, retrieve, calibrate, qualify, and benchmark the local matcher."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from category_matcher.artifacts import build, load_manifest, resolve_bundle
from category_matcher.matcher import Matcher
from category_matcher.retrieval import METHODS


def policy_arguments(command):
    policies = command.add_mutually_exclusive_group()
    policies.add_argument("--policy", type=Path, help="Independently qualified policy")
    policies.add_argument("--experimental-policy", type=Path, help="Draft policy previews only")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--artifacts", type=Path, default=Path("artifacts/google"))
    commands = root.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build", help="Build a pinned offline taxonomy/vector bundle")
    builder.add_argument("--source", type=Path, default=Path("google_category_list.txt"))
    builder.add_argument("--aliases", type=Path, default=Path("data/google/aliases.jsonl"))
    builder.add_argument("--model", type=Path, default=Path("models/embedding_model.json"))
    commands.add_parser("inspect", help="Validate and display the active artifact manifest")
    commands.add_parser("verify", help="Check canonical lookups and SQLite vector equivalence")
    match = commands.add_parser("match", help="Return a canonical match or provisional candidates")
    match.add_argument("category", nargs="?", default="")
    match.add_argument("--id", dest="category_id")
    match.add_argument("--parent", dest="parent_id")
    match.add_argument("--context", default="")
    match.add_argument("--language", default="en")
    match.add_argument("--method", choices=METHODS, default="hybrid")
    match.add_argument("--limit", type=int, default=5)
    match.add_argument(
        "--rerank", action="store_true", help="Experimental reranking; still provisional"
    )
    match.add_argument("--reranker-model", type=Path, default=Path("models/reranker_model.json"))
    policy_arguments(match)
    evaluate = commands.add_parser("evaluate", help="Compare baselines on a labeled JSONL file")
    evaluate.add_argument(
        "--dataset", type=Path, default=Path("data/google/evaluation/development.jsonl")
    )
    evaluate.add_argument(
        "--output", type=Path, default=Path("reports/sprint_2/retrieval_comparison.json")
    )
    evaluate.add_argument(
        "--allow-draft",
        action="store_true",
        help="Diagnostic run only; never treat unreviewed labels as ground truth",
    )
    evaluate.add_argument("--rerank", action="store_true", help="Also compare hybrid + reranking")
    evaluate.add_argument("--reranker-model", type=Path, default=Path("models/reranker_model.json"))
    for name in ("calibrate", "qualify", "check-data"):
        command = commands.add_parser(name)
        command.add_argument(
            "--development", type=Path, default=Path("data/google/evaluation/development.jsonl")
        )
        command.add_argument(
            "--calibration",
            type=Path,
            default=Path("data/google/evaluation/calibration.draft.jsonl"),
        )
        if name in {"qualify", "check-data"}:
            command.add_argument(
                "--test", type=Path, default=Path("data/google/evaluation/test.draft.jsonl")
            )
        if name in {"calibrate", "qualify"}:
            command.add_argument("--allow-draft", action="store_true")
            command.add_argument(
                "--output", type=Path, default=Path(f"reports/sprint_2/{name}_policy.json")
            )
            command.add_argument(
                "--reranker-model", type=Path, default=Path("models/reranker_model.json")
            )
        if name == "calibrate":
            command.add_argument("--rerank", action="store_true")
        if name == "qualify":
            command.add_argument("--policy", type=Path, required=True)
    timing = commands.add_parser("benchmark")
    timing.add_argument(
        "--dataset", type=Path, default=Path("data/google/evaluation/development.jsonl")
    )
    timing.add_argument("--requests", type=int, default=200)
    timing.add_argument(
        "--output", type=Path, default=Path("reports/sprint_2/resource_report.json")
    )
    timing.add_argument("--rerank", action="store_true")
    timing.add_argument("--reranker-model", type=Path, default=Path("models/reranker_model.json"))
    policy_arguments(timing)
    return root


def main() -> int:
    args = parser().parse_args()
    matcher = None
    try:
        if args.command == "check-data":
            from category_matcher.datasets import check_splits

            report = check_splits(
                {
                    "development": args.development,
                    "calibration": args.calibration,
                    "test": args.test,
                }
            )
            print(json.dumps(report, indent=2))
            return 0 if report["passed"] else 2
        if args.command == "build":
            bundle, manifest = build(args.source, args.aliases, args.model, args.artifacts)
            print(json.dumps({"bundle": str(bundle), "manifest": manifest}, indent=2))
            return 0
        bundle = resolve_bundle(args.artifacts)
        if args.command == "inspect":
            print(json.dumps(load_manifest(bundle), indent=2))
            return 0
        wants_rerank = getattr(args, "rerank", False)
        if args.command == "qualify":
            wants_rerank = json.loads(args.policy.read_text())["ranking_method"] == "hybrid+rerank"
        if wants_rerank:
            from category_matcher.reranker import MiniLMReranker

            metadata = json.loads(args.reranker_model.read_text())
            matcher = Matcher(bundle, MiniLMReranker(metadata), metadata)
        else:
            matcher = Matcher(bundle)
        if args.command in {"match", "benchmark"}:
            from category_matcher.decision import DecisionPolicy

            policy_path = args.policy or args.experimental_policy
            if policy_path:
                matcher.policy = DecisionPolicy.load(
                    policy_path,
                    artifact_version=matcher.manifest["artifact_version"],
                    experimental=args.experimental_policy is not None,
                )
        if args.command == "match":
            if (
                not 1
                <= args.limit
                <= matcher.config["fusion_limit"] + matcher.config["parent_limit"]
            ):
                raise ValueError("Display limit exceeds the configured candidate pool")
            result = matcher.match(
                args.category,
                method=args.method,
                category_id=args.category_id,
                parent_id=args.parent_id,
                language=args.language,
                context=args.context,
                rerank=args.rerank,
            )
            result.candidates = result.candidates[: args.limit]
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        elif args.command in {"calibrate", "qualify"}:
            from category_matcher.calibration import calibrate, qualify
            from category_matcher.decision import write_policy

            if args.command == "calibrate":
                report = calibrate(
                    matcher,
                    args.calibration,
                    allow_draft=args.allow_draft,
                    development=args.development,
                    rerank=args.rerank,
                )
            else:
                report = qualify(
                    matcher,
                    args.policy,
                    args.test,
                    args.development,
                    args.calibration,
                    allow_draft=args.allow_draft,
                )
            saved = write_policy(args.output, report)
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "policy_id": saved["policy_id"],
                        "calibration": saved["calibration"],
                        "qualification": saved["qualification"],
                    },
                    indent=2,
                )
            )
        elif args.command == "benchmark":
            from category_matcher.benchmark import benchmark

            report = benchmark(matcher, args.dataset, args.requests, rerank=args.rerank)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
        elif args.command == "verify":
            from category_matcher.verification import verify

            print(json.dumps(verify(matcher), indent=2))
        elif args.command == "evaluate":
            from category_matcher.evaluation import evaluate

            report = evaluate(matcher, args.dataset, allow_draft=args.allow_draft)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "report": str(args.output),
                        "dataset_status": report["dataset_status"],
                        "methods": {
                            name: results["non_exact"]
                            for name, results in report["methods"].items()
                        },
                    },
                    indent=2,
                )
            )
        return 0
    except (ValueError, OSError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    finally:
        if matcher is not None:
            matcher.close()


if __name__ == "__main__":
    raise SystemExit(main())
