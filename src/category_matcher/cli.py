"""Build, retrieve, calibrate, qualify, and benchmark the local matcher."""

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

from category_matcher.artifacts import digest
from category_matcher.bundles import activate, load_active, rollback, stage_build
from category_matcher.matcher import Matcher
from category_matcher.retrieval import METHODS


def policy_arguments(command):
    policies = command.add_mutually_exclusive_group()
    policies.add_argument("--policy", type=Path, help="Independently qualified policy")
    policies.add_argument("--experimental-policy", type=Path, help="Draft policy previews only")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--artifacts", type=Path, default=Path("artifacts/google"))
    root.add_argument("--runtime-db", type=Path, default=Path("var/category_matcher.sqlite3"))
    commands = root.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build", help="Build a pinned offline taxonomy/vector bundle")
    builder.add_argument("--source", type=Path, default=Path("google_category_list.txt"))
    builder.add_argument("--aliases", type=Path, default=Path("data/google/aliases.jsonl"))
    builder.add_argument("--model", type=Path, default=Path("models/embedding_model.json"))
    builder.add_argument(
        "--ranking-profile", type=Path, default=Path("config/category_ranking.json")
    )
    builder.add_argument("--retrieval-only", action="store_true")
    builder.add_argument(
        "--activate", action="store_true", help="Activate after staging; API must be stopped"
    )
    activation = commands.add_parser(
        "activate", help="Pin a bundle/models/policy release with the API stopped"
    )
    selection = activation.add_mutually_exclusive_group(required=True)
    selection.add_argument("--bundle")
    selection.add_argument("--release-id")
    activation.add_argument("--policy", type=Path)
    undo = commands.add_parser(
        "rollback", help="Activate the previous retained release; API must be stopped"
    )
    undo.add_argument("--release-id")
    commands.add_parser("runtime-migrate", help="Create/upgrade the separate runtime database")
    for name in (
        "runtime-backup",
        "runtime-restore",
        "feedback-export",
        "aliases-apply",
        "pilot-export",
    ):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name == "runtime-restore":
            command.add_argument("--backup", type=Path, required=True)
        if name == "aliases-apply":
            command.add_argument("--reviews", type=Path, required=True)
            command.add_argument("--aliases", type=Path, default=Path("data/google/aliases.jsonl"))
            command.add_argument("--taxonomy", type=Path, default=Path("google_category_list.txt"))
    for name in ("capture-inputs", "review-export", "review-apply"):
        command = commands.add_parser(name)
        command.add_argument("--taxonomy", type=Path, default=Path("google_category_list.txt"))
        command.add_argument("--output", type=Path, required=True)
        if name == "capture-inputs":
            command.add_argument("--inputs", type=Path, required=True)
            command.add_argument("--source-name", required=True)
        else:
            command.add_argument("--dataset", type=Path, required=True)
        if name == "review-apply":
            command.add_argument("--reviews", type=Path, required=True)
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
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use the artifact's ranking model; --no-rerank selects retrieval only",
    )
    match.add_argument("--reranker-model", type=Path)
    policy_arguments(match)
    evaluate = commands.add_parser("evaluate", help="Compare baselines on a labeled JSONL file")
    evaluate.add_argument(
        "--dataset", type=Path, default=Path("data/google/evaluation/development.jsonl")
    )
    evaluate.add_argument("--output", type=Path, default=Path("reports/retrieval_comparison.json"))
    evaluate.add_argument(
        "--allow-draft",
        action="store_true",
        help="Diagnostic run only; never treat unreviewed labels as ground truth",
    )
    evaluate.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None)
    evaluate.add_argument("--reranker-model", type=Path)
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
            command.add_argument("--output", type=Path, default=Path(f"reports/{name}_policy.json"))
            command.add_argument("--reranker-model", type=Path)
        if name == "calibrate":
            command.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None)
        if name == "qualify":
            command.add_argument("--policy", type=Path, required=True)
    timing = commands.add_parser("benchmark")
    timing.add_argument(
        "--dataset", type=Path, default=Path("data/google/evaluation/development.jsonl")
    )
    timing.add_argument("--requests", type=int, default=200)
    timing.add_argument("--idle-baseline-seconds", type=float, default=0)
    timing.add_argument("--output", type=Path, default=Path("reports/resource_report.json"))
    timing.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None)
    timing.add_argument("--reranker-model", type=Path)
    policy_arguments(timing)
    return root


def main() -> int:
    args = parser().parse_args()
    matcher = None
    try:
        if args.command in {"activate", "rollback"}:
            if args.command == "activate":
                if args.release_id and args.policy:
                    raise ValueError("A retained release already pins its policy")
                report = activate(
                    args.artifacts, args.bundle, policy=args.policy, release_id=args.release_id
                )
            else:
                report = rollback(args.artifacts, args.release_id)
            print(json.dumps(report, indent=2))
            return 0
        if args.command in {
            "runtime-migrate",
            "runtime-backup",
            "runtime-restore",
            "feedback-export",
            "aliases-apply",
            "pilot-export",
        }:
            from category_matcher.alias_review import (
                apply_alias_reviews,
                export_feedback,
                export_pilot_inputs,
            )
            from category_matcher.runtime import RuntimeStore, restore

            if args.runtime_db.resolve().is_relative_to(args.artifacts.resolve()):
                raise ValueError("Runtime database must be outside artifact storage")
            if hasattr(args, "output") and args.output.resolve().is_relative_to(
                args.artifacts.resolve()
            ):
                raise ValueError("Runtime outputs must be outside artifact storage")
            store = RuntimeStore(args.runtime_db)
            if args.command == "runtime-migrate":
                store.initialize()
                report = {"runtime_db": str(store.path), "schema_version": 2}
            elif args.command == "runtime-backup":
                report = store.backup(args.output)
            elif args.command == "runtime-restore":
                report = restore(args.backup, args.output)
            elif args.command == "feedback-export":
                report = export_feedback(store, args.output)
            elif args.command == "pilot-export":
                report = export_pilot_inputs(store, args.output)
            else:
                report = apply_alias_reviews(
                    store, args.reviews, args.aliases, args.taxonomy, args.output
                )
            print(json.dumps(report, indent=2))
            return 0
        if args.command in {"capture-inputs", "review-export", "review-apply"}:
            from category_matcher.review import apply_reviews, capture_inputs, export_reviews

            if args.command == "capture-inputs":
                report = capture_inputs(args.inputs, args.taxonomy, args.output, args.source_name)
            elif args.command == "review-export":
                report = export_reviews(args.dataset, args.taxonomy, args.output)
            else:
                report = apply_reviews(args.dataset, args.taxonomy, args.reviews, args.output)
            print(json.dumps(report, indent=2))
            return 0
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
            bundle, manifest, _ = stage_build(
                args.source,
                args.aliases,
                args.model,
                args.artifacts,
                None if args.retrieval_only else args.ranking_profile,
            )
            report = {
                "bundle": str(bundle),
                "artifact_version": manifest["artifact_version"],
                "activated": args.activate,
            }
            if args.activate:
                report["release"] = activate(args.artifacts, bundle.name)
            print(json.dumps(report, indent=2))
            return 0
        bundle, manifest, release_policy = load_active(args.artifacts)
        if args.command == "inspect":
            print(json.dumps(manifest, indent=2))
            return 0
        wants_rerank = getattr(args, "rerank", None)
        override = getattr(args, "reranker_model", None)
        if wants_rerank is None:
            wants_rerank = override is not None or manifest.get("ranking_model") is not None
            if args.command == "match" and args.method != "hybrid":
                wants_rerank = False
        if args.command == "qualify":
            wants_rerank = json.loads(args.policy.read_text())["ranking_method"] == "hybrid+rerank"
        metadata = None
        if wants_rerank:
            metadata = (
                json.loads(override.read_text()) if override else manifest.get("ranking_model")
            )
            if metadata is None:
                metadata = json.loads(Path("models/reranker_model.json").read_text())
        args.rerank = wants_rerank
        matcher = Matcher(bundle, reranker_metadata=metadata)
        if args.command in {"match", "benchmark"}:
            from category_matcher.decision import DecisionPolicy

            policy_path = args.policy or args.experimental_policy or release_policy
            if policy_path:
                if (
                    args.policy
                    and "release_id" in json.loads((args.artifacts / "active.json").read_text())
                    and (release_policy is None or digest(args.policy) != digest(release_policy))
                ):
                    raise ValueError(
                        "Activate the release with this policy instead of overriding it"
                    )
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

            report = benchmark(
                matcher,
                args.dataset,
                args.requests,
                rerank=args.rerank,
                idle_baseline_seconds=args.idle_baseline_seconds,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
        elif args.command == "verify":
            from category_matcher.verification import verify

            print(json.dumps(verify(matcher), indent=2))
        elif args.command == "evaluate":
            from category_matcher.evaluation import evaluate

            report = evaluate(
                matcher, args.dataset, allow_draft=args.allow_draft, include_reranking=wants_rerank
            )
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
    except (ValueError, OSError, KeyError, sqlite3.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    finally:
        if matcher is not None:
            matcher.close()


if __name__ == "__main__":
    raise SystemExit(main())
