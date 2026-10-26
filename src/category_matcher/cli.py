"""Sprint 1 commands: build, inspect, match, verify, and evaluate."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from category_matcher.artifacts import build, load_manifest, resolve_bundle
from category_matcher.matcher import Matcher
from category_matcher.retrieval import METHODS


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
    evaluate = commands.add_parser("evaluate", help="Compare baselines on a labeled JSONL file")
    evaluate.add_argument(
        "--dataset", type=Path, default=Path("data/google/evaluation/development.jsonl")
    )
    evaluate.add_argument(
        "--output", type=Path, default=Path("reports/sprint_1/retrieval_baseline.json")
    )
    evaluate.add_argument(
        "--allow-draft",
        action="store_true",
        help="Diagnostic run only; never treat unreviewed labels as ground truth",
    )
    evaluate.add_argument("--rerank", action="store_true", help="Also compare hybrid + reranking")
    evaluate.add_argument("--reranker-model", type=Path, default=Path("models/reranker_model.json"))
    return root


def main() -> int:
    args = parser().parse_args()
    matcher = None
    try:
        if args.command == "build":
            bundle, manifest = build(args.source, args.aliases, args.model, args.artifacts)
            print(json.dumps({"bundle": str(bundle), "manifest": manifest}, indent=2))
            return 0
        bundle = resolve_bundle(args.artifacts)
        if args.command == "inspect":
            print(json.dumps(load_manifest(bundle), indent=2))
            return 0
        if getattr(args, "rerank", False):
            from category_matcher.reranker import MiniLMReranker

            metadata = json.loads(args.reranker_model.read_text())
            matcher = Matcher(bundle, MiniLMReranker(metadata), metadata)
        else:
            matcher = Matcher(bundle)
        if args.command == "match":
            if not 1 <= args.limit <= 45:
                raise ValueError("Display limit must be between 1 and 45")
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
