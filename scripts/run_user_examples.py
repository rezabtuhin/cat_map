"""Save an inspectable comparison for the five inputs supplied in this conversation."""

import argparse
import json
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path

from category_matcher.artifacts import resolve_bundle
from category_matcher.matcher import Matcher
from category_matcher.reranker import MiniLMReranker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/sprint_2/user_examples.json"))
    args = parser.parse_args()
    metadata = json.loads(Path("models/reranker_model.json").read_text())
    started = time.perf_counter()
    matcher = Matcher(resolve_bundle(Path("artifacts/google")), MiniLMReranker(metadata), metadata)
    matcher.encode_queries(["warmup category"])
    load_seconds = time.perf_counter() - started
    records = [
        json.loads(line)
        for line in Path("data/google/evaluation/user_examples.jsonl").read_text().splitlines()
    ]
    details = []
    try:
        for row in records:
            started = time.perf_counter()
            vector = matcher.encode_queries([matcher.retrieval_query(row["query"])])[0]
            encode_ms = (time.perf_counter() - started) * 1000
            baseline = matcher.match(row["query"], query_vector=vector)
            started = time.perf_counter()
            ranked = matcher.match(row["query"], query_vector=vector, rerank=True)
            rank_ms = (time.perf_counter() - started) * 1000
            details.append(
                {
                    "query": row["query"],
                    "review_status": row["review_status"],
                    "expected_ids": row["acceptable_category_ids"],
                    "explicitly_rejected_ids": row.get("rejected_category_ids", []),
                    "query_encoding_ms": encode_ms,
                    "retrieval_plus_reranking_ms": rank_ms,
                    "baseline": asdict(baseline),
                    "experimental_reranked": asdict(ranked),
                }
            )
            print(
                row["query"],
                "=>",
                ranked.status,
                [(candidate["id"], candidate["label"]) for candidate in ranked.candidates[:3]],
            )
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report = {
            "artifact_version": matcher.manifest["artifact_version"],
            "ranking_model": metadata,
            "load_and_warmup_seconds": load_seconds,
            "process_peak_resident_mib": peak / (1024**2 if sys.platform == "darwin" else 1024),
            "notes": [
                "User-confirmed mappings are regression/training examples, not a holdout.",
                "Three positive labels are drafts; only the baby-tops rejection is confirmed.",
                "Reranking is experimental and does not authorize semantic auto-acceptance.",
                "Five inputs are insufficient for a latency or quality release claim.",
            ],
            "examples": details,
        }
        output = args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    finally:
        matcher.close()


if __name__ == "__main__":
    main()
