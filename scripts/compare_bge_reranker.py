"""Compare a new reranker on frozen development candidates, never test inputs."""

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

from category_matcher.reranker import CrossEncoderReranker
from category_matcher.taxonomy import read_google_taxonomy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/bge_reranker_model.json"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/sprint_2/improvement/bge_comparison.json")
    )
    args = parser.parse_args()
    baseline_path = Path("reports/sprint_2/retrieval_comparison.json")
    baseline = json.loads(baseline_path.read_text())
    categories, audit = read_google_taxonomy(Path("google_category_list.txt"))
    if audit["source_sha256"] != baseline["source_sha256"]:
        raise ValueError("Historical candidate source differs from the current taxonomy")
    by_id = {c.category_id: c for c in categories}
    metadata = json.loads(args.model.read_text())
    model = CrossEncoderReranker(metadata)
    output = []
    started = time.perf_counter()
    items = baseline["examples"]["hybrid"]
    for index, row in enumerate(items, 1):
        ids = row["candidate_ids"]
        if row["exact_fast_path"] or not ids or row["actual_status"] == "unsupported_language":
            ranked = ids
            elapsed = 0.0
        else:
            candidates = [
                {"id": cid, "label": by_id[cid].label, "path": by_id[cid].full_path, "rank": i}
                for i, cid in enumerate(ids[: args.limit], 1)
            ]
            before = time.perf_counter()
            ranked = [c["id"] for c in model.rank(row["query"], candidates)] + ids[args.limit :]
            elapsed = time.perf_counter() - before
        gold = row["acceptable_category_ids"]
        position = next((i for i, cid in enumerate(ranked, 1) if cid in gold), 0)
        output.append(
            {
                **row,
                "baseline_rank": row["first_relevant_rank"],
                "first_relevant_rank": position,
                "candidate_ids": ranked,
                "reranking_seconds": elapsed,
            }
        )
        if index % 10 == 0:
            print(
                f"Completed {index}/{len(items)} ({time.perf_counter() - started:.1f}s)", flush=True
            )
    non_exact = [
        row for row in output if row["expected_status"] == "matched" and not row["exact_fast_path"]
    ]
    timings = sorted(row["reranking_seconds"] for row in output if row["reranking_seconds"] > 0)
    report = {
        "source_baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "baseline_artifact_version": baseline["artifact_version"],
        "ranking_model": metadata,
        "reranking_prefix_limit": args.limit,
        "label_status": "unreviewed_draft",
        "non_exact_resolvable": len(non_exact),
        "baseline_top1_correct": sum(r["baseline_rank"] == 1 for r in non_exact),
        "reranked_top1_correct": sum(r["first_relevant_rank"] == 1 for r in non_exact),
        "reranked_top3_correct": sum(0 < r["first_relevant_rank"] <= 3 for r in non_exact),
        "reranking_p95_seconds": timings[int(0.95 * (len(timings) - 1))] if timings else None,
        "peak_resident_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2 if sys.platform == "darwin" else 1024),
        "examples": output,
        "notes": [
            "Uses the same frozen candidate IDs and unchanged draft labels as Sprint 2.",
            "Only the prefix is reranked; remaining candidates retain their prior order.",
            "This is model selection on development data, not independent validation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    main()
