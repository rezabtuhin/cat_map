"""Compare NLI representations on frozen development candidates, never holdout data."""

import argparse
import json
import time
from pathlib import Path

import torch

from category_matcher.artifacts import digest
from category_matcher.reranker import CrossEncoderReranker
from category_matcher.taxonomy import read_google_taxonomy


def pairs_for(query, candidates, variant):
    query_text = f"These products are {query}."
    pairs = []
    for candidate in candidates:
        label = candidate["label"]
        if variant == "alternatives":
            label = label.replace(" & ", " or ")
        parent = candidate["path"].split(" > ")[-2:-1]
        if variant in {"context", "context_reverse"} and parent:
            label = f"{label} ({parent[0]})"
        if variant == "category":
            pair = (f"The product category is {query}.", f"The product category is {label}.")
        else:
            premise = query_text.replace(" & ", " or ") if variant == "alternatives" else query_text
            pair = (premise, f"These products are {label}.")
        if variant in {"reverse", "context_reverse"}:
            pair = pair[::-1]
        pairs.append(pair)
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["reverse", "context", "category"],
        choices=["reverse", "context", "category", "alternatives", "context_reverse"],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = Path("reports/sprint_2/improvement/retrieval_comparison.json")
    report = json.loads(baseline.read_text())
    selected = {r["example_id"]: r for r in report["examples"]["hybrid_rerank"]}
    categories, _ = read_google_taxonomy(Path("google_category_list.txt"))
    cats = {c.category_id: c for c in categories}
    rows = report["examples"]["hybrid"]
    if args.pilot:
        rows = [
            r
            for r in rows
            if r["expected_status"] == "matched"
            and selected[r["example_id"]]["first_relevant_rank"] != 1
        ]
    users = json.loads(Path("reports/sprint_2/improvement/user_examples.json").read_text())
    rows = [
        *rows,
        *[
            {
                "example_id": "user-" + str(i),
                "query": r["query"],
                "candidate_ids": [c["id"] for c in r["baseline"]["candidates"]],
                "acceptable_category_ids": r["expected_ids"],
                "expected_status": "matched",
                "actual_status": r["baseline"]["status"],
                "query_analysis": r["baseline"]["query_analysis"],
                "exact_fast_path": False,
            }
            for i, r in enumerate(users["examples"])
        ],
    ]
    metadata = json.loads(Path("models/nli_reranker_model.json").read_text())
    ranker = CrossEncoderReranker(metadata)
    output = []
    started = time.perf_counter()
    for i, row in enumerate(rows):
        ids = row["candidate_ids"]
        if row["actual_status"] != "provisional" or len(ids) < 2:
            continue
        candidates = [
            {"id": cid, "label": cats[cid].label, "path": cats[cid].full_path} for cid in ids[:100]
        ]
        scores = {}
        query = row["query"].split("|")[0]
        for variant in args.variants:
            logits = torch.as_tensor(
                ranker.model.predict(
                    pairs_for(query, candidates, variant),
                    batch_size=8,
                    show_progress_bar=False,
                    activation_fn=torch.nn.Identity(),
                )
            )
            margin = logits[:, 1] - torch.logsumexp(logits[:, [0, 2]], dim=1)
            scores[variant] = dict(zip(ids[:100], margin.tolist(), strict=True))
        output.append({**row, "scores": scores})
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(rows)} rows, {time.perf_counter() - started:.1f}s", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "baseline_sha256": digest(baseline),
                "label_status": "UNREVIEWED_DEVELOPMENT_ONLY",
                "pilot_failures_only": args.pilot,
                "variants": args.variants,
                "prefix_limit": 100,
                "ranking_model": metadata,
                "seconds": time.perf_counter() - started,
                "examples": output,
            },
            indent=2,
        )
        + "\n"
    )
    for row in output:
        print(row["example_id"], row["query"], flush=True)
        for variant, scores in row["scores"].items():
            model = sorted(scores, key=lambda cid: (-scores[cid], cid))
            ranks = {cid: i for i, cid in enumerate(model, 1)}
            fused = sorted(
                row["candidate_ids"],
                key=lambda cid: (
                    -(
                        1 / (10 + row["candidate_ids"].index(cid) + 1)
                        + (3 / (10 + ranks[cid]) if cid in ranks else 0)
                    ),
                    cid,
                ),
            )
            print(variant, [cats[cid].label for cid in fused[:3]], flush=True)


if __name__ == "__main__":
    main()
