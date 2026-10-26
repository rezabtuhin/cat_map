"""Select representation on development data and disclose all changed predictions."""

import json
from pathlib import Path

from category_matcher.artifacts import digest
from category_matcher.evaluation import summarize


def main():
    folder = Path("reports/sprint_2/specificity")
    path = folder / "representation_comparison.json"
    experiment = json.loads(path.read_text())
    previous = json.loads(
        Path("reports/sprint_2/improvement/retrieval_comparison.json").read_text()
    )
    old_rows = {r["example_id"]: r for r in previous["examples"]["hybrid_rerank"]}
    inputs = {r["example_id"]: r for r in experiment["examples"]}
    configs = [(v,) for v in experiment["variants"]]
    configs += [("category", "context"), ("category", "alternatives"), ("context", "alternatives")]
    results = []
    for variants in configs:
        for weight in (1, 2, 3):
            rows, changes, users = [], [], []
            for original in previous["examples"]["hybrid"]:
                row = dict(original)
                sample = inputs.get(row["example_id"])
                if sample:
                    row["candidate_ids"] = rank(sample, variants, weight)
                    row["first_relevant_rank"] = next(
                        (
                            i
                            for i, cid in enumerate(row["candidate_ids"], 1)
                            if cid in row["acceptable_category_ids"]
                        ),
                        0,
                    )
                if row["candidate_ids"][:1] != old_rows[row["example_id"]]["candidate_ids"][:1]:
                    changes.append(
                        {
                            "example_id": row["example_id"],
                            "query": row["query"],
                            "previous_rank": old_rows[row["example_id"]]["first_relevant_rank"],
                            "new_rank": row["first_relevant_rank"],
                            "new_top3": row["candidate_ids"][:3],
                        }
                    )
                rows.append(row)
            for key, sample in inputs.items():
                if key.startswith("user-"):
                    users.append(
                        {"query": sample["query"], "top3": rank(sample, variants, weight)[:3]}
                    )
            results.append(
                {
                    "representations": variants,
                    "total_model_weight": weight,
                    "non_exact": summarize([r for r in rows if not r["exact_fast_path"]]),
                    "changes": changes,
                    "user_candidates": users,
                }
            )
    results.sort(
        key=lambda r: (
            -r["non_exact"]["top1_correct"],
            -r["non_exact"]["top3_correct"],
            len(r["representations"]),
        )
    )
    (folder / "representation_selection.json").write_text(
        json.dumps(
            {
                "input_sha256": digest(path),
                "label_status": "UNREVIEWED_DEVELOPMENT_SELECTION",
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    for result in results:
        print(
            result["representations"],
            result["total_model_weight"],
            result["non_exact"]["top1_correct"],
            result["non_exact"]["top3_correct"],
            result["user_candidates"],
        )


def rank(sample, variants, weight):
    model_ranks = []
    for variant in variants:
        scores = sample["scores"][variant]
        ranked = sorted(scores, key=lambda cid: (-scores[cid], cid))
        model_ranks.append({cid: i for i, cid in enumerate(ranked, 1)})
    ids = sample["candidate_ids"]
    scores = {
        cid: 1 / (10 + i)
        + sum(weight / len(variants) / (10 + ranks[cid]) for ranks in model_ranks if cid in ranks)
        for i, cid in enumerate(ids, 1)
    }
    ranked = sorted(ids, key=lambda cid: (-scores[cid], cid))
    spelling = sample.get("query_analysis", {}).get("spelling", {})
    if spelling.get("eligible") and len(spelling.get("candidates", [])) == 1:
        ranked.sort(key=lambda cid: cid != spelling["candidates"][0])
    return ranked


if __name__ == "__main__":
    main()
