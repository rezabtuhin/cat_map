"""Development-only rank-fusion comparison using saved model predictions."""

import json
from pathlib import Path

from category_matcher.artifacts import resolve_bundle
from category_matcher.query import unsupported_audience_ids
from category_matcher.retrieval import LexicalRetriever
from category_matcher.taxonomy import read_google_taxonomy


def main():
    directory = Path("reports/sprint_2/improvement")
    rules = json.loads((directory / "rules_comparison.json").read_text())["examples"]["hybrid"]
    nli = json.loads((directory / "nli_comparison.json").read_text())
    predictions = {row["example_id"]: row for row in nli["examples"]}
    categories, _ = read_google_taxonomy(Path("google_category_list.txt"))
    cats = {c.category_id: c for c in categories}
    bundle = resolve_bundle(Path("artifacts/google"))
    lexical = LexicalRetriever(
        bundle / "bm25", cats, json.loads((bundle / "aliases.json").read_text())
    )
    spelling = {
        row["example_id"]: lexical.spelling(
            row["query"], set(cats) - unsupported_audience_ids(row["query"], cats), 5
        )
        for row in rules
        if row["query"]
    }
    scores = []
    for weight in (0.25, 0.5, 1.0, 2.0, 3.0):
        for constant in (10, 60):
            correct = count = top3 = 0
            for row in rules:
                if row["expected_status"] != "matched" or row["exact_fast_path"]:
                    continue
                count += 1
                original = row["candidate_ids"]
                if len(original) <= 1:
                    ranked = original
                else:
                    predicted = predictions[row["example_id"]]["candidate_ids"][
                        : nli["reranking_prefix_limit"]
                    ]
                    ranks = {cid: i for i, cid in enumerate(predicted, 1)}
                    total = {
                        cid: 1 / (constant + i)
                        + (weight / (constant + ranks[cid]) if cid in ranks else 0)
                        for i, cid in enumerate(original, 1)
                    }
                    ranked = sorted(total, key=lambda cid: (-total[cid], cid))
                    hint = spelling.get(row["example_id"], {})
                    if hint.get("eligible") and len(hint["candidates"]) == 1:
                        ranked.sort(key=lambda cid: cid != hint["candidates"][0])
                correct += bool(ranked and ranked[0] in row["acceptable_category_ids"])
                top3 += bool(set(ranked[:3]) & set(row["acceptable_category_ids"]))
            scores.append(
                {
                    "entailment_weight": weight,
                    "rank_constant": constant,
                    "top1_correct": correct,
                    "top3_correct": top3,
                    "denominator": count,
                }
            )
    (directory / "entailment_fusion_ablation.json").write_text(
        json.dumps(
            {
                "purpose": "Draft development selection; confirm with end-to-end execution",
                "results": scores,
            },
            indent=2,
        )
        + "\n"
    )
    for score in sorted(scores, key=lambda item: (-item["top1_correct"], -item["top3_correct"])):
        print(score)


if __name__ == "__main__":
    main()
