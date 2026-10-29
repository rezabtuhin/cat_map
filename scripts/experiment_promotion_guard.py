"""Diagnose model promotions using frozen development scores and live retrieval evidence."""

import json
from dataclasses import asdict
from pathlib import Path

from category_matcher.artifacts import digest, resolve_bundle
from category_matcher.evaluation import summarize
from category_matcher.matcher import Matcher


def main():
    folder = Path("reports/sprint_2/revision_3")
    matcher = Matcher(resolve_bundle(Path("artifacts/google")))
    source = Path("reports/sprint_2/specificity/retrieval_comparison.json")
    previous = json.loads(source.read_text())
    experiments = json.loads(
        Path("reports/sprint_2/specificity/representation_comparison.json").read_text()
    )
    scores = {row["example_id"]: row["scores"]["category"] for row in experiments["examples"]}
    rows = previous["examples"]["hybrid_rerank"]
    pending = [r for r in rows if r["actual_status"] == "provisional"]
    vectors = matcher.encode_queries([matcher.retrieval_query(r["query"]) for r in pending])
    evidence = {}
    for row, vector in zip(pending, vectors, strict=True):
        result = matcher.match(row["query"], query_vector=vector, rerank=False)
        evidence[row["example_id"]] = asdict(result)
    (folder / "retrieval_evidence.json").write_text(
        json.dumps(
            {
                "artifact_version": matcher.manifest["artifact_version"],
                "source_report_sha256": digest(source),
                "examples": evidence,
            },
            indent=2,
        )
        + "\n"
    )
    outputs = []
    for gap in (0.0, 0.01, 0.03, 0.05):
        for support in (0.0, 1.0, 2.0):
            changed, items = [], []
            for row in rows:
                row = dict(row)
                candidates = evidence.get(row["example_id"], {}).get("candidates", [])
                if candidates and row["candidate_ids"]:
                    old = row["candidate_ids"][0]
                    top = candidates[0]
                    by_id = {c["id"]: c for c in candidates}

                    def semantic(c):
                        return c.get("evidence", {}).get("semantic", {}).get("raw_score", -1)

                    difference = semantic(top) - semantic(by_id[old])
                    model_support = scores.get(row["example_id"], {}).get(old)
                    spelling = row["query_analysis"].get("spelling", {})
                    protected = (
                        spelling.get("eligible") and len(spelling.get("candidates", [])) == 1
                    )
                    if (
                        old != top["id"]
                        and difference > gap
                        and model_support is not None
                        and model_support < support
                        and not protected
                    ):
                        row["candidate_ids"] = [top["id"]] + [
                            c for c in row["candidate_ids"] if c != top["id"]
                        ]
                        rank = next(
                            (
                                i
                                for i, cid in enumerate(row["candidate_ids"], 1)
                                if cid in row["acceptable_category_ids"]
                            ),
                            0,
                        )
                        changed.append(
                            {
                                "query": row["query"],
                                "previous_rank": row["first_relevant_rank"],
                                "new_rank": rank,
                                "semantic_gap": difference,
                                "model_support": model_support,
                            }
                        )
                        row["first_relevant_rank"] = rank
                items.append(row)
            metrics = summarize([r for r in items if not r["exact_fast_path"]])
            outputs.append(
                {
                    "minimum_semantic_gap": gap,
                    "minimum_entailment_support": support,
                    "non_exact": metrics,
                    "changes": changed,
                }
            )
    (folder / "promotion_guard_experiment.json").write_text(
        json.dumps(
            {"purpose": "UNREVIEWED_DEVELOPMENT_SELECTION_ONLY", "results": outputs}, indent=2
        )
        + "\n"
    )
    for out in outputs:
        print(
            out["minimum_semantic_gap"],
            out["minimum_entailment_support"],
            out["non_exact"]["top1_correct"],
            out["changes"],
        )
    matcher.close()


if __name__ == "__main__":
    main()
