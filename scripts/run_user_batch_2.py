"""Record the real store labels; legacy outputs and AI proposals are not ground truth."""

import json
from dataclasses import asdict
from pathlib import Path

from category_matcher.artifacts import digest, resolve_bundle
from category_matcher.datasets import read_jsonl
from category_matcher.decision import DecisionPolicy
from category_matcher.matcher import Matcher


def main():
    folder = Path("reports/sprint_2/revision_3")
    dataset = Path("data/google/evaluation/user_examples_2.jsonl")
    matcher = Matcher(resolve_bundle(Path("artifacts/google")))
    policy = DecisionPolicy.load(
        folder / "draft_policy.json",
        artifact_version=matcher.manifest["artifact_version"],
        experimental=True,
    )
    examples = []
    for row in read_jsonl(dataset):
        baseline = matcher.match(row["query"], rerank=False)
        result = matcher.match(row["query"])
        ranked = asdict(result)
        preview = asdict(policy.apply(result))
        legacy = row["observed_legacy_mapping"]["category_id"]
        examples.append(
            {
                "example_id": row["example_id"],
                "query": row["query"],
                "label_review_status": row["review_status"],
                "proposed_expected_status": row["expected_status"],
                "proposed_expected_ids": row["acceptable_category_ids"],
                "reported_legacy_output": {
                    **row["observed_legacy_mapping"],
                    "canonical_path": matcher.store.categories[legacy].full_path,
                },
                "retrieval_only": asdict(baseline),
                "default_ranking": ranked,
                "experimental_policy_preview": preview,
            }
        )
    report = {
        "artifact_version": matcher.manifest["artifact_version"],
        "dataset_sha256": digest(dataset),
        "label_status": "REAL_INPUTS_WITH_UNREVIEWED_AI_LABEL_PROPOSALS",
        "notes": [
            "All five strings and legacy outputs were supplied by the user.",
            "They were inspected during development and cannot serve as an independent holdout.",
            "No product descriptions or confirmed category IDs were supplied.",
        ],
        "examples": examples,
    }
    (folder / "user_batch_2.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in examples:
        result = row["default_ranking"]
        print(
            row["query"],
            result["status"],
            [c["id"] for c in result["candidates"][:3]],
            result["reason_code"],
        )
    matcher.close()


if __name__ == "__main__":
    main()
