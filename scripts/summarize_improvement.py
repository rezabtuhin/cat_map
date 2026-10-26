"""Summarize saved development results without running or inspecting test predictions."""

import copy
import json
from pathlib import Path

from category_matcher.artifacts import digest
from category_matcher.evaluation import summarize
from category_matcher.taxonomy import read_google_taxonomy


def main():
    directory = Path("reports/sprint_2/improvement")
    original = Path("data/google/evaluation/development.jsonl")
    audited = Path("data/google/evaluation/development.audited.draft.jsonl")
    report = json.loads((directory / "retrieval_comparison.json").read_text())
    if report["dataset_sha256"] != digest(original):
        raise ValueError("Original development inputs changed since prediction")
    originals = {r["example_id"]: r for r in map(json.loads, original.read_text().splitlines())}
    proposals = {r["example_id"]: r for r in map(json.loads, audited.read_text().splitlines())}
    if originals.keys() != proposals.keys():
        raise ValueError("Audit changed the example cohort")
    allowed = {"acceptable_category_ids", "expected_status", "audit_reason", "audit_status"}
    for key, row in originals.items():
        proposed = proposals[key]
        if {k: v for k, v in row.items() if k not in allowed} != {
            k: v for k, v in proposed.items() if k not in allowed
        }:
            raise ValueError("Audit changed prediction inputs or review provenance")
    derived = {}
    for method in ("hybrid", "hybrid_rerank"):
        rows = copy.deepcopy(report["examples"][method])
        for row in rows:
            proposal = proposals[row["example_id"]]
            row["expected_status"] = proposal["expected_status"]
            row["acceptable_category_ids"] = proposal["acceptable_category_ids"]
            row["first_relevant_rank"] = next(
                (
                    i
                    for i, cid in enumerate(row["candidate_ids"], 1)
                    if cid in row["acceptable_category_ids"]
                ),
                0,
            )
        derived[method] = summarize([row for row in rows if not row["exact_fast_path"]])
    (directory / "audit_rescore.json").write_text(
        json.dumps(
            {
                "label_status": "AI_AUDIT_PROPOSALS_NOT_HUMAN_REVIEWED",
                "prediction_report_sha256": digest(directory / "retrieval_comparison.json"),
                "original_dataset_sha256": digest(original),
                "proposed_dataset_sha256": digest(audited),
                "notes": [
                    "Diagnostic using saved predictions and six proposed label changes.",
                    "Original-label results remain the primary comparison; no holdout was scored.",
                ],
                "non_exact": derived,
            },
            indent=2,
        )
        + "\n"
    )
    categories, _ = read_google_taxonomy(Path("google_category_list.txt"))
    paths = {c.category_id: c.full_path for c in categories}
    failures = []
    for row in report["examples"]["hybrid_rerank"]:
        if row["expected_status"] != "matched" or row["first_relevant_rank"] == 1:
            continue
        failures.append(
            {
                "example_id": row["example_id"],
                "query": row["query"],
                "draft_expected_paths": {cid: paths[cid] for cid in row["acceptable_category_ids"]},
                "actual_top3_paths": {cid: paths[cid] for cid in row["candidate_ids"][:3]},
                "first_relevant_rank": row["first_relevant_rank"],
                "audit_proposed_status": proposals[row["example_id"]]["expected_status"],
            }
        )
    (directory / "remaining_failures.json").write_text(
        json.dumps(
            {
                "artifact_version": report["artifact_version"],
                "label_status": "UNREVIEWED_DRAFT_DIAGNOSTIC_ONLY",
                "examples": failures,
            },
            indent=2,
        )
        + "\n"
    )
    resource = json.loads((directory / "resource_report.json").read_text())
    policy = json.loads((directory / "draft_policy.json").read_text())
    decision = {
        "artifact_version": report["artifact_version"],
        "decision": "revise",
        "automatic_semantic_acceptance_enabled": False,
        "primary_non_exact": report["methods"]["hybrid_rerank"]["non_exact"],
        "calibration_point_targets_met": policy["calibration"]["point_targets_met"],
        "independent_qualification_passed": policy["qualification"]["passed"],
        "resource_checks": resource["checks"],
        "resource_targets_passed": resource["resource_targets_passed"],
        "remaining_work": [
            "Human review of labels and independently collected calibration/holdout inputs",
            "Reduce broad-parent and related-product ranking errors, including the hoodie example",
            "Repeat resource qualification under recorded memory conditions; investigate swap",
            "Meet accuracy/coverage and resource gates before the Sprint 3 API",
        ],
    }
    (directory / "proof_of_concept_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(
        json.dumps(
            {
                "remaining_original_label_failures": len(failures),
                "audit_proposal_non_exact": derived,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
