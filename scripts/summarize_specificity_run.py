"""Publish reproducible revision evidence and a human-review queue, without scoring tests."""

import json
from pathlib import Path

from category_matcher.artifacts import digest
from category_matcher.evaluation import summarize
from category_matcher.taxonomy import read_google_taxonomy


def main():
    folder = Path("reports/sprint_2/specificity")
    current = json.loads((folder / "retrieval_comparison.json").read_text())
    previous = json.loads(
        Path("reports/sprint_2/improvement/retrieval_comparison.json").read_text()
    )
    if previous["dataset_sha256"] != current["dataset_sha256"]:
        raise ValueError("Development cohort changed between revisions")
    categories, _ = read_google_taxonomy(Path("google_category_list.txt"))
    paths = {c.category_id: c.full_path for c in categories}
    old = {r["example_id"]: r for r in previous["examples"]["hybrid_rerank"]}
    audit = json.loads(Path("reports/sprint_2/improvement/label_audit.json").read_text())
    if audit["original_dataset_sha256"] != current["dataset_sha256"]:
        raise ValueError("Audit proposals belong to a different development cohort")
    proposals = {
        r["example_id"]: r for r in audit["findings"] if r["finding"] != "ranking_error_proposal"
    }
    changes, failures, audit_rows = [], [], []
    review = [
        "# Development label review queue",
        "",
        "Every label below is an AI draft awaiting human review. The original development",
        "dataset remains unchanged. Rankings are diagnostic alternatives, not label approvals.",
        "This queue covers remaining failures and the earlier six audit proposals; it does",
        "not replace review of all development/calibration labels. Test inputs remain unscored.",
        "",
        "For each item: check the taxonomy, record acceptable IDs or an uncertainty status,",
        "and record a human reviewer and review date in the JSONL source. Keep ambiguity when",
        "the input lacks enough context. Do not create aliases from benchmark-only proposals.",
        "",
    ]
    for row in current["examples"]["hybrid_rerank"]:
        amended = dict(row)
        if proposal := proposals.get(row["example_id"]):
            amended["expected_status"] = proposal["proposed_status"]
            amended["acceptable_category_ids"] = proposal["proposed_expected_ids"]
            amended["first_relevant_rank"] = next(
                (
                    i
                    for i, cid in enumerate(row["candidate_ids"], 1)
                    if cid in amended["acceptable_category_ids"]
                ),
                0,
            )
        audit_rows.append(amended)
        earlier = old[row["example_id"]]
        changed = row["candidate_ids"][:1] != earlier["candidate_ids"][:1]
        if changed:
            changes.append(
                {
                    "example_id": row["example_id"],
                    "query": row["query"],
                    "previous_top1": earlier["candidate_ids"][:1],
                    "current_top1": row["candidate_ids"][:1],
                    "previous_rank": earlier["first_relevant_rank"],
                    "current_rank": row["first_relevant_rank"],
                }
            )
        failed = row["expected_status"] == "matched" and row["first_relevant_rank"] != 1
        if failed:
            failures.append(row)
        if not failed and row["example_id"] not in proposals:
            continue
        review.extend(
            [
                f"## {row['example_id']}: {row['query']}",
                "",
                f"Draft status: `{row['expected_status']}`. Human review: **pending**.",
                "",
            ]
        )
        for cid in row["acceptable_category_ids"]:
            review.append(f"- Draft target: `{cid}` — {paths[cid]}")
        for cid in row["candidate_ids"][:3]:
            review.append(f"- Ranked alternative: `{cid}` — {paths[cid]}")
        spelling = row.get("query_analysis", {}).get("spelling", {})
        for interpretation in spelling.get("interpretations", []):
            cid = interpretation["id"]
            review.append(f"- Spelling interpretation: `{cid}` — {paths[cid]}")
        if proposal := proposals.get(row["example_id"]):
            review.append(f"- Earlier unapproved audit proposal: {proposal['reason']}")
        review.extend(["", "Reviewer decision and reasoning: _pending_", ""])
    (folder / "LABEL_REVIEW.md").write_text("\n".join(review))
    (folder / "remaining_failures.json").write_text(
        json.dumps(
            {
                "artifact_version": current["artifact_version"],
                "label_status": "UNREVIEWED_DRAFT_DIAGNOSTIC_ONLY",
                "examples": failures,
            },
            indent=2,
        )
        + "\n"
    )
    (folder / "audit_rescore.json").write_text(
        json.dumps(
            {
                "artifact_version": current["artifact_version"],
                "label_status": "AI_AUDIT_PROPOSALS_NOT_HUMAN_REVIEWED",
                "audit_sha256": digest(Path("reports/sprint_2/improvement/label_audit.json")),
                "original_dataset_sha256": current["dataset_sha256"],
                "notes": [
                    "Same six earlier proposals, applied to saved predictions; no new inference.",
                    "Original-label metrics remain primary; this is a sensitivity diagnostic.",
                ],
                "non_exact": summarize([r for r in audit_rows if not r["exact_fast_path"]]),
            },
            indent=2,
        )
        + "\n"
    )
    policy = json.loads((folder / "draft_policy.json").read_text())
    resource = json.loads((folder / "resource_report.json").read_text())
    for report in (policy, resource):
        if report["artifact_version"] != current["artifact_version"]:
            raise ValueError("Evidence belongs to different artifact versions")
    summary = {
        "artifact_version": current["artifact_version"],
        "decision": "revise",
        "automatic_semantic_acceptance_enabled": False,
        "label_status": current["dataset_status"],
        "dataset_sha256": current["dataset_sha256"],
        "comparison_report_sha256": digest(folder / "retrieval_comparison.json"),
        "previous_non_exact": previous["methods"]["hybrid_rerank"]["non_exact"],
        "current_non_exact": current["methods"]["hybrid_rerank"]["non_exact"],
        "top1_changes": changes,
        "remaining_original_label_failures": len(failures),
        "calibration_point_targets_met": policy["calibration"]["point_targets_met"],
        "independent_qualification_passed": policy["qualification"]["passed"],
        "resource_checks": resource["checks"],
        "resource_targets_passed": resource["resource_targets_passed"],
        "test_predictions_run_by_this_workflow": False,
    }
    (folder / "proof_of_concept_decision.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "top1_changes"}, indent=2))


if __name__ == "__main__":
    main()
