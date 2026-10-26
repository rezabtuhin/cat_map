"""Record reasoned label proposals separately; never manufacture human approval."""

import json
from collections import Counter
from pathlib import Path

from category_matcher.artifacts import digest
from category_matcher.taxonomy import read_google_taxonomy

PROPOSALS = {
    "draft-080": (
        "more_specific_supported_label",
        ["543543"],
        "matched",
        "Printed reading books explicitly supports Print Books, a child of Books.",
    ),
    "draft-081": (
        "more_specific_supported_label",
        ["543527"],
        "matched",
        "Movies on DVD supports Film & Television DVDs rather than all DVDs & Videos.",
    ),
    "draft-097": (
        "ambiguous_input",
        [],
        "needs_clarification",
        "Adult-oriented does not establish erotic versus other adult-use products.",
    ),
    "draft-104": (
        "ambiguous_typo",
        [],
        "needs_clarification",
        "Soas can be a deletion from sofas or soaps; the query has no product context.",
    ),
    "draft-109": (
        "ambiguous_typo",
        [],
        "needs_clarification",
        "Shes alone does not uniquely support shoes; sheds and other spellings compete.",
    ),
    "draft-113": (
        "ambiguous_typo",
        [],
        "needs_clarification",
        "Tos can be a deletion from toys or tops; choosing Toys requires extra context.",
    ),
}


def main():
    source = Path("data/google/evaluation/development.jsonl")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    baseline = json.loads(Path("reports/sprint_2/retrieval_comparison.json").read_text())
    result = {row["example_id"]: row for row in baseline["examples"]["hybrid"]}
    categories, _ = read_google_taxonomy(Path("google_category_list.txt"))
    by_id = {c.category_id: c for c in categories}
    audit_rows, amended = [], []
    for row in rows:
        copy = dict(row)
        item = result[row["example_id"]]
        failed = row["expected_status"] == "matched" and item["first_relevant_rank"] != 1
        if row["example_id"] in PROPOSALS:
            kind, proposed_ids, proposed_status, reason = PROPOSALS[row["example_id"]]
            if not all(cid in by_id for cid in proposed_ids):
                raise ValueError("Invalid proposed category ID")
            copy.update(
                {
                    "acceptable_category_ids": proposed_ids,
                    "expected_status": proposed_status,
                    "audit_reason": reason,
                    "audit_status": "ai_audited_pending_human_review",
                }
            )
        else:
            kind = "ranking_error_proposal" if failed else "unchanged_draft"
            reason = (
                "Draft target appears plausible from product meaning and taxonomy; improve ranking."
                if failed
                else "No label change proposed by this targeted failure audit."
            )
            proposed_ids, proposed_status = row["acceptable_category_ids"], row["expected_status"]
        if failed or row["example_id"] in PROPOSALS:
            audit_rows.append(
                {
                    "example_id": row["example_id"],
                    "query": row["query"],
                    "finding": kind,
                    "reason": reason,
                    "original_expected_ids": row["acceptable_category_ids"],
                    "proposed_expected_ids": proposed_ids,
                    "proposed_status": proposed_status,
                    "category_paths": {cid: by_id[cid].full_path for cid in proposed_ids},
                    "prior_first_ids": item["candidate_ids"][:3],
                    "human_reviewed": False,
                }
            )
        amended.append(copy)
    directory = Path("reports/sprint_2/improvement")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "label_audit.json").write_text(
        json.dumps(
            {
                "original_dataset_sha256": digest(source),
                "changed_label_proposals": len(PROPOSALS),
                "finding_counts": dict(Counter(row["finding"] for row in audit_rows)),
                "human_review_complete": False,
                "findings": audit_rows,
            },
            indent=2,
        )
        + "\n"
    )
    target = Path("data/google/evaluation/development.audited.draft.jsonl")
    if target.exists():
        raise SystemExit("Refusing to overwrite an existing audit proposal dataset")
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in amended))
    print(
        f"Audited {len(audit_rows)} failures; {len(PROPOSALS)} label proposals. Original unchanged."
    )


if __name__ == "__main__":
    main()
