"""Generate the Sprint 2 decision and label-review queue from recorded executions."""

import json
from pathlib import Path


def main():
    directory = Path("reports/sprint_2")
    comparison = json.loads((directory / "retrieval_comparison.json").read_text())
    prior = json.loads(Path("reports/sprint_1/retrieval_baseline.json").read_text())
    policy = json.loads((directory / "draft_policy.json").read_text())
    resources = json.loads((directory / "resource_report.json").read_text())
    users = json.loads((directory / "user_examples.json").read_text())
    queue = []
    for item in comparison["examples"]["hybrid"]:
        if item["expected_status"] == "matched" and item["first_relevant_rank"] != 1:
            queue.append(
                {
                    "example_id": item["example_id"],
                    "query": item["query"],
                    "draft_expected_ids": item["acceptable_category_ids"],
                    "first_candidates": item["candidate_ids"][:5],
                    "reason": "missing_from_shortlist"
                    if item["first_relevant_rank"] == 0
                    else "ranking_or_draft_label_needs_review",
                    "review_status": "pending_review",
                }
            )
    queue.sort(key=lambda row: (row["reason"] != "missing_from_shortlist", row["example_id"]))
    (directory / "label_review_queue.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in queue)
    )
    decision = {
        "decision": "revise",
        "artifact_version": comparison["artifact_version"],
        "quality_gate_passed": False,
        "resource_targets_passed": resources["resource_targets_passed"],
        "trusted_labels_available": False,
        "independent_test_evaluated": False,
        "draft_policy_id": policy["policy_id"],
        "semantic_auto_acceptance_enabled": False,
        "reasons": [
            "Development and calibration labels remain unreviewed AI proposals.",
            "No independent reviewed holdout has been evaluated.",
            "Draft fitted coverage is below the 80% target.",
            "Draft development shortlist recall remains below the 99.5% target.",
        ],
        "next_work": [
            "Adjudicate development labels and review the queued failures.",
            "Improve ranking using development data without inspecting test outcomes.",
            "Fit on reviewed calibration data, then qualify once on a fresh reviewed holdout.",
            "Keep API and deployment work deferred until the quality gate passes.",
        ],
    }
    (directory / "proof_of_concept_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    metrics = policy["calibration"]["metrics"]
    rows = [
        "# Sprint 2 findings",
        "",
        "**Decision: revise. The local runtime target passes; matching quality is not qualified.**",
        "",
        "All accuracy figures below use unreviewed AI-authored labels. Development "
        "figures informed implementation choices; calibration figures are fitted on the "
        "same draft data used to choose thresholds. Neither is independent accuracy evidence.",
        "",
        f"Active artifact: `{comparison['artifact_version']}`.",
        "",
        "## Retrieval changes and development comparison",
        "",
        "BM25 now indexes category labels with retrieval-only plural folding and content-word "
        "normalization. BGE uses plain queries. Fusion weights semantic ranks by five and "
        "lexical/fuzzy ranks by one, with rank constant ten. Candidate limits are 100/100/50, "
        "with up to 200 fused results plus five parents. These are development-selected settings.",
        "",
        "| Method | Sprint 1 top-1 | Sprint 2 top-1 | Sprint 2 top-3 | Sprint 2 recall at 20 | "
        "Sprint 2 final recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("fuzzy", "bm25", "semantic", "hybrid"):
        old, new = prior["methods"][name]["non_exact"], comparison["methods"][name]["non_exact"]
        rows.append(
            f"| {name} | {old['top1_accuracy']:.1%} | {new['top1_accuracy']:.1%} | "
            f"{new['top3_recall']:.1%} | {new['candidate_recall_at_20']:.1%} | "
            f"{new['final_shortlist_recall']:.1%} |"
        )
    reranked = comparison["methods"]["hybrid_rerank"]["non_exact"]
    rows += [
        "",
        "The same 120 resolvable non-exact development examples are compared. Wider "
        "shortlist limits contribute to final-recall gains; recall at 20 keeps a common budget.",
        "",
        f"Optional MiniLM reranking reaches {reranked['top1_accuracy']:.1%} top-1 "
        "on these examples and remains disabled by default.",
        "",
        "The two absent draft targets are `adult oriented products` → Mature and "
        "`Tos` → Toys. Both require label/ambiguity review; a three-letter input can be "
        "underspecified. No alias was added merely to force these benchmark cases to pass.",
        "",
        "## User inputs under the default matcher",
        "",
        "| Input | First category | Outcome |",
        "|---|---|---|",
    ]
    for item in users["examples"]:
        result = item["baseline"]
        first = result["candidates"][0]
        query = item["query"].replace("|", "\\|")
        rows.append(f"| {query} | {first['id']} — {first['label']} | {result['status']} |")
    rows += [
        "",
        "Only Clothing and Tops have user-confirmed positive labels. The remaining "
        "positive labels are proposals; the user explicitly rejected baby tops for hoodies. "
        "The hoodie result excludes unsupported baby clothing. Eco Friendly keeps the broad "
        "Clothing context. Model-based candidates remain provisional.",
        "",
        "## Policy fit and data provenance",
        "",
        "There are 200 development, 200 calibration proposals, and 200 test proposals. "
        "Cross-split IDs, query keys, declared families, and category targets do not overlap. "
        "All 600 labels still await human review. Test proposals have not been scored.",
        "",
        f"The fitted draft policy would accept {metrics['accepted']}/{metrics['resolvable']} "
        f"resolvable calibration cases ({metrics['coverage']:.1%} coverage, target 80%). "
        f"It has {metrics['errors']} fitted-label errors among those acceptances. This is "
        "a calibration fit on draft labels, not a precision claim. Its one-sided 95% lower "
        f"precision bound is {metrics['precision_lower_95_one_sided']:.1%}, before considering "
        "the label-quality and data-reuse limitations.",
        "",
        "The implemented policy can preview matched, clarification, and no-match outcomes. "
        "`--experimental-policy` preserves provisional status and puts its proposed decision "
        "in `decision_preview`. `--policy` rejects an unqualified policy. Qualification also "
        "checks review status, independent holdout provenance, split isolation, minimum data "
        "sizes, recall, coverage, precision, and non-exact evidence. It records holdout access "
        "to prevent evaluating changed policies on previously inspected inputs.",
        "",
        "## Local performance",
        "",
        f"- Serial timed requests: {resources['requests']} over "
        f"{resources['unique_inputs_available']} distinct available non-exact development inputs.",
        f"- Warm end-to-end p95: {resources['end_to_end_warm_latency_ms']['p95']:.1f} ms.",
        "- Process peak resident memory: "
        f"{resources['process_peak_resident_bytes'] / 1024**2:.1f} MiB.",
        f"- Encoder load/warmup: {resources['query_model_load_and_warmup_seconds']:.2f} seconds.",
        f"- Observed system swapout page change: {resources['system_swapout_pages_delta']}.",
        "- This run uses default retrieval without reranking or a decision policy; "
        "it is not a production concurrency test.",
        "",
        "## Acceptance and next work",
        "",
        "Sprint 2's implementation and diagnostic runs are delivered. Full acceptance is "
        "pending trusted labels and a passing independent quality gate. The required next "
        "work is label adjudication and further ranking improvement. API integration "
        "remains deferred. See `label_review_queue.jsonl` and "
        "`proof_of_concept_decision.json` for the actionable records.",
    ]
    (directory / "FINDINGS.md").write_text("\n".join(rows) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
