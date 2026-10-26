"""Compare saved profile evidence and preserve the explicit keep/reject decision."""

import json
from pathlib import Path

from category_matcher.artifacts import digest, load_manifest, resolve_bundle


def first_id(result):
    return result["candidates"][0]["id"] if result["candidates"] else None


def main():
    folder = Path("reports/sprint_2/quantization")
    baseline_file = Path("reports/sprint_2/revision_3/retrieval_comparison.json")
    baseline = json.loads(baseline_file.read_text())
    old = {r["example_id"]: r for r in baseline["examples"]["hybrid_rerank"]}
    baseline_users = {}
    for filename, key in (
        ("user_examples", "experimental_reranked"),
        ("user_batch_2", "default_ranking"),
    ):
        for row in json.loads(Path("reports/sprint_2/revision_3", filename + ".json").read_text())[
            "examples"
        ]:
            baseline_users[row["query"]] = row[key]
    profiles = []
    for name in ("dynamic", "embeddings"):
        report_file = folder / name / "retrieval_comparison.json"
        report = json.loads(report_file.read_text())
        if report["dataset_sha256"] != baseline["dataset_sha256"]:
            raise ValueError("Development cohort changed")
        regressions, gains, changes = [], [], []
        for row in report["examples"]["hybrid_rerank"]:
            before = old[row["example_id"]]
            if row["expected_status"] != "matched":
                continue
            detail = {
                "example_id": row["example_id"],
                "query": row["query"],
                "before_rank": before["first_relevant_rank"],
                "after_rank": row["first_relevant_rank"],
                "before_top1": before["candidate_ids"][:1],
                "after_top1": row["candidate_ids"][:1],
            }
            if detail["before_top1"] != detail["after_top1"]:
                changes.append(detail)
            if detail["before_rank"] == 1 and detail["after_rank"] != 1:
                regressions.append(detail)
            if detail["before_rank"] != 1 and detail["after_rank"] == 1:
                gains.append(detail)
        user_report = json.loads((folder / name / "user_examples.json").read_text())
        user_checks = []
        for row in user_report["examples"]:
            current = row["result"]
            before = baseline_users[row["query"]]
            user_checks.append(
                {
                    "query": row["query"],
                    "same_status": current["status"] == before["status"],
                    "same_assigned_category": current["category"] == before["category"],
                    "same_first_candidate": first_id(current) == first_id(before),
                    "uncertain_stays_unmapped": current["status"] == "matched"
                    or current["category"] is None,
                }
            )
        resource = json.loads((folder / name / "resource_report.json").read_text())
        summary = json.loads((folder / name / "evaluate_summary.json").read_text())
        profiles.append(
            {
                "profile": summary["runtime_override"]["inference_profile"],
                "quality_report_sha256": digest(report_file),
                "metrics": report["methods"]["hybrid_rerank"]["non_exact"],
                "top1_regressions": regressions,
                "top1_gains": gains,
                "changed_resolvable_predictions": changes,
                "user_checks": user_checks,
                "parameter_tensors": summary["reranker_tensors"],
                "resource_requests": resource["requests"],
                "warm_latency_ms": resource["end_to_end_warm_latency_ms"],
                "process_peak_resident_bytes": resource["process_peak_resident_bytes"],
                "process_after_inference": resource["resource_diagnostics"][
                    "process_after_inference"
                ],
                "resource_checks": resource["checks"],
            }
        )
    manifest = load_manifest(resolve_bundle(Path("artifacts/google")))
    if manifest["artifact_version"] != baseline["artifact_version"]:
        raise ValueError("Active baseline changed during the experiments")
    evidence = {
        "baseline_artifact_version": baseline["artifact_version"],
        "baseline_quality_report_sha256": digest(baseline_file),
        "decision": "keep_float32_default",
        "automatic_semantic_acceptance_enabled": False,
        "default_configuration_changed": False,
        "baseline_metrics": baseline["methods"]["hybrid_rerank"]["non_exact"],
        "profiles": profiles,
        "reasons": [
            "Full int8 has development ranking regressions.",
            "Embedding-only preserves top-1 outcomes but raises memory peak and footprint.",
            "No quantized profile passes the recorded conservative resource check.",
        ],
        "limitations": [
            "All 200 labels remain unreviewed; this is development model selection.",
            "Float32 and embedding-only resource runs use 200 identical serial inputs.",
            "Full-int8 timing is a 30-request diagnostic after quality rejection.",
            "Sequential runs do not control other applications or establish swap attribution.",
            "Test inputs and calibration fitting were not used for this experiment.",
        ],
    }
    (folder / "decision.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        json.dumps(
            {
                "decision": evidence["decision"],
                "active_artifact": manifest["artifact_version"],
                "all_user_contract_checks_passed": all(
                    all(
                        r[key]
                        for key in (
                            "same_status",
                            "same_assigned_category",
                            "same_first_candidate",
                            "uncertain_stays_unmapped",
                        )
                    )
                    for profile in profiles
                    for r in profile["user_checks"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
