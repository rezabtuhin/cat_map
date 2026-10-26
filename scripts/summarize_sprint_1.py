"""Build readable Sprint 1 findings from the saved real execution reports."""

import json
from pathlib import Path


def first_result(row, name):
    candidates = row[name]["candidates"]
    return f"{candidates[0]['id']} — {candidates[0]['label']}" if candidates else "none"


def main():
    directory = Path("reports/sprint_1")
    experiment = json.loads((directory / "reranking_experiment.json").read_text())
    verification = json.loads((directory / "verification.json").read_text())
    users = json.loads((directory / "user_examples.json").read_text())
    baseline = {
        **experiment,
        "ranking_model": None,
        "methods": {
            key: value for key, value in experiment["methods"].items() if key != "hybrid_rerank"
        },
        "examples": {
            key: value for key, value in experiment["examples"].items() if key != "hybrid_rerank"
        },
    }
    baseline["notes"] = baseline["notes"] + [
        "Baseline methods extracted from the comparison run; reranking was not applied."
    ]
    (directory / "retrieval_baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")

    lines = [
        "# Sprint 1 findings",
        "",
        "**Outcome: executable prototype; quality not qualified.**",
        "",
        "The evaluation labels are AI-authored development drafts awaiting human review. "
        "These numbers diagnose retrieval behavior; they do not establish real-world accuracy.",
        "",
        f"Artifact: `{experiment['artifact_version']}`.",
        f"Dataset SHA-256: `{experiment['dataset_sha256']}`.",
        "",
        "## Integrity verification",
        "",
        f"- {verification['category_count']:,} canonical categories.",
        f"- {verification['canonical_roundtrip_checks']:,} successful ID/label/path checks.",
        f"- {verification['embedding_count']:,} normalized embeddings in SQLite.",
        f"- {verification['sqlite_vs_numpy_scoped_queries']} SQLite/NumPy numerical comparisons.",
        "",
        "## Non-exact retrieval diagnostics",
        "",
        "Exact shortcuts are excluded from this table. There are "
        f"{experiment['methods']['hybrid']['non_exact']['resolvable_examples']} draft resolvable "
        "non-exact examples. Recall at 20 uses the same candidate budget; "
        "final shortlist sizes can vary by method and parent expansion.",
        "",
        "| Method | Top-1 correct | Top-3 recall | Recall at 20 | Final shortlist recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, result in experiment["methods"].items():
        row = result["non_exact"]
        lines.append(
            f"| {method} | {row['top1_correct']}/{row['resolvable_examples']} "
            f"({row['top1_accuracy']:.1%}) | {row['top3_recall']:.1%} | "
            f"{row['candidate_recall_at_20']:.1%} | {row['final_shortlist_recall']:.1%} |"
        )
    lines += [
        "",
        "Hybrid retrieval improves candidate coverage over the simpler branches, but its "
        "first result is not reliably correct. The selected MiniLM passage reranker reduces "
        "top-1 accuracy on this draft set and remains disabled by default. Neither pipeline "
        "meets the planned production target.",
        "",
        "## User-provided inputs",
        "",
        "| Input | Default retrieval first result | Experimental reranked first result | Status |",
        "|---|---|---|---|",
    ]
    for row in users["examples"]:
        query = row["query"].replace("|", "\\|")
        lines.append(
            f"| {query} | {first_result(row, 'baseline')} | "
            f"{first_result(row, 'experimental_reranked')} | "
            f"{row['baseline']['status']} |"
        )
    lines += [
        "",
        "Clothing and Tops were confirmed by the user. Tops now has an approved alias; "
        "it is a training/regression example, not a held-out success. Positive labels for the "
        "other three inputs are drafts. The user explicitly rejected baby/toddler tops for "
        "the hoodie query.",
        "",
        "The default pipeline fixes that reported demographic error through pipe-context "
        "scoping and an unsupported-infant-apparel filter. Resolved parent words are not "
        "repeated as retrieval query terms, which previously overwhelmed the leaf concept. "
        "There is no hoodie/sweatshirt-specific entry in the supplied taxonomy.",
        "",
        "The phone example remains an important failure: default hybrid retrieval ranks "
        "accessories above phones, while the optional reranker corrects this example but "
        "hurts the broader draft benchmark. Do not select a different method per example "
        "and claim that cherry-picked combination as one evaluated pipeline.",
        "",
        "## Runtime observations",
        "",
        f"- Process peak resident memory: {users['process_peak_resident_mib']:.1f} MiB.",
        f"- Model loading and warmup: {users['load_and_warmup_seconds']:.2f} seconds.",
        "- Five inputs do not establish p95 latency, capacity, or the Sprint 2 resource gate.",
        "- The local reranker initially crashed in CPU linear inference. Its checkpoint "
        "parameters included addresses offset by two bytes. Copying misaligned parameters "
        "into aligned PyTorch storage resolved the observed failure; value preservation "
        "has a regression test.",
        "",
        "## Next work",
        "",
        "1. Review/correct development labels, preserving real user examples separately.",
        "2. Investigate categories missing from the shortlist before tuning acceptance.",
        "3. Improve ranking on phone-versus-accessory and unsupported-specialization errors; "
        "reconsider the generic MiniLM reranker using development comparisons.",
        "4. Collect distinct calibration and untouched-test inputs, calibrate abstention, "
        "and measure serial end-to-end inference before the Sprint 2 gate.",
        "5. Keep API and larger architecture expansion deferred until the quality gate passes.",
    ]
    (directory / "FINDINGS.md").write_text("\n".join(lines) + "\n")
    print(directory / "FINDINGS.md")


if __name__ == "__main__":
    main()
