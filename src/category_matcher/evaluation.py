"""Retrieval diagnostics with explicit label provenance and equal-budget metrics."""

import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from category_matcher.artifacts import digest
from category_matcher.matcher import Matcher
from category_matcher.query import source_ancestry, source_ancestry_options
from category_matcher.retrieval import METHODS


def read_examples(path: Path, matcher: Matcher, allow_draft: bool) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seen = set()
    for row in records:
        source_ancestry(row.get("source_ancestors"))
        if row["example_id"] in seen:
            raise ValueError("Duplicate evaluation example ID")
        seen.add(row["example_id"])
        if (row["taxonomy_key"], row["taxonomy_version"]) != ("google", matcher.version):
            raise ValueError("Dataset belongs to another taxonomy/version")
        ids = row["acceptable_category_ids"]
        if row["expected_status"] not in {
            "matched",
            "needs_clarification",
            "no_match",
            "unsupported_language",
        }:
            raise ValueError("Invalid expected semantic status")
        if bool(ids) != (row["expected_status"] == "matched"):
            raise ValueError("Only resolvable examples must have acceptable category IDs")
        if not all(isinstance(cid, str) and cid in matcher.store.categories for cid in ids):
            raise ValueError("Dataset contains an invalid expected category ID")
        approved = (
            row.get("review_status") == "approved"
            and row.get("reviewer")
            and row.get("reviewed_at")
        )
        if not approved and not allow_draft:
            raise ValueError(
                "Unreviewed labels: obtain human review or use --allow-draft diagnostics"
            )
    if not records:
        raise ValueError("Empty evaluation dataset")
    return records


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(rows: list[dict]) -> dict:
    resolvable = [row for row in rows if row["expected_status"] == "matched"]
    accepted = [row for row in rows if row["actual_status"] == "matched"]
    correct_accepted = sum(row["accepted_id"] in row["acceptable_category_ids"] for row in accepted)
    summary = {
        "examples": len(rows),
        "resolvable_examples": len(resolvable),
        "top1_correct": sum(row["first_relevant_rank"] == 1 for row in resolvable),
        "top3_correct": sum(0 < row["first_relevant_rank"] <= 3 for row in resolvable),
        "recall20_correct": sum(0 < row["first_relevant_rank"] <= 20 for row in resolvable),
        "shortlist_correct": sum(row["first_relevant_rank"] > 0 for row in resolvable),
        "automatic_matches": len(accepted),
        "correct_automatic_matches": correct_accepted,
        "incorrect_automatic_matches": len(accepted) - correct_accepted,
        "resolvable_automatic_matches": sum(
            row["expected_status"] == "matched" for row in accepted
        ),
    }
    for name, count in [
        ("top1_accuracy", "top1_correct"),
        ("top3_recall", "top3_correct"),
        ("candidate_recall_at_20", "recall20_correct"),
        ("final_shortlist_recall", "shortlist_correct"),
    ]:
        summary[name] = ratio(summary[count], len(resolvable))
    summary["automatic_match_precision"] = ratio(correct_accepted, len(accepted))
    summary["resolvable_automatic_coverage"] = ratio(
        summary["resolvable_automatic_matches"],
        len(resolvable),
    )
    uncertain = [row for row in rows if row["expected_status"] != "matched"]
    summary["unresolvable_examples"] = len(uncertain)
    summary["unresolvable_false_acceptances"] = sum(
        row["actual_status"] == "matched" for row in uncertain
    )
    return summary


def evaluate(
    matcher: Matcher,
    dataset: Path,
    *,
    allow_draft: bool = False,
    include_reranking: bool | None = None,
) -> dict:
    records = read_examples(dataset, matcher, allow_draft)
    queries = {}
    exact_results = {}
    for row in records:
        exact = matcher.match(
            row["query"],
            method="exact",
            category_id=row.get("category_id"),
            parent_id=row.get("parent_category_id"),
            language=row.get("language", "en"),
            context=row.get("context", ""),
            **source_ancestry_options(row),
        )
        exact_results[row["example_id"]] = exact
        if exact.status == "provisional":
            queries[row["example_id"]] = matcher.retrieval_query(
                row["query"], row.get("context", ""), **source_ancestry_options(row)
            )
    started = time.perf_counter()
    vectors = matcher.encode_queries(list(queries.values())) if queries else []
    encoding_seconds = time.perf_counter() - started
    by_example_vector = dict(zip(queries, vectors, strict=True))
    method_reports = {}
    details = {}
    if include_reranking is None:
        include_reranking = matcher.default_rerank or matcher.reranker is not None
    methods = list(METHODS) + (["hybrid_rerank"] if include_reranking else [])
    for method in methods:
        method_rows = []
        durations = []
        for row in records:
            started = time.perf_counter()
            result = matcher.match(
                row["query"],
                method="hybrid" if method == "hybrid_rerank" else method,
                category_id=row.get("category_id"),
                parent_id=row.get("parent_category_id"),
                language=row.get("language", "en"),
                context=row.get("context", ""),
                query_vector=by_example_vector.get(row["example_id"]),
                rerank=method == "hybrid_rerank",
                **source_ancestry_options(row),
            )
            durations.append((time.perf_counter() - started) * 1000)
            ids = [candidate["id"] for candidate in result.candidates]
            rank = next(
                (i for i, cid in enumerate(ids, 1) if cid in row["acceptable_category_ids"]), 0
            )
            method_rows.append(
                {
                    "example_id": row["example_id"],
                    "query": row["query"],
                    "slice": row["slice"],
                    "review_status": row["review_status"],
                    "expected_status": row["expected_status"],
                    "acceptable_category_ids": row["acceptable_category_ids"],
                    "actual_status": result.status,
                    "actual_method": result.method,
                    "reason_code": result.reason_code,
                    "query_analysis": result.query_analysis,
                    "accepted_id": result.category["id"] if result.category else None,
                    "first_relevant_rank": rank,
                    "candidate_ids": ids,
                    "exact_fast_path": exact_results[row["example_id"]].status == "matched",
                }
            )
        method_reports[method] = {
            "all": summarize(method_rows),
            "non_exact": summarize([row for row in method_rows if not row["exact_fast_path"]]),
            "slices": {
                name: summarize([row for row in method_rows if row["slice"] == name])
                for name in sorted({row["slice"] for row in records})
            },
            "retrieval_only_latency_ms": {
                "p50": float(np.percentile(durations, 50)),
                "p95": float(np.percentile(durations, 95)),
            },
        }
        details[method] = method_rows
    pending = sum(
        not (
            row.get("review_status") == "approved"
            and row.get("reviewer")
            and row.get("reviewed_at")
        )
        for row in records
    )
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": str(dataset),
        "dataset_sha256": digest(dataset),
        "dataset_status": "UNREVIEWED_DRAFT_DIAGNOSTIC_ONLY" if pending else "human_reviewed",
        "unreviewed_examples": pending,
        "example_count": len(records),
        "slice_counts": dict(Counter(row["slice"] for row in records)),
        "source_sha256": matcher.manifest["source_sha256"],
        "artifact_version": matcher.manifest["artifact_version"],
        "model": matcher.manifest["model"],
        "ranking_model": matcher.reranker_metadata,
        "retrieval_config": matcher.config,
        "packages": matcher.manifest["packages"],
        "batch_query_encoding_seconds_including_first_model_load": encoding_seconds,
        "notes": [
            "Draft labels are not ground truth or evidence of real-world accuracy.",
            "Model-based candidates remain provisional; no decision policy is applied in this run.",
            "All methods share canonical/approved-alias shortcuts; non-exact results are separate.",
            "Recall at 20 uses the same result budget; final shortlist sizes vary by branch.",
            "Retrieval latency excludes batch query encoding; it is not end-to-end latency.",
            "Methods share canonical, broad-label, and spelling rules: a pipeline comparison.",
            "Automatic precision/coverage here describe deterministic shortcuts only.",
            "Decision-policy outcomes are measured separately during calibration/qualification.",
        ],
        "methods": method_reports,
        "examples": details,
    }
