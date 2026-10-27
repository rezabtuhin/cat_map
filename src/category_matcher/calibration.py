"""Calibrate thresholds without using test labels; draft runs generate previews only."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.stats import beta

from category_matcher.artifacts import digest
from category_matcher.datasets import check_splits, is_reviewed, read_jsonl
from category_matcher.decision import DecisionPolicy, evidence
from category_matcher.evaluation import read_examples
from category_matcher.matcher import Matcher
from category_matcher.query import source_ancestry_options


def observations(matcher: Matcher, dataset: Path, *, allow_draft: bool, rerank: bool) -> list[dict]:
    if matcher.policy is not None:
        raise ValueError(
            "Calibration/qualification requires an unmodified matcher without a policy"
        )
    rows = read_examples(dataset, matcher, allow_draft)
    output = []
    for row in rows:
        result = matcher.match(
            row["query"],
            category_id=row.get("category_id"),
            context=row.get("context", ""),
            parent_id=row.get("parent_category_id"),
            language=row.get("language", "en"),
            rerank=rerank,
            **source_ancestry_options(row),
        )
        output.append({"row": row, "result": result, "features": evidence(result)})
    return output


def score_policy(items: list[dict], policy: DecisionPolicy) -> dict:
    accepted = correct = coverage = ranked = model_accepted = model_correct = 0
    resolvable = 0
    confusion = {}
    for item in items:
        row, result = item["row"], item["result"]
        gold = set(row["acceptable_category_ids"])
        resolvable += bool(gold)
        ranked += bool(gold & {c["id"] for c in result.candidates})
        if result.status == "provisional":
            predicted = policy.preview(result)
            status, category = predicted["status"], predicted["category"]
            model = True
        else:
            status, category, model = result.status, result.category, False
        key = row["expected_status"] + " -> " + status
        confusion[key] = confusion.get(key, 0) + 1
        if status == "matched":
            accepted += 1
            good = category is not None and category["id"] in gold
            correct += good
            coverage += bool(gold)
            model_accepted += model
            model_correct += model and good
    errors = accepted - correct
    return {
        "examples": len(items),
        "resolvable": resolvable,
        "accepted": accepted,
        "correct": correct,
        "errors": errors,
        "precision": correct / accepted if accepted else None,
        "precision_lower_95_one_sided": float(beta.ppf(0.05, correct, errors + 1))
        if correct
        else 0.0,
        "coverage": coverage / resolvable if resolvable else None,
        "candidate_recall": ranked / resolvable if resolvable else None,
        "model_accepted": model_accepted,
        "model_precision": model_correct / model_accepted if model_accepted else None,
        "confusion": confusion,
    }


def calibrate(
    matcher: Matcher,
    dataset: Path,
    *,
    allow_draft: bool,
    development: Path,
    rerank: bool | None = None,
) -> dict:
    if rerank is None:
        rerank = getattr(matcher, "default_rerank", False)
    splits = check_splits({"development": development, "calibration": dataset})
    if not splits["passed"]:
        raise ValueError("Development/calibration leakage: " + "; ".join(splits["conflicts"][:3]))
    items = observations(matcher, dataset, allow_draft=allow_draft, rerank=rerank)
    samples = [item["features"] for item in items if item["result"].status == "provisional"]
    if not samples:
        raise ValueError("Calibration needs non-exact input examples")
    semantic_values = np.array([item["semantic"] for item in samples])
    margins = np.array([item["margin"] for item in samples])
    # Quantiles are derived exclusively from calibration evidence. A deliberately
    # impossible cutoff remains available when no safe acceptance region exists.
    semantic_grid = sorted(
        set(np.quantile(semantic_values, np.linspace(0, 1, 16)).tolist() + [1.0])
    )
    margin_grid = sorted(set(np.quantile(margins, np.linspace(0, 1, 12)).tolist() + [0.0]))
    document = {
        "format_version": 1,
        "taxonomy_key": "google",
        "taxonomy_version": matcher.version,
        "artifact_version": matcher.manifest["artifact_version"],
        "created_at": datetime.now(UTC).isoformat(),
        "ranking_method": "hybrid+rerank" if rerank else "hybrid",
        "ranking_model": matcher.reranker_metadata if rerank else None,
        "calibration": {
            "dataset": str(dataset),
            "dataset_sha256": digest(dataset),
            "example_count": len(items),
            "unreviewed_examples": sum(not is_reviewed(i["row"]) for i in items),
            "development_sha256": digest(development),
        },
        "qualification": {"passed": False, "reason": "Independent reviewed holdout required"},
    }
    reject = float(np.quantile(semantic_values, 0.10))
    best = None
    for semantic in semantic_grid:
        for margin in margin_grid:
            thresholds = {
                "min_semantic": semantic,
                "min_margin": margin,
                "no_match_semantic": min(reject, semantic),
                "fuzzy_support": 75.0,
            }
            proposal = {**document, "thresholds": thresholds, "policy_id": "grid-search"}
            metrics = score_policy(items, DecisionPolicy(proposal, experimental=True))
            feasible = metrics["precision"] is not None and metrics["precision"] >= 0.99
            key = (
                feasible,
                metrics["coverage"] or 0 if feasible else -(metrics["errors"]),
                metrics["model_accepted"] if feasible else metrics["correct"],
            )
            if best is None or key > best[0]:
                best = key, thresholds, metrics
    document["thresholds"] = best[1]
    # Separately fit rejection behavior; acceptance settings remain fixed.
    best_rejection = None
    for cutoff in np.quantile(semantic_values, np.linspace(0, 1, 12)):
        for fuzzy_support in (60.0, 75.0, 90.0):
            thresholds = {
                **best[1],
                "no_match_semantic": min(float(cutoff), best[1]["min_semantic"]),
                "fuzzy_support": fuzzy_support,
            }
            proposal = {**document, "thresholds": thresholds, "policy_id": "rejection-search"}
            policy = DecisionPolicy(proposal, experimental=True)
            correct_rejections = wrong_rejections = 0
            for item in items:
                if item["result"].status != "provisional":
                    continue
                predicted = policy.preview(item["result"])["status"]
                if predicted == "no_match":
                    if item["row"]["expected_status"] == "no_match":
                        correct_rejections += 1
                    else:
                        wrong_rejections += 1
            key = (correct_rejections - 5 * wrong_rejections, -wrong_rejections)
            if best_rejection is None or key > best_rejection[0]:
                best_rejection = key, thresholds
    document["thresholds"] = best_rejection[1]
    metrics = score_policy(items, DecisionPolicy({**document, "policy_id": "fitted"}, True))
    document["calibration"]["metrics"] = metrics
    document["calibration"]["point_targets_met"] = (metrics["precision"] or 0) >= 0.99 and (
        metrics["coverage"] or 0
    ) >= 0.8
    document["notes"] = [
        "No test labels used to choose thresholds.",
        "Draft labels permit diagnostic previews only.",
        "Calibrated scores are not correctness probabilities.",
    ]
    return document


def qualify(
    matcher: Matcher,
    policy_path: Path,
    dataset: Path,
    development: Path,
    calibration_path: Path,
    *,
    allow_draft: bool = False,
    ledger: Path = Path("reports/sprint_2/holdout_access.jsonl"),
) -> dict:
    policy = DecisionPolicy.load(
        policy_path, artifact_version=matcher.manifest["artifact_version"], experimental=True
    )
    document = policy.document
    if digest(calibration_path) != document["calibration"]["dataset_sha256"]:
        raise ValueError("Calibration data changed after fitting this policy")
    if digest(development) != document["calibration"]["development_sha256"]:
        raise ValueError("Development data changed after fitting this policy")
    splits = check_splits(
        {"development": development, "calibration": calibration_path, "test": dataset}
    )
    if not splits["passed"]:
        raise ValueError("Evaluation split leakage: " + "; ".join(splits["conflicts"][:3]))
    rows = read_jsonl(dataset)
    # Hash inputs without review/role metadata: marking previously inspected
    # proposals reviewed does not turn them into a fresh independent holdout.
    import hashlib

    inputs = [
        {
            key: row.get(key)
            for key in ("query", "category_id", "context", "parent_category_id", "language")
        }
        for row in rows
    ]
    inputs.sort(key=lambda item: json.dumps(item, sort_keys=True))
    input_hash = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    independent = all(row.get("dataset_role") == "independent_holdout" for row in rows)
    history = read_jsonl(ledger) if ledger.exists() else []
    for access in history:
        if (
            independent
            and access["input_hash"] == input_hash
            and (access["policy_id"] != document["policy_id"] or not access["independent"])
        ):
            raise ValueError("These holdout inputs were already inspected; collect a fresh holdout")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "input_hash": input_hash,
                    "policy_id": document["policy_id"],
                    "independent": independent,
                    "accessed_at": datetime.now(UTC).isoformat(),
                }
            )
            + "\n"
        )
    rerank = document["ranking_method"] == "hybrid+rerank"
    if rerank and matcher.reranker_metadata != document["ranking_model"]:
        raise ValueError("Reranker configuration differs from calibration")
    items = observations(matcher, dataset, allow_draft=allow_draft, rerank=rerank)
    metrics = score_policy(items, policy)
    checks = {
        "reviewed_development": splits["counts"]["development"]["unreviewed"] == 0,
        "reviewed_calibration": document["calibration"]["unreviewed_examples"] == 0,
        "reviewed_test": all(is_reviewed(row) for row in rows),
        "independent_holdout": independent,
        "calibration_size": document["calibration"]["example_count"] >= 200,
        "test_size": len(rows) >= 200,
        "precision": (metrics["precision"] or 0) >= 0.99,
        "coverage": (metrics["coverage"] or 0) >= 0.8,
        "candidate_recall": (metrics["candidate_recall"] or 0) >= 0.995,
        "model_evidence": metrics["model_accepted"] >= 30
        and (metrics["model_precision"] or 0) >= 0.99,
    }
    document["qualification"] = {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "dataset": str(dataset),
        "dataset_sha256": digest(dataset),
        "example_count": len(rows),
        "unreviewed_examples": sum(not is_reviewed(row) for row in rows),
        "independent_holdout": independent,
        "notes": "Pilot quality gate only; production requires the larger Sprint 5 qualification.",
    }
    return document
