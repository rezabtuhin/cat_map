"""Versioned, data-calibrated abstention rules with explicit draft-mode isolation."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from category_matcher.domain import MatchResult

QUALIFICATION_CHECKS = {
    "reviewed_development",
    "reviewed_calibration",
    "reviewed_test",
    "independent_holdout",
    "calibration_size",
    "test_size",
    "precision",
    "coverage",
    "candidate_recall",
    "model_evidence",
}


def evidence(result: MatchResult) -> dict[str, float]:
    if not result.candidates:
        return {"semantic": -1.0, "margin": 0.0, "fuzzy": 0.0, "bm25": 0.0}
    top = result.candidates[0]
    branches = top.get("evidence", {})
    score_key = "reranker_score" if "reranker_score" in top else "score"
    second = result.candidates[1].get(score_key, 0) if len(result.candidates) > 1 else 0
    return {
        "semantic": branches.get("semantic", {}).get("raw_score", -1.0),
        "margin": max(0.0, top.get(score_key, 0) - second),
        "fuzzy": branches.get("fuzzy", {}).get("raw_score", 0.0),
        "bm25": branches.get("bm25", {}).get("raw_score", 0.0),
    }


def policy_id(document: dict) -> str:
    content = {key: value for key, value in document.items() if key != "policy_id"}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:20]


def write_policy(path: Path, document: dict) -> dict:
    document = {**document, "policy_id": policy_id(document)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n")
    return document


@dataclass(frozen=True)
class DecisionPolicy:
    document: dict
    experimental: bool

    @classmethod
    def load(cls, path: Path, *, artifact_version: str, experimental: bool = False):
        document = json.loads(path.read_text())
        if document.get("policy_id") != policy_id(document):
            raise ValueError("Policy checksum mismatch")
        if document.get("format_version") != 1 or document.get("taxonomy_key") != "google":
            raise ValueError("Unsupported decision policy")
        if document.get("artifact_version") != artifact_version:
            raise ValueError("Decision policy belongs to a different artifact version")
        thresholds = document["thresholds"]
        for name in ("min_semantic", "min_margin", "no_match_semantic", "fuzzy_support"):
            value = thresholds[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Invalid policy threshold: {name}")
        if not (
            -1 <= thresholds["min_semantic"] <= 1
            and -1 <= thresholds["no_match_semantic"] <= thresholds["min_semantic"]
            and thresholds["min_margin"] >= 0
            and 0 <= thresholds["fuzzy_support"] <= 100
        ):
            raise ValueError("Policy thresholds are out of range")
        qualification = document.get("qualification", {})
        calibration = document.get("calibration", {})
        validated = (
            qualification.get("passed") is True
            and all(
                qualification.get("checks", {}).get(name) is True for name in QUALIFICATION_CHECKS
            )
            and qualification.get("unreviewed_examples") == 0
            and calibration.get("unreviewed_examples") == 0
            and qualification.get("independent_holdout") is True
            and qualification.get("dataset_sha256") != calibration.get("dataset_sha256")
            and qualification.get("example_count", 0) >= 200
            and calibration.get("example_count", 0) >= 200
        )
        if not experimental and not validated:
            raise ValueError(
                "Policy is not independently validated; use --experimental-policy for previews"
            )
        return cls(document, experimental)

    def preview(self, result: MatchResult) -> dict:
        if result.method != self.document["ranking_method"]:
            raise ValueError("Decision policy was calibrated for a different ranking method")
        if (
            result.method == "hybrid+rerank"
            and result.ranking_model != self.document["ranking_model"]
        ):
            raise ValueError("Decision policy reranker revision/configuration mismatch")
        thresholds = self.document["thresholds"]
        values = evidence(result)
        if not result.candidates:
            status, reason = "no_match", "no_candidates"
        elif result.query_analysis.get("spelling", {}).get("ambiguous"):
            status, reason = "needs_clarification", "competing_spelling_interpretations"
        elif result.query_analysis.get("specificity_conflict"):
            status, reason = "needs_clarification", "scope_only_model_support"
        elif result.candidates[0].get("source") in {"context_only", "explicit_broad"}:
            status, reason = "needs_clarification", "attribute_only_broad_context"
        elif (
            values["semantic"] >= thresholds["min_semantic"]
            and values["margin"] >= thresholds["min_margin"]
        ):
            status, reason = "matched", "calibrated_score_and_margin"
        elif (
            values["semantic"] < thresholds["no_match_semantic"]
            and values["fuzzy"] < thresholds["fuzzy_support"]
            and values["bm25"] <= 0
        ):
            status, reason = "no_match", "insufficient_category_evidence"
        else:
            status, reason = "needs_clarification", "insufficient_separation_or_support"
        category = None
        if status == "matched":
            top = result.candidates[0]
            category = {key: top[key] for key in ("id", "label", "path")}
        return {
            "status": status,
            "category": category,
            "reason_code": reason,
            "features": values,
            "policy_id": self.document["policy_id"],
        }

    def apply(self, result: MatchResult) -> MatchResult:
        if result.status != "provisional":
            return result
        decision = self.preview(result)
        result.policy_id = self.document["policy_id"]
        if self.experimental:
            result.decision_preview = decision
            result.reason_code = "experimental_policy_preview_only"
        else:
            result.status = decision["status"]
            result.category = decision["category"]
            result.reason_code = decision["reason_code"]
        if decision["status"] == "needs_clarification":
            interpretations = result.query_analysis.get("spelling", {}).get("interpretations")
            names = [candidate["label"] for candidate in (interpretations or result.candidates[:2])]
            result.clarification = (
                "Please specify the product type. Possible categories: " + ", ".join(names) + "."
            )
        return result
