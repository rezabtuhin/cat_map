import copy

import pytest

from category_matcher.decision import QUALIFICATION_CHECKS, DecisionPolicy, write_policy
from category_matcher.domain import MatchResult


def document():
    return {
        "format_version": 1,
        "taxonomy_key": "google",
        "artifact_version": "test-artifact",
        "ranking_method": "hybrid",
        "ranking_model": None,
        "thresholds": {
            "min_semantic": 0.8,
            "min_margin": 0.05,
            "no_match_semantic": 0.5,
            "fuzzy_support": 75,
        },
        "calibration": {
            "example_count": 200,
            "dataset_sha256": "calibration",
            "unreviewed_examples": 200,
        },
        "qualification": {"passed": False},
    }


def result(score=0.95, margin=0.2):
    return MatchResult(
        "provisional",
        "google",
        "test-version",
        "test-artifact",
        "hybrid",
        candidates=[
            {
                "id": "3",
                "label": "Mobile Phones",
                "path": "Electronics > Mobile Phones",
                "score": 1,
                "evidence": {"semantic": {"raw_score": score}},
            },
            {
                "id": "4",
                "label": "Phone Cases",
                "path": "Electronics > Cases",
                "score": 1 - margin,
                "evidence": {"semantic": {"raw_score": 0.6}},
            },
        ],
    )


def test_draft_cannot_enable_actual_automatic_acceptance(tmp_path):
    path = tmp_path / "policy.json"
    write_policy(path, document())
    with pytest.raises(ValueError, match="not independently validated"):
        DecisionPolicy.load(path, artifact_version="test-artifact")
    policy = DecisionPolicy.load(path, artifact_version="test-artifact", experimental=True)
    answer = policy.apply(result())
    assert answer.status == "provisional" and answer.category is None
    assert answer.decision_preview["status"] == "matched"
    assert answer.decision_preview["category"]["id"] == "3"


def test_validated_policy_can_accept_clarify_and_reject(tmp_path):
    doc = document()
    doc["calibration"]["unreviewed_examples"] = 0
    doc["qualification"] = {
        "passed": True,
        "checks": dict.fromkeys(QUALIFICATION_CHECKS, True),
        "example_count": 200,
        "unreviewed_examples": 0,
        "independent_holdout": True,
        "dataset_sha256": "holdout",
    }
    path = tmp_path / "policy.json"
    write_policy(path, doc)
    policy = DecisionPolicy.load(path, artifact_version="test-artifact")
    assert policy.apply(result()).status == "matched"
    uncertain = policy.apply(result(margin=0.01))
    assert uncertain.status == "needs_clarification" and uncertain.category is None
    assert "product type" in uncertain.clarification
    assert policy.apply(result(score=0.2)).status == "no_match"


def test_artifact_thresholds_and_model_compatibility(tmp_path):
    path = tmp_path / "policy.json"
    write_policy(path, document())
    with pytest.raises(ValueError, match="different artifact"):
        DecisionPolicy.load(path, artifact_version="wrong", experimental=True)
    doc = document()
    doc["thresholds"]["min_semantic"] = float("nan")
    write_policy(path, doc)
    with pytest.raises(ValueError, match="Invalid policy threshold"):
        DecisionPolicy.load(path, artifact_version="test-artifact", experimental=True)
    doc = document()
    doc["ranking_method"] = "hybrid+rerank"
    doc["ranking_model"] = {"revision": "first"}
    policy = DecisionPolicy(doc, experimental=True)
    answer = result()
    with pytest.raises(ValueError, match="ranking method"):
        policy.preview(answer)
    answer.method = "hybrid+rerank"
    answer.ranking_model = {"revision": "other"}
    with pytest.raises(ValueError, match="revision"):
        policy.preview(answer)


def test_partial_or_unreviewed_qualification_never_enables_policy(tmp_path):
    doc = document()
    doc["calibration"]["unreviewed_examples"] = 0
    doc["qualification"] = {
        "passed": True,
        "checks": {"precision": True},
        "example_count": 200,
        "unreviewed_examples": 0,
        "independent_holdout": True,
        "dataset_sha256": "other",
    }
    path = tmp_path / "policy.json"
    write_policy(path, copy.deepcopy(doc))
    with pytest.raises(ValueError, match="not independently"):
        DecisionPolicy.load(path, artifact_version="test-artifact")
