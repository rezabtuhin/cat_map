import json
from types import SimpleNamespace

import pytest

from category_matcher.artifacts import digest
from category_matcher.calibration import calibrate, qualify
from category_matcher.decision import write_policy
from category_matcher.domain import MatchResult


def save(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def row(identifier, target, query=None, role="calibration_proposal"):
    return {
        "example_id": identifier,
        "query": query or identifier,
        "taxonomy_key": "google",
        "taxonomy_version": "v1",
        "input_family": identifier,
        "expected_status": "matched",
        "acceptable_category_ids": [target],
        "review_status": "approved",
        "reviewer": "unit-test-fixture",
        "reviewed_at": "fixture",
        "dataset_role": role,
    }


def test_calibration_selects_abstention_instead_of_accepting_wrong_top_results(
    tmp_path, monkeypatch
):
    development, calibration = tmp_path / "development.jsonl", tmp_path / "calibration.jsonl"
    save(development, [row("dev", "1")])
    rows = [row("cal-right", "3"), row("cal-wrong", "4")]
    save(calibration, rows)
    samples = []
    for source, semantic in zip(rows, [0.95, 0.6], strict=True):
        result = MatchResult(
            "provisional",
            "google",
            "v1",
            "a",
            "hybrid",
            candidates=[
                {
                    "id": "3",
                    "label": "Phones",
                    "path": "Phones",
                    "score": 1,
                    "evidence": {"semantic": {"raw_score": semantic}},
                },
                {"id": "4", "label": "Cases", "path": "Cases", "score": 0.7},
            ],
        )
        samples.append(
            {"row": source, "result": result, "features": {"semantic": semantic, "margin": 0.3}}
        )
    monkeypatch.setattr("category_matcher.calibration.observations", lambda *a, **kw: samples)
    matcher = SimpleNamespace(
        version="v1", manifest={"artifact_version": "a"}, reranker_metadata=None
    )
    policy = calibrate(matcher, calibration, development=development, allow_draft=False)
    assert policy["calibration"]["metrics"]["precision"] == 1
    assert policy["calibration"]["metrics"]["coverage"] == 0.5
    assert policy["qualification"]["passed"] is False


def test_exact_only_success_cannot_qualify_semantic_policy_and_holdout_cannot_be_reused(
    tmp_path, monkeypatch
):
    dev, cal, test = [tmp_path / name for name in ("dev.jsonl", "cal.jsonl", "test.jsonl")]
    save(dev, [row("dev", "dev-target")])
    save(cal, [row(f"cal-{i}", f"cal-target-{i}") for i in range(200)])
    test_rows = [row(f"test-{i}", f"test-target-{i}") for i in range(200)]
    save(test, test_rows)
    policy = {
        "format_version": 1,
        "taxonomy_key": "google",
        "artifact_version": "a",
        "ranking_method": "hybrid",
        "ranking_model": None,
        "thresholds": {
            "min_semantic": 0.9,
            "min_margin": 0.1,
            "no_match_semantic": 0.5,
            "fuzzy_support": 75,
        },
        "calibration": {
            "dataset_sha256": digest(cal),
            "development_sha256": digest(dev),
            "example_count": 200,
            "unreviewed_examples": 0,
        },
        "qualification": {"passed": False},
    }
    policy_path = tmp_path / "policy.json"
    write_policy(policy_path, policy)
    items = [
        {
            "row": source,
            "result": MatchResult(
                "matched",
                "google",
                "v1",
                "a",
                "exact",
                category={"id": source["acceptable_category_ids"][0]},
                candidates=[{"id": source["acceptable_category_ids"][0]}],
            ),
        }
        for source in test_rows
    ]
    monkeypatch.setattr("category_matcher.calibration.observations", lambda *a, **kw: items)
    matcher = SimpleNamespace(manifest={"artifact_version": "a"}, reranker_metadata=None)
    ledger = tmp_path / "ledger.jsonl"
    output = qualify(matcher, policy_path, test, dev, cal, ledger=ledger)
    assert not output["qualification"]["passed"]
    assert not output["qualification"]["checks"]["model_evidence"]
    assert output["qualification"]["metrics"]["precision"] == 1
    for source in test_rows:
        source["dataset_role"] = "independent_holdout"
    test_rows.reverse()
    save(test, test_rows)
    with pytest.raises(ValueError, match="already inspected"):
        qualify(matcher, policy_path, test, dev, cal, ledger=ledger)
