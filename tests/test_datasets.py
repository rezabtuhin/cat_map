import json
from pathlib import Path

from category_matcher.datasets import check_splits
from category_matcher.normalization import exact_key, retrieval_terms


def test_retrieval_normalization_never_changes_exact_keys():
    assert retrieval_terms("Phones and accessories") == "phone accessory"
    assert retrieval_terms("cases for phones") == "case phone"
    assert exact_key("phones") != exact_key("phone")
    assert "non" in retrieval_terms("non-prescription food")


def test_prepared_draft_splits_are_separate_and_not_reviewed():
    base = Path("data/google/evaluation")
    report = check_splits(
        {
            "development": base / "development.jsonl",
            "calibration": base / "calibration.draft.jsonl",
            "test": base / "test.draft.jsonl",
        }
    )
    assert report["passed"] and not report["human_review_complete"]
    assert sum(v["examples"] for v in report["counts"].values()) == 600


def test_family_and_duplicate_query_leaks_detected(tmp_path):
    first = {
        "example_id": "1",
        "query": "mobile phone",
        "input_family": "phones",
        "taxonomy_key": "google",
        "taxonomy_version": "v1",
        "acceptable_category_ids": ["267"],
    }
    second = {**first, "example_id": "2", "query": "cellphone"}
    paths = {"development": tmp_path / "dev.jsonl", "test": tmp_path / "test.jsonl"}
    paths["development"].write_text(json.dumps(first))
    paths["test"].write_text(json.dumps(second))
    report = check_splits(paths)
    assert not report["passed"]
    assert any("target category family" in message for message in report["conflicts"])
