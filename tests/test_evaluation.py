import json

import pytest

from category_matcher.artifacts import read_aliases
from category_matcher.evaluation import read_examples, summarize
from category_matcher.matcher import Matcher


def test_draft_labels_cannot_be_silently_evaluated_as_ground_truth(bundle, tmp_path):
    matcher = Matcher(bundle)
    path = tmp_path / "draft.jsonl"
    row = {
        "example_id": "1",
        "taxonomy_key": "google",
        "taxonomy_version": matcher.version,
        "query": "phone",
        "expected_status": "matched",
        "acceptable_category_ids": ["3"],
        "review_status": "pending_review",
    }
    path.write_text(json.dumps(row) + "\n")
    try:
        with pytest.raises(ValueError, match="Unreviewed"):
            read_examples(path, matcher, False)
        assert len(read_examples(path, matcher, True)) == 1
        row["acceptable_category_ids"] = ["not-present"]
        path.write_text(json.dumps(row) + "\n")
        with pytest.raises(ValueError, match="invalid expected"):
            read_examples(path, matcher, True)
    finally:
        matcher.close()


def test_pending_aliases_do_not_enter_automatic_lookup(tmp_path):
    path = tmp_path / "aliases.jsonl"
    alias = {
        "taxonomy_key": "google",
        "alias": "phones",
        "category_id": "3",
        "status": "pending_review",
    }
    path.write_text(json.dumps(alias))
    assert read_aliases(path, {"3"}) == ([], 1)
    alias["status"] = "approved"
    path.write_text(json.dumps(alias))
    with pytest.raises(ValueError, match="reviewer"):
        read_aliases(path, {"3"})


def test_abstention_is_not_scored_as_perfect_precision():
    empty = summarize([])
    assert empty["automatic_match_precision"] is None
    assert empty["top1_accuracy"] is None
