import json

import pytest

from category_matcher.cli import main
from category_matcher.decision import DecisionPolicy
from category_matcher.matcher import Matcher


@pytest.mark.parametrize("query", ["Collections", "Erin Recommends", "New Luma Yoga Collection"])
def test_non_product_titles_stay_unmapped_even_with_permissive_policy(bundle, query):
    matcher = Matcher(bundle)
    matcher.policy = DecisionPolicy(
        {
            "ranking_method": "hybrid",
            "ranking_model": None,
            "policy_id": "permissive-test",
            "thresholds": {
                "min_semantic": -1,
                "min_margin": 0,
                "no_match_semantic": -1,
                "fuzzy_support": 0,
            },
        },
        experimental=False,
    )
    try:
        result = matcher.match(query)
        assert result.status == "needs_clarification"
        assert result.category is None and result.candidates == []
        assert matcher.encoder is None
    finally:
        matcher.close()


@pytest.mark.parametrize(
    "query,expected_id", [("Collections", None), ("Erin Recommends", None), ("Mobile Phones", "3")]
)
def test_cli_serializes_only_accepted_categories_as_assignments(
    bundle, monkeypatch, capsys, query, expected_id
):
    (bundle.parent / "active.json").write_text(json.dumps({"artifact_version": bundle.name}))
    monkeypatch.setattr(
        "sys.argv", ["category-matcher", "--artifacts", str(bundle.parent), "match", query]
    )
    assert main() == 0
    output = json.loads(capsys.readouterr().out)
    if expected_id is None:
        assert output["status"] == "needs_clarification" and output["category"] is None
    else:
        assert output["status"] == "matched" and output["category"]["id"] == expected_id
