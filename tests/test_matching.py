import numpy as np
import pytest

from category_matcher.domain import Hit
from category_matcher.matcher import Matcher
from category_matcher.retrieval import expand_parents, fuse


def test_exact_alias_id_and_ambiguity(bundle):
    matcher = Matcher(bundle)
    try:
        for query in ("mobile phones", "Products > Electronics > Mobile Phones", "cellphones"):
            result = matcher.match(query)
            assert result.status == "matched" and result.category["id"] == "3"
        assert matcher.match(category_id="4").category["id"] == "4"
        assert matcher.match("conflicting alias", method="exact").category is None
        assert (
            matcher.match("Mobile Phones", method="exact", context="I need covers").category is None
        )
        assert matcher.match("Mobile", language="bn").status == "unsupported_language"
    finally:
        matcher.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"text": "   "},
        {"category_id": "404"},
        {"text": "Mobile Phones", "category_id": "4"},
        {"text": "Mobile Phones", "parent_id": "5"},
        {"text": "x" * 301},
        {"text": "phone", "context": "x" * 1001},
        {"text": "phone", "method": "invalid"},
        {"text": "phone", "parent_id": "404"},
    ],
)
def test_invalid_or_conflicting_inputs(bundle, kwargs):
    matcher = Matcher(bundle)
    try:
        with pytest.raises(ValueError):
            matcher.match(**kwargs)
    finally:
        matcher.close()


def test_provisional_branches_and_parent_constraint(bundle):
    matcher = Matcher(bundle)
    try:
        for method in ("fuzzy", "bm25", "semantic", "hybrid"):
            result = matcher.match("mobil phones", method=method, query_vector=np.array([1, 0, 0]))
            assert result.status == "provisional" and result.category is None
            assert "3" in [c["id"] for c in result.candidates]
            assert len({c["id"] for c in result.candidates}) == len(result.candidates)
        scoped = matcher.match("toyz", parent_id="5", query_vector=np.array([1, 0, 0]))
        assert all(c["id"] in {"5", "6", "7"} for c in scoped.candidates)
        assert matcher.match("zxqvplm", method="bm25").candidates == []
    finally:
        matcher.close()


def test_fusion_deduplicates_votes_and_expansion_is_bounded(bundle):
    duplicate = Hit("3", 100, "fuzzy", 1)
    hits = fuse({"fuzzy": [duplicate, duplicate], "bm25": [Hit("4", 1, "bm25", 1)]}, 40)
    assert hits[0].score == hits[1].score
    matcher = Matcher(bundle)
    try:
        original = [Hit("3", 1, "semantic", 1), Hit("6", 0.9, "semantic", 2)]
        expanded = expand_parents(
            original, matcher.store.categories, matcher.store.allowed_ids(), 1
        )
        assert len(expanded) == 3
        assert expanded[-1].category_id == "2"
    finally:
        matcher.close()


def test_pipe_context_filters_before_ranking_and_keeps_attributes_broad(bundle):
    matcher = Matcher(bundle)
    try:
        result = matcher.match("mobil phones|Electronics", query_vector=np.array([0, 1, 0]))
        assert result.candidates
        assert all(c["id"] in {"2", "3", "4"} for c in result.candidates)
        broad = matcher.match("Eco Friendly|Electronics")
        assert broad.status == "provisional" and broad.category is None
        assert broad.candidates[0]["id"] == "2"
        assert matcher.match("Mobile Phones|Electronics").category["id"] == "3"
        assert matcher.retrieval_query("Hoodies|Electronics") == "Hoodies"
        assert matcher.retrieval_query("Hoodies|Electronics", "for babies") == (
            "Hoodies. Context: for babies"
        )
        with pytest.raises(ValueError, match="contradicts"):
            matcher.match("Cat Toys|Electronics")
        with pytest.raises(ValueError, match="nonempty context"):
            matcher.match("Tops|")
    finally:
        matcher.close()


def test_baby_apparel_is_not_inferred_from_generic_hoodies():
    from pathlib import Path

    from category_matcher.query import unsupported_audience_ids
    from category_matcher.taxonomy import read_google_taxonomy

    categories, _ = read_google_taxonomy(Path("google_category_list.txt"))
    by_id = {c.category_id: c for c in categories}
    excluded = unsupported_audience_ids("Hoodies & Sweatshirts. Context: Clothing", by_id)
    assert "5410" in excluded and "212" not in excluded
    assert not unsupported_audience_ids("hoodies for babies", by_id)


def test_injected_reranker_cannot_turn_candidates_into_confirmed_matches(bundle):
    class ReverseReranker:
        def rank(self, query, candidates):
            assert query == "mobil phones"
            return list(reversed(candidates))

    matcher = Matcher(bundle, ReverseReranker(), {"name": "unit-test-double"})
    try:
        result = matcher.match("mobil phones", method="fuzzy", rerank=True)
        assert result.status == "provisional" and result.category is None
        assert result.method == "fuzzy+rerank"
        assert result.ranking_model == {"name": "unit-test-double"}
    finally:
        matcher.close()
