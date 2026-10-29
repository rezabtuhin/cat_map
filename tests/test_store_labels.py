from pathlib import Path

import numpy as np
import pytest

from category_matcher.domain import Category
from category_matcher.matcher import Matcher
from category_matcher.query import literal_group_candidates, weak_promotion_evidence
from category_matcher.taxonomy import read_google_taxonomy


def test_real_taxonomy_coordinated_nouns_preserve_qualifiers_and_scope():
    rows, _ = read_google_taxonomy(Path("google_category_list.txt"))
    cats = {c.category_id: c for c in rows}
    assert literal_group_candidates("Jackets", cats, set(cats)) == ["5598"]
    assert literal_group_candidates("Fitness Equipment", cats, set(cats)) == ["2890", "990"]
    assert literal_group_candidates("Life Jackets", cats, set(cats)) == []
    assert literal_group_candidates("Fitness Equipment Mats", cats, set(cats)) == []
    assert literal_group_candidates("Jackets", cats, {"1128"}) == []


@pytest.mark.parametrize("query", ["Collections", "Erin Recommends", "New Luma Yoga Collection"])
def test_store_section_titles_request_context_without_loading_models(bundle, query):
    matcher = Matcher(bundle)
    try:
        assert (
            matcher.retrieval_query("Erin Recommends", "portable handsets") == "portable handsets"
        )
        result = matcher.match(query)
        assert result.status == "needs_clarification" and result.category is None
        assert result.candidates == [] and result.clarification
        assert matcher.encoder is None and matcher.reranker is None
    finally:
        matcher.close()


def test_product_description_can_resolve_store_title_without_treating_title_as_product(bundle):
    matcher = Matcher(bundle)
    try:
        result = matcher.match(
            "Erin Recommends",
            context="portable handsets",
            query_vector=np.array([1.0, 0.0, 0.0]),
            rerank=False,
        )
        assert result.candidates[0]["id"] == "3"
        assert result.query_analysis["merchandising_label"]["used_product_context"]
        assert result.status == "provisional"
    finally:
        matcher.close()


def test_promotion_guard_requires_both_semantic_scores_and_weak_model_support():
    first = {"id": "a", "evidence": {"semantic": {"raw_score": 0.8}}}
    other = {
        "id": "b",
        "evidence": {"semantic": {"raw_score": 0.7}},
        "entailment_logit_margin": 0.5,
    }
    guard = {"minimum_semantic_gap": 0.03, "minimum_entailment_support": 1}
    assert weak_promotion_evidence(first, other, guard)["retrieval_first_id"] == "a"
    other["entailment_logit_margin"] = 2
    assert weak_promotion_evidence(first, other, guard) is None
    del first["evidence"]
    other["entailment_logit_margin"] = 0.5
    assert weak_promotion_evidence(first, other, guard) is None


def test_literal_group_priority_survives_a_model_that_prefers_a_specific_child(bundle):
    class Ranker:
        def rank(self, query, candidates):
            ranked = sorted(candidates, key=lambda c: c["id"] != "4")
            return [
                {**c, "rank": i, "entailment_logit_margin": 3, "reranker_score": 1 / i}
                for i, c in enumerate(ranked, 1)
            ]

    matcher = Matcher(bundle, reranker=Ranker())
    matcher.store.categories["3"] = Category(
        "3", "Handsets & Phones", "Products > Electronics > Handsets & Phones", "2", 3, True
    )
    try:
        result = matcher.match(
            "Handsets", parent_id="2", query_vector=np.array([1.0, 0.0, 0.2]), rerank=True
        )
        assert result.candidates[0]["id"] == "3"
        assert result.status == "provisional" and result.category is None
        assert result.query_analysis["literal_group"]["candidate_ids"] == ["3"]
        assert {c["id"] for c in result.candidates} <= {"2", "3", "4"}
    finally:
        matcher.close()
