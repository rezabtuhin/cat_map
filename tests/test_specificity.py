import numpy as np
import pytest

from category_matcher.matcher import Matcher
from category_matcher.query import scope_only_entailment_support


class ScopeOnlyRanker:
    def __init__(self, child_support=-2):
        self.child_support = child_support

    def rank(self, query, candidates):
        candidates = sorted(candidates, key=lambda candidate: candidate["id"] != "2")
        return [
            {
                **candidate,
                "rank": i,
                "entailment_logit_margin": 3 if candidate["id"] == "2" else self.child_support,
                "reranker_score": 1 / i,
            }
            for i, candidate in enumerate(candidates, 1)
        ]


@pytest.mark.parametrize("scope", [{"parent_id": "2"}, {"context": "Electronics"}])
def test_scope_only_model_support_keeps_retrieval_order_and_exposes_uncertainty(bundle, scope):
    matcher = Matcher(bundle, reranker=ScopeOnlyRanker())
    try:
        vector = np.array([1.0, 0.0, 0.0])
        baseline = matcher.match("portable handsets", query_vector=vector, rerank=False, **scope)
        result = matcher.match("portable handsets", query_vector=vector, rerank=True, **scope)
        assert result.status == "provisional" and result.category is None
        assert [c["id"] for c in result.candidates] == [c["id"] for c in baseline.candidates]
        assert result.candidates[0]["id"] == "3"
        assert result.candidates[0]["model_rank"] > 1
        assert result.query_analysis["specificity_conflict"]["parent_id"] == "2"
        assert result.clarification and "uncertain" in result.clarification
        assert result.candidates[0]["evidence"] == baseline.candidates[0]["evidence"]
    finally:
        matcher.close()


def test_positive_child_evidence_does_not_trigger_scope_fallback(bundle):
    matcher = Matcher(bundle, reranker=ScopeOnlyRanker(child_support=1))
    try:
        result = matcher.match(
            "portable handsets", parent_id="2", query_vector=np.array([1.0, 0.0, 0.0]), rerank=True
        )
        assert result.candidates[0]["id"] == "2"
        assert "specificity_conflict" not in result.query_analysis
    finally:
        matcher.close()


def test_unscored_candidates_and_missing_scope_do_not_imply_rejected_descendants():
    candidates = [
        {"id": "parent", "entailment_logit_margin": 3},
        {"id": "child", "entailment_logit_margin": -2},
        {"id": "tail", "entailment_logit_margin": None},
    ]
    assert not scope_only_entailment_support(candidates, "parent")
    assert not scope_only_entailment_support(candidates, None)
    candidates[-1]["entailment_logit_margin"] = -1
    assert scope_only_entailment_support(candidates, "parent")
    candidates[0]["entailment_logit_margin"] = -3
    assert not scope_only_entailment_support(candidates, "parent")
