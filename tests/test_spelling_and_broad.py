from pathlib import Path

import numpy as np

from category_matcher.domain import Category
from category_matcher.matcher import Matcher
from category_matcher.query import explicit_broad_category, literal_broad_label
from category_matcher.retrieval import LexicalRetriever, build_bm25
from category_matcher.taxonomy import read_google_taxonomy


def test_unique_spelling_is_a_provisional_candidate_not_an_alias(bundle):
    matcher = Matcher(bundle)
    try:
        result = matcher.match("mobil phones", query_vector=np.array([0.0, 1.0, 0.0]))
        assert result.status == "provisional" and result.category is None
        assert result.candidates[0]["id"] == "3"
        assert result.query_analysis["spelling"]["candidates"] == ["3"]
        # The semantic score remains evidence for the original input. It must
        # not become 1.0 by replacing the input with its proposed category label.
        assert result.candidates[0]["evidence"]["semantic"]["raw_score"] < 0.1
        assert matcher.match("mobil phones", method="exact").category is None
    finally:
        matcher.close()


def test_known_vocabulary_is_not_rewritten(bundle):
    matcher = Matcher(bundle)
    try:
        assert matcher.lexical.spelling("cases", matcher.store.allowed_ids(), 5)["candidates"] == []
        assert not matcher.lexical.spelling("cat", matcher.store.allowed_ids(), 5)["eligible"]
        assert matcher.lexical.spelling("mobil phones", {"5", "6", "7"}, 5)["candidates"] == []
    finally:
        matcher.close()


def test_explicit_general_wrapper_uses_canonical_label_and_preserves_scope(bundle):
    matcher = Matcher(bundle)
    try:
        result = matcher.match("mobile phones in general")
        assert result.status == "matched" and result.category["id"] == "3"
        assert result.method == "explicit_broad"
        assert matcher.match("general electronics category").category["id"] == "2"
        assert matcher.match("mobile phones in general", method="exact").category is None
        assert matcher.match("mobil phones", retrieve=False).reason_code == "retrieval_required"
        assert matcher.encoder is None
    finally:
        matcher.close()


def test_general_query_does_not_select_a_specialized_child():
    rows, _ = read_google_taxonomy(Path("google_category_list.txt"))
    categories = {c.category_id: c for c in rows}
    assert (
        explicit_broad_category("office supplies in general", categories, set(categories)) == "922"
    )
    assert (
        explicit_broad_category("baby and toddler products in general", categories, set(categories))
        == "537"
    )
    assert (
        explicit_broad_category("eco friendly clothing in general", categories, set(categories))
        is None
    )
    assert literal_broad_label("baby & toddler products in general") == "baby and toddler"
    assert (
        explicit_broad_category(
            "general general office supplies category", categories, set(categories)
        )
        == "932"
    )
    assert (
        explicit_broad_category("handheld devices in general", categories, set(categories)) == "298"
    )


def test_audit_does_not_turn_ai_proposals_into_human_review():
    import json

    records = [
        json.loads(line)
        for line in Path("data/google/evaluation/development.audited.draft.jsonl")
        .read_text()
        .splitlines()
    ]
    by_id = {row["example_id"]: row for row in records}
    assert by_id["draft-080"]["acceptable_category_ids"] == ["543543"]
    assert by_id["draft-113"]["expected_status"] == "needs_clarification"
    assert all(row["review_status"] == "pending_review" for row in records)


def test_spelling_ambiguity_includes_approved_aliases_and_respects_scope(tmp_path):
    cats = [
        Category("a", "Chairs", "Furniture > Chairs", None, 2, True),
        Category("b", "Chips", "Food > Chips", None, 2, True),
        Category("c", "Shirts & Tops", "Clothing > Shirts & Tops", None, 2, True),
        Category("d", "Toys", "Toys", None, 1, True),
    ]
    build_bm25(cats, tmp_path / "index")
    retriever = LexicalRetriever(
        tmp_path / "index",
        {c.category_id: c for c in cats},
        [{"alias": "Tops", "category_id": "c"}],
    )
    choices = retriever.spelling("Chirs", {"a", "b", "c", "d"}, 5)
    assert choices["eligible"] and choices["ambiguous"]
    assert choices["candidates"] == ["a", "b"]
    short = retriever.spelling("Tos", {"a", "b", "c", "d"}, 5)
    assert short["ambiguous"] and not short["eligible"]
    assert short["candidates"] == ["c", "d"]
    assert retriever.spelling("Chirs", {"a"}, 5)["candidates"] == ["a"]
