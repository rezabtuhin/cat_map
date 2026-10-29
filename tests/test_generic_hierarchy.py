import asyncio

import pytest
from test_api import client_for, factory

from category_matcher.api import create_app
from category_matcher.api.settings import APISettings
from category_matcher.matcher import Matcher
from category_matcher.query import is_placeholder_label


@pytest.mark.parametrize(
    "label",
    [
        "Uncategorized",
        "Uncategorised",
        "Miscellaneous",
        "Other",
        "N/A",
        "--",
        "123",
        "Unknown products",
    ],
)
def test_catch_all_labels_do_not_load_models_or_offer_random_candidates(bundle, label):
    matcher = Matcher(bundle)
    try:
        result = matcher.match(label, source_ancestors=["Electronics", "Mobile Phones"])
        assert result.status == "needs_clarification"
        assert result.category is None and result.candidates == []
        assert result.reason_code == "placeholder_label_requires_product_context"
        assert matcher.encoder is None and matcher.reranker is None
        assert not is_placeholder_label("Other Musical Instruments")
    finally:
        matcher.close()


@pytest.mark.parametrize("label", ["Collections", "Erin Recommends", "Sale", "Eco Friendly", "Men"])
def test_nonproduct_labels_use_the_nearest_grounded_ancestor_provisionally(bundle, label):
    matcher = Matcher(bundle)
    try:
        result = matcher.match(
            label, source_ancestors=["Electronics", "Men", "cellphones|Collections"]
        )
        assert result.status == "provisional" and result.category is None
        assert result.candidates[0]["id"] == "3"
        assert result.method == "source_ancestor"
        assert matcher.encoder is None and matcher.reranker is None
    finally:
        matcher.close()


@pytest.mark.parametrize(
    "label,ancestors",
    [
        ("Collections", ["Electronics", "Pet Supplies"]),
        ("Collections", ["Unknown store section"]),
        ("New Cat Toys Collection", ["Electronics", "Mobile Phones"]),
        ("Collections", ["Electronics", "Mobile Phones", "Dog Food"]),
        ("Collections", ["Electronics", "Mobile Phones", "mystery cat toys"]),
        ("Collections", ["Electronics", "conflicting alias"]),
    ],
)
def test_ambiguous_or_conflicting_source_context_never_creates_a_selection(
    bundle, label, ancestors
):
    matcher = Matcher(bundle)
    try:
        result = matcher.match(label, source_ancestors=ancestors)
        assert result.status == "needs_clarification" and result.candidates == []
        assert matcher.encoder is None and matcher.reranker is None
    finally:
        matcher.close()


def test_store_parentage_does_not_force_canonical_siblings_into_the_same_subtree(bundle):
    matcher = Matcher(bundle)
    try:
        result = matcher.match(
            "Collections", source_ancestors=["Electronics", "Mobile Phones", "Mobile Phone Cases"]
        )
        assert result.status == "provisional" and result.candidates[0]["id"] == "4"
        result = matcher.match(
            "Collections", parent_id="5", source_ancestors=["Electronics", "Mobile Phones"]
        )
        assert result.status == "needs_clarification" and result.candidates == []
    finally:
        matcher.close()


def test_generic_label_api_selects_inherited_candidate_and_keeps_catch_all_empty(bundle, tmp_path):
    app = create_app(
        APISettings(runtime_db=tmp_path / "runtime.sqlite3"), matcher_factory=factory(bundle)
    )

    async def scenario():
        async with client_for(app) as client:
            for label, expected in [("Collections", "3"), ("Uncategorized", None)]:
                response = await client.post(
                    "/v1/categories/match",
                    json={
                        "taxonomy_key": "google",
                        "category": label,
                        "source_ancestors": ["Electronics", "cellphones"],
                    },
                )
                assert response.status_code == 200
                result = response.json()
                assert result["category"] is None
                assert result["requires_review"] is (expected is not None)
                if expected:
                    assert result["provisional_category"]["id"] == expected
                else:
                    assert result["provisional_category"] is None and result["alternatives"] == []

    asyncio.run(scenario())
