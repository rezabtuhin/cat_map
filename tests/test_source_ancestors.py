import asyncio

import numpy as np
import pytest
from pydantic import ValidationError
from test_api import client_for, factory

from category_matcher.api import create_app
from category_matcher.api.schemas import MatchRequest
from category_matcher.api.settings import APISettings
from category_matcher.matcher import Matcher
from category_matcher.query import source_supports_lookup
from category_matcher.review import review_input


@pytest.mark.parametrize(
    "ancestors", ["Clothing", [1], [None], [" "], ["a"] * 21, ["a" * 301], ["a" * 300] * 4]
)
def test_invalid_source_ancestry_is_rejected(ancestors):
    with pytest.raises(ValidationError):
        MatchRequest(taxonomy_key="google", category="Tops", source_ancestors=ancestors)


def test_store_ancestors_preserve_supported_lookup_and_never_force_a_google_scope(bundle):
    matcher = Matcher(bundle)
    try:
        matched = matcher.match("Mobile Phones", source_ancestors=["Products", "Electronics"])
        assert matched.status == "matched" and matched.category["id"] == "3"
        result = matcher.match(
            "Mobile Phones",
            source_ancestors=["Pet Supplies"],
            query_vector=np.array([1.0, 0.0, 0.0]),
        )
        assert result.status == "provisional" and result.category is None
        assert "3" in {c["id"] for c in result.candidates}
        assert "5" in {c["id"] for c in result.candidates}
        assert not source_supports_lookup(
            ["Electronics", "Products"], matcher.store.categories["3"]
        )
        assert (
            matcher.match("Collections", source_ancestors=["Electronics"]).status
            == "needs_clarification"
        )
        assert matcher.match("Mobile Phones", source_ancestors=["Collections"]).status == "matched"
        # Source IDs have no role in these string-only breadcrumbs.
        assert (
            matcher.match(
                "Mobile Phones", source_ancestors=["3"], query_vector=np.array([1.0, 0.0, 0.0])
            ).category
            is None
        )
        query = matcher.retrieval_query("portable handsets", source_ancestors=["Electronics"])
        assert "Store ancestry: Electronics" in query
    finally:
        matcher.close()


def test_inline_context_and_new_source_ancestry_can_coexist(bundle):
    matcher = Matcher(bundle)
    try:
        result = matcher.match("Mobile Phones|Electronics", source_ancestors=["Products"])
        assert result.category["id"] == "3"
        assert matcher.match("Mobile Phones").category == result.category
    finally:
        matcher.close()


def test_api_keeps_authoritative_hierarchy_in_runtime_event(bundle, tmp_path):
    app = create_app(
        APISettings(runtime_db=tmp_path / "runtime.sqlite3"), matcher_factory=factory(bundle)
    )

    async def scenario():
        async with client_for(app) as client:
            payload = {
                "taxonomy_key": "google",
                "category": "Mobile Phones",
                "source_ancestors": [" Products ", "Electronics"],
            }
            response = await client.post("/v1/categories/match", json=payload)
            assert response.status_code == 200
            result = response.json()
            assert result["category"]["id"] == "3"
            stored = app.state.service.runtime.event(result["event_id"])
            assert stored["request"]["source_ancestors"] == ["Products", "Electronics"]
            ready = (await client.get("/health/ready")).json()
            assert ready["capabilities"]["source_ancestors"] is True

    asyncio.run(scenario())


def test_legacy_review_bindings_remain_compatible_and_new_ancestry_is_bound():
    old = {"query": "Phones"}
    assert "source_ancestors" not in review_input(old)
    assert review_input({**old, "source_ancestors": ["Electronics"]})["source_ancestors"] == [
        "Electronics"
    ]
