import sqlite3

import numpy as np
import pytest

from category_matcher.artifacts import load_manifest
from category_matcher.matcher import Matcher
from category_matcher.storage import CategoryStore, connect, vector_blob
from category_matcher.verification import verify


def test_vector_search_groups_views_and_filters_scope_before_limit(bundle):
    manifest = load_manifest(bundle)
    store = CategoryStore(bundle / "taxonomy.sqlite3", "google", manifest["taxonomy_version"], 3)
    try:
        hits = store.semantic_search(np.array([1, 0, 0]), 3)
        assert len({hit.category_id for hit in hits}) == 3
        assert hits[0].category_id == "3"
        scoped = store.semantic_search(np.array([1, 0, 0]), 1, "5")
        assert len(scoped) == 1 and scoped[0].category_id in {"5", "6", "7"}
        assert store.allowed_ids("6") == {"6"}
        with pytest.raises(ValueError):
            store.semantic_search(np.array([1, 0, 0]), 1, "missing")
    finally:
        store.close()


@pytest.mark.parametrize(
    "vector", [[0, 0, 0], [1, float("nan"), 0], [1, 0], [float("inf"), 1, 0], [[1, 0, 0]]]
)
def test_invalid_vectors_rejected(vector):
    with pytest.raises(ValueError):
        vector_blob(np.array(vector), 3)


def test_schema_dimension_and_foreign_keys(bundle):
    manifest = load_manifest(bundle)
    connection = connect(bundle / "taxonomy.sqlite3")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO category_embeddings VALUES (?, ?, ?, ?, ?)",
                (
                    "google",
                    manifest["taxonomy_version"],
                    "unknown",
                    "label",
                    vector_blob(np.array([1, 0, 0]), 3),
                ),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE category_embeddings SET embedding=?",
                (np.array([1, 0], dtype="<f4").tobytes(),),
            )
    finally:
        connection.close()


def test_sqlite_scores_and_round_trips_agree_with_independent_reference(bundle):
    matcher = Matcher(bundle)
    try:
        report = verify(matcher)
        assert report["status"] == "passed"
        assert report["canonical_roundtrip_checks"] == 21
    finally:
        matcher.close()


def test_corrupt_artifact_rejected(bundle):
    (bundle / "aliases.json").write_text("[]")
    with pytest.raises(ValueError, match="integrity"):
        load_manifest(bundle)
