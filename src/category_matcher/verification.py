"""Validate the real built artifacts without downloading or loading a model."""

import numpy as np

from category_matcher.matcher import Matcher
from category_matcher.normalization import exact_key
from category_matcher.retrieval import cosine_reference


def verify(matcher: Matcher) -> dict:
    checks = 0
    for cid, category in matcher.store.categories.items():
        for text in (category.full_path, category.label):
            if len(matcher.exact[exact_key(text)]) != 1:
                continue
            result = matcher.match(text, method="exact")
            if result.category is None or result.category["id"] != cid:
                raise ValueError(f"Canonical text round trip failed for {cid}")
            checks += 1
        result = matcher.match(category_id=cid, method="exact")
        if result.category is None or result.category["id"] != cid:
            raise ValueError(f"Canonical ID round trip failed for {cid}")
        checks += 1
    store = matcher.store
    rows = store.connection.execute(
        "SELECT category_id, embedding FROM category_embeddings "
        "WHERE taxonomy_key=? AND taxonomy_version=? ORDER BY category_id, view",
        (store.taxonomy_key, store.version),
    ).fetchall()
    if len(rows) != matcher.manifest["embedding_rows"]:
        raise ValueError("Embedding count differs from the manifest")
    vectors = np.array([np.frombuffer(row[1], dtype="<f4") for row in rows])
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all() or not np.allclose(norms, 1, atol=1e-5):
        raise ValueError("Invalid or unnormalized stored vectors")
    generator = np.random.default_rng(41)
    numerical_checks = 0
    for index in generator.choice(len(rows), size=min(8, len(rows)), replace=False):
        query = vectors[index]
        for parent in (None, rows[index][0]):
            allowed = store.allowed_ids(parent)
            reference = cosine_reference(vectors, query)
            grouped = {}
            for row, score in zip(rows, reference, strict=True):
                if row[0] in allowed:
                    grouped[row[0]] = max(grouped.get(row[0], -float("inf")), float(score))
            actual = store.semantic_search(query, min(10, len(allowed)), parent)
            for hit in actual:
                if not np.isclose(hit.score, grouped[hit.category_id], atol=2e-6):
                    raise ValueError("SQLite cosine scores differ from NumPy")
            # Near-ties are numerically equivalent; compare cutoff scores rather than
            # requiring different float32 accumulation orders to break ties identically.
            expected_scores = sorted(grouped.values(), reverse=True)[: len(actual)]
            if not np.allclose([hit.score for hit in actual], expected_scores, atol=2e-6):
                raise ValueError("SQLite scoped ranking differs from exact NumPy ranking")
            numerical_checks += 1
    return {
        "artifact_version": matcher.manifest["artifact_version"],
        "category_count": len(store.categories),
        "canonical_roundtrip_checks": checks,
        "embedding_count": len(rows),
        "normalized_embeddings": True,
        "sqlite_vs_numpy_scoped_queries": numerical_checks,
        "status": "passed",
    }
