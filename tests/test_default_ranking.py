import json

import numpy as np

from category_matcher.matcher import Matcher


def test_default_ranking_is_lazy_and_can_be_disabled(bundle, monkeypatch):
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["ranking_model"] = {"name": "unit-test-model"}
    manifest_path.write_text(json.dumps(manifest))
    calls = []

    class Model:
        def __init__(self, metadata):
            calls.append(("load", metadata["name"]))

        def rank(self, query, candidates):
            calls.append(("rank", query))
            return candidates

    monkeypatch.setattr("category_matcher.reranker.CrossEncoderReranker", Model)
    matcher = Matcher(bundle)
    try:
        assert matcher.match("Mobile Phones").status == "matched"
        assert matcher.match("mobile phones in general").status == "matched"
        matcher.match("mobil phones", query_vector=np.array([1.0, 0.0, 0.0]), rerank=False)
        assert not calls
        result = matcher.match("mobil phones", query_vector=np.array([1.0, 0.0, 0.0]))
        assert result.method == "hybrid+rerank" and result.category is None
        assert calls == [("load", "unit-test-model"), ("rank", "mobil phones")]
    finally:
        matcher.close()
