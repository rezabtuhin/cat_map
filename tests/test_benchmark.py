import json

import numpy as np

from category_matcher.benchmark import benchmark
from category_matcher.matcher import Matcher


def test_short_timing_run_does_not_claim_resource_qualification(bundle, tmp_path, monkeypatch):
    matcher = Matcher(bundle)
    path = tmp_path / "queries.jsonl"
    path.write_text(
        json.dumps(
            {"query": "mobil phones", "taxonomy_key": "google", "taxonomy_version": matcher.version}
        )
    )
    monkeypatch.setattr(matcher, "encode_queries", lambda texts: np.array([[1.0, 0.0, 0.0]]))
    monkeypatch.setattr("category_matcher.benchmark.swapouts", lambda: 0)
    try:
        report = benchmark(matcher, path, 2)
        assert report["concurrent_inference"] == 1
        assert report["requests"] == 2 and report["unique_inputs_available"] == 1
        assert not report["resource_targets_passed"]
        assert not report["checks"]["at_least_200_requests"]
        assert report["outcomes"] == {"provisional": 2}
    finally:
        matcher.close()
