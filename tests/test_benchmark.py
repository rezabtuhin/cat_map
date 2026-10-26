import json

import numpy as np
import pytest

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


def test_swap_observations_separate_idle_load_and_inference(bundle, tmp_path, monkeypatch):
    matcher = Matcher(bundle)
    path = tmp_path / "queries.jsonl"
    path.write_text(
        json.dumps(
            {"query": "mobil phones", "taxonomy_key": "google", "taxonomy_version": matcher.version}
        )
    )
    monkeypatch.setattr(matcher, "encode_queries", lambda texts: np.array([[1.0, 0.0, 0.0]]))
    counters = iter([10, 12, 15, 20, 20])
    monkeypatch.setattr("category_matcher.benchmark.swapouts", lambda: next(counters))
    monkeypatch.setattr("category_matcher.benchmark.time.sleep", lambda seconds: None)
    try:
        report = benchmark(matcher, path, 2, idle_baseline_seconds=5)
        phases = report["resource_diagnostics"]
        assert phases["idle_system_swapout_pages_delta"] == 2
        assert phases["load_and_warmup_system_swapout_pages_delta"] == 3
        assert phases["timed_inference_system_swapout_pages_delta"] == 5
        assert phases["inference_windows"][0]["system_swapout_pages_delta"] == 5
        assert report["system_swapout_pages_delta"] == 8
        assert not report["checks"]["no_observed_system_swapouts"]
        with pytest.raises(ValueError, match="Idle baseline"):
            benchmark(matcher, path, 2, idle_baseline_seconds=float("nan"))
    finally:
        matcher.close()
