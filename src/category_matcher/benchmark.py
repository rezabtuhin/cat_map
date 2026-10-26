"""Serial, end-to-end local inference measurements, separate from label accuracy."""

import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from category_matcher.artifacts import digest
from category_matcher.datasets import read_jsonl
from category_matcher.matcher import Matcher


def swapouts() -> int | None:
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(["/usr/bin/vm_stat"], capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if line.startswith("Swapouts:"):
                return int(line.split(":", 1)[1].strip().rstrip("."))
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None
    return None


def benchmark(matcher: Matcher, dataset: Path, requests: int, *, rerank: bool = False) -> dict:
    if not 1 <= requests <= 10000:
        raise ValueError("Benchmark requests must be between 1 and 10000")
    rows = read_jsonl(dataset)
    for row in rows:
        if (row["taxonomy_key"], row["taxonomy_version"]) != ("google", matcher.version):
            raise ValueError("Benchmark input taxonomy/version mismatch")
    samples = []
    for row in rows:
        if row.get("language", "en") != "en":
            continue
        exact = matcher.match(
            row["query"],
            method="exact",
            category_id=row.get("category_id"),
            context=row.get("context", ""),
            parent_id=row.get("parent_category_id"),
        )
        if exact.status == "provisional":
            samples.append(row)
    if not samples:
        raise ValueError("Benchmark requires non-exact English queries")
    swaps_before = swapouts()
    started = time.perf_counter()
    matcher.encode_queries(
        [matcher.retrieval_query(samples[0]["query"], samples[0].get("context", ""))]
    )
    warmup = time.perf_counter() - started
    durations, outcomes = [], {}
    for index in range(requests):
        row = samples[index % len(samples)]
        started = time.perf_counter()
        result = matcher.match(
            row["query"],
            context=row.get("context", ""),
            parent_id=row.get("parent_category_id"),
            rerank=rerank,
        )
        durations.append((time.perf_counter() - started) * 1000)
        outcomes[result.status] = outcomes.get(result.status, 0) + 1
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    swaps_after = swapouts()
    swap_delta = None if swaps_before is None or swaps_after is None else swaps_after - swaps_before
    p95 = float(np.percentile(durations, 95))
    checks = {
        "at_least_200_requests": requests >= 200,
        "p95_under_3_seconds": p95 <= 3000,
        "peak_under_3GB": peak_bytes < 3_000_000_000,
        "no_observed_system_swapouts": swap_delta == 0,
    }
    return {
        "artifact_version": matcher.manifest["artifact_version"],
        "dataset_sha256": digest(dataset),
        "requests": requests,
        "unique_inputs_available": len(samples),
        "concurrent_inference": 1,
        "ranking_method": "hybrid+rerank" if rerank else "hybrid",
        "policy_id": matcher.policy.document["policy_id"] if matcher.policy else None,
        "query_model_load_and_warmup_seconds": warmup,
        "end_to_end_warm_latency_ms": {
            "p50": float(np.percentile(durations, 50)),
            "p95": p95,
            "max": max(durations),
        },
        "serial_requests_per_second": requests / (sum(durations) / 1000),
        "process_peak_resident_bytes": peak_bytes,
        "system_swapout_pages_delta": swap_delta,
        "checks": checks,
        "resource_targets_passed": all(checks.values()),
        "outcomes": outcomes,
        "notes": [
            "Each timed request includes query encoding, retrieval, and selected ranking/policy.",
            "Inference requests run serially; this does not establish multi-user capacity.",
            "Repeated draft queries measure execution cost, not matching accuracy.",
            "Peak memory is for this process; system swap counters can include other applications.",
            "Encoder load is separate; CLI adapter initialization occurs before this function.",
        ],
    }
