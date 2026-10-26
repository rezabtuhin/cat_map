"""Serial, end-to-end local inference measurements, separate from label accuracy."""

import math
import platform
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


def swap_delta(before: int | None, after: int | None) -> int | None:
    return None if before is None or after is None else after - before


def benchmark(
    matcher: Matcher,
    dataset: Path,
    requests: int,
    *,
    rerank: bool | None = None,
    idle_baseline_seconds: float = 0,
) -> dict:
    if rerank is None:
        rerank = matcher.default_rerank
    if not 1 <= requests <= 10000:
        raise ValueError("Benchmark requests must be between 1 and 10000")
    if not math.isfinite(idle_baseline_seconds) or not 0 <= idle_baseline_seconds <= 30:
        raise ValueError("Idle baseline must be between 0 and 30 seconds")
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
            category_id=row.get("category_id"),
            context=row.get("context", ""),
            parent_id=row.get("parent_category_id"),
            retrieve=False,
        )
        if exact.reason_code == "retrieval_required":
            samples.append(row)
    if not samples:
        raise ValueError("Benchmark requires non-exact English queries")
    idle_before = swapouts()
    if idle_baseline_seconds:
        time.sleep(idle_baseline_seconds)
    swaps_before = swapouts()
    started = time.perf_counter()
    matcher.encode_queries(
        [matcher.retrieval_query(samples[0]["query"], samples[0].get("context", ""))]
    )
    matcher.match(
        samples[0]["query"],
        context=samples[0].get("context", ""),
        parent_id=samples[0].get("parent_category_id"),
        rerank=rerank,
    )
    warmup = time.perf_counter() - started
    swaps_after_warmup = swapouts()
    window_swaps = swaps_after_warmup
    windows = []
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
        if (index + 1) % 20 == 0 or index + 1 == requests:
            window_after = swapouts()
            window_start = len(windows) * 20
            windows.append(
                {
                    "first_request": window_start + 1,
                    "last_request": index + 1,
                    "p95_ms": float(np.percentile(durations[window_start:], 95)),
                    "system_swapout_pages_delta": swap_delta(window_swaps, window_after),
                }
            )
            window_swaps = window_after
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    swaps_after = swapouts()
    total_swap_delta = swap_delta(swaps_before, swaps_after)
    p95 = float(np.percentile(durations, 95))
    checks = {
        "at_least_200_requests": requests >= 200,
        "p95_under_3_seconds": p95 <= 3000,
        "peak_under_3GB": peak_bytes < 3_000_000_000,
        "no_observed_system_swapouts": total_swap_delta == 0,
    }
    return {
        "artifact_version": matcher.manifest["artifact_version"],
        "dataset_sha256": digest(dataset),
        "requests": requests,
        "unique_inputs_available": len(samples),
        "concurrent_inference": 1,
        "ranking_method": "hybrid+rerank" if rerank else "hybrid",
        "ranking_model": matcher.reranker_metadata if rerank else None,
        "policy_id": matcher.policy.document["policy_id"] if matcher.policy else None,
        "query_model_load_and_warmup_seconds": warmup,
        "model_load_and_warmup_seconds": warmup,
        "end_to_end_warm_latency_ms": {
            "p50": float(np.percentile(durations, 50)),
            "p95": p95,
            "max": max(durations),
        },
        "serial_requests_per_second": requests / (sum(durations) / 1000),
        "process_peak_resident_bytes": peak_bytes,
        "system_swapout_pages_delta": total_swap_delta,
        "resource_diagnostics": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "idle_baseline_seconds": idle_baseline_seconds,
            "idle_system_swapout_pages_delta": (
                swap_delta(idle_before, swaps_before) if idle_baseline_seconds else None
            ),
            "load_and_warmup_system_swapout_pages_delta": (
                swap_delta(swaps_before, swaps_after_warmup)
            ),
            "timed_inference_system_swapout_pages_delta": (
                swap_delta(swaps_after_warmup, swaps_after)
            ),
            "inference_windows": windows,
        },
        "checks": checks,
        "resource_targets_passed": all(checks.values()),
        "outcomes": outcomes,
        "notes": [
            "Each timed request includes query encoding, retrieval, and selected ranking/policy.",
            "Inference requests run serially; this does not establish multi-user capacity.",
            "Repeated draft queries measure execution cost, not matching accuracy.",
            "Peak memory is for this process; system swap counters can include other applications.",
            "Model loading and the first prediction occur in the untimed warmup.",
            "Idle/load/inference swap observations are system-wide and do not prove attribution.",
        ],
    }
