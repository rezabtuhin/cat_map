"""Run a loopback server, compare frozen CLI outcomes, and measure HTTP overhead."""

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import httpx
import numpy as np

from category_matcher.artifacts import digest, load_manifest, resolve_bundle
from category_matcher.benchmark import swap_delta, swapouts
from category_matcher.datasets import read_jsonl


def percentiles(values):
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": max(values),
    }


async def run_checks(url, output, manifest, count):
    async with httpx.AsyncClient(base_url=url, timeout=45) as client:
        startup = time.perf_counter()
        observed_starting = False
        while True:
            try:
                response = await client.get("/health/ready")
                if response.status_code == 200:
                    break
                if response.status_code == 503:
                    observed_starting = True
            except httpx.TransportError:
                pass
            if time.perf_counter() - startup > 90:
                raise RuntimeError("API did not become ready; inspect server.log")
            await asyncio.sleep(0.1)
        ready_seconds = time.perf_counter() - startup
        readiness = response.json()
        assert readiness["artifact_version"] == manifest["artifact_version"]
        assert readiness["semantic_auto_acceptance_enabled"] is False
        (output / "openapi.json").write_text(
            json.dumps((await client.get("/openapi.json")).json(), indent=2) + "\n"
        )
        print(f"Ready in {ready_seconds:.2f}s; comparing frozen CLI inputs", flush=True)
        dataset = Path("data/google/evaluation/development.jsonl")
        rows = read_jsonl(dataset)
        reference_file = Path("reports/sprint_2/revision_3/retrieval_comparison.json")
        reference = json.loads(reference_file.read_text())
        if reference["artifact_version"] != manifest["artifact_version"] or reference[
            "dataset_sha256"
        ] != digest(dataset):
            raise RuntimeError(
                "CLI reference does not belong to the current frozen artifact/dataset"
            )
        by_id = {r["example_id"]: r for r in reference["examples"]["hybrid_rerank"]}
        equivalence, model_rows = [], []
        for i, row in enumerate(rows, 1):
            payload = {
                "taxonomy_key": "google",
                "category": row["query"],
                "category_id": row.get("category_id"),
                "context": row.get("context"),
                "parent_category_id": row.get("parent_category_id"),
                "language": row.get("language", "en"),
            }
            response = await client.post("/v1/categories/match", json=payload)
            response.raise_for_status()
            actual = response.json()
            expected = by_id[row["example_id"]]
            assigned_id = actual["category"]["id"] if actual["category"] else None
            expected_alternatives = (
                expected["candidate_ids"][:3]
                if expected["actual_status"] in {"provisional", "needs_clarification"}
                else []
            )
            equivalent = (
                actual["status"] == expected["actual_status"]
                and assigned_id == expected["accepted_id"]
                and actual["method"] == expected["actual_method"]
                and actual["reason_code"] == expected["reason_code"]
            )
            assert actual["status"] == "matched" or actual["category"] is None
            equivalence.append(
                {
                    "example_id": row["example_id"],
                    "semantic_outcome_agrees": equivalent,
                    "alternatives_agree": [c["id"] for c in actual["alternatives"]]
                    == expected_alternatives,
                    "response": actual,
                }
            )
            if actual["method"] == "hybrid+rerank" and actual["reason_code"] not in {
                "explicit_broad_category",
                "attribute_only_use_broad_context_provisionally",
            }:
                model_rows.append(payload)
            if i % 40 == 0:
                print(f"Equivalence {i}/{len(rows)}", flush=True)
        (output / "cli_equivalence.json").write_text(
            json.dumps(
                {
                    "artifact_version": manifest["artifact_version"],
                    "cli_reference_sha256": digest(reference_file),
                    "dataset_sha256": digest(dataset),
                    "examples": equivalence,
                    "semantic_disagreements": sum(
                        not r["semantic_outcome_agrees"] for r in equivalence
                    ),
                    "alternative_disagreements": sum(
                        not r["alternatives_agree"] for r in equivalence
                    ),
                    "notes": [
                        "Compares API behavior with frozen CLI predictions; labels are unreviewed.",
                        "Test-set predictions are not used.",
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        users = []
        for name in ("user_examples.jsonl", "user_examples_2.jsonl"):
            for row in read_jsonl(Path("data/google/evaluation") / name):
                response = await client.post(
                    "/v1/categories/match",
                    json={"taxonomy_key": "google", "category": row["query"]},
                )
                response.raise_for_status()
                result = response.json()
                assert result["status"] == "matched" or result["category"] is None
                users.append({"query": row["query"], "response": result})
        (output / "user_examples.json").write_text(
            json.dumps(
                {"artifact_version": manifest["artifact_version"], "examples": users}, indent=2
            )
            + "\n"
        )
        print(f"Benchmarking {count} serial model-path HTTP requests", flush=True)
        memory_before = (await client.get("/health/ready")).json()["process_memory"]
        before_swap = swapouts()
        latencies, queues, inference, overhead = [], [], [], []
        for i in range(count):
            started = time.perf_counter()
            response = await client.post(
                "/v1/categories/match", json=model_rows[i % len(model_rows)]
            )
            elapsed = (time.perf_counter() - started) * 1000
            response.raise_for_status()
            result = response.json()
            times = result["timings"]
            latencies.append(elapsed)
            queues.append(times["queue_wait_ms"])
            inference.append(times["inference_ms"])
            overhead.append(max(0, elapsed - times["queue_wait_ms"] - times["inference_ms"]))
            if (i + 1) % 40 == 0:
                print(f"HTTP benchmark {i + 1}/{count}", flush=True)
        after_swap = swapouts()
        memory_after = (await client.get("/health/ready")).json()["process_memory"]
        resource = {
            "artifact_version": manifest["artifact_version"],
            "requests": count,
            "concurrent_inference": 1,
            "unique_model_inputs": len(model_rows),
            "http_warm_latency_ms": percentiles(latencies),
            "queue_wait_ms": percentiles(queues),
            "inference_ms": percentiles(inference),
            "http_and_framework_overhead_ms": percentiles(overhead),
            "server_memory_before_serial_run": memory_before,
            "server_memory_after_serial_run": memory_after,
            "system_swapout_pages_delta": swap_delta(before_swap, after_swap),
            "readiness_seconds_including_process_start": ready_seconds,
            "observed_starting_unready": observed_starting,
            "notes": [
                "Warm measurements follow CLI-equivalence and real-input checks.",
                "Overhead is client wall time minus server queue and matching duration.",
                "System swap counters include other applications.",
                "These execution checks do not qualify matching accuracy or production capacity.",
            ],
        }
        (output / "resource_report.json").write_text(json.dumps(resource, indent=2) + "\n")
        print("Checking bounded overload and health responsiveness", flush=True)
        jobs = [
            asyncio.create_task(client.post("/v1/categories/match", json=model_rows[0]))
            for _ in range(10)
        ]
        probes = []
        while not all(job.done() for job in jobs):
            started = time.perf_counter()
            live = await client.get("/health/live")
            live_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            ready = await client.get("/health/ready")
            ready_ms = (time.perf_counter() - started) * 1000
            probes.append(
                {
                    "live_status": live.status_code,
                    "ready_status": ready.status_code,
                    "live_ms": live_ms,
                    "ready_ms": ready_ms,
                    **ready.json(),
                }
            )
            await asyncio.sleep(0.05)
        results = await asyncio.gather(*jobs)
        codes = {
            str(code): sum(r.status_code == code for r in results)
            for code in {r.status_code for r in results}
        }
        assert all(p["active_requests"] <= 1 and p["pending_requests"] <= 8 for p in probes)
        assert codes.get("503", 0) >= 1 and all(p["live_status"] == 200 for p in probes)
        (output / "concurrency_report.json").write_text(
            json.dumps(
                {
                    "artifact_version": manifest["artifact_version"],
                    "burst_requests": 10,
                    "http_statuses": codes,
                    "liveness_ms": percentiles([p["live_ms"] for p in probes]),
                    "readiness_ms": percentiles([p["ready_ms"] for p in probes]),
                    "max_observed_active": max(p["active_requests"] for p in probes),
                    "max_observed_pending": max(p["pending_requests"] for p in probes),
                    "probes": probes,
                },
                indent=2,
            )
            + "\n"
        )
        return resource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/sprint_3"))
    parser.add_argument("--requests", type=int, default=200)
    args = parser.parse_args()
    if not 1 <= args.requests <= 10000:
        parser.error("Requests must be between 1 and 10000")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "resource_report.json").exists():
        raise SystemExit("Use a new output directory to preserve previous evidence")
    manifest = load_manifest(resolve_bundle(Path("artifacts/google")))
    api_files = {str(path): digest(path) for path in Path("src/category_matcher/api").glob("*.py")}
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    environment = {k: v for k, v in os.environ.items() if not k.startswith("CATEGORY_MATCHER_API_")}
    environment.update(
        CATEGORY_MATCHER_API_ARTIFACTS=str(Path("artifacts/google").resolve()),
        CATEGORY_MATCHER_API_QUEUE_SIZE="8",
        CATEGORY_MATCHER_API_TIMEOUT_SECONDS="30",
        CATEGORY_MATCHER_API_MAX_BODY_BYTES="16384",
        CATEGORY_MATCHER_BENCH_FD=str(listener.fileno()),
    )
    code = "\n".join(
        [
            "import logging, os, uvicorn",
            "logging.basicConfig(level=logging.INFO, format='%(message)s')",
            "uvicorn.run('category_matcher.api:create_app', factory=True,",
            "    fd=int(os.environ['CATEGORY_MATCHER_BENCH_FD']), workers=1, access_log=False)",
        ]
    )
    with (args.output_dir / "server.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            pass_fds=(listener.fileno(),),
        )
        listener.close()
        try:
            resource = asyncio.run(
                run_checks(f"http://127.0.0.1:{port}", args.output_dir, manifest, args.requests)
            )
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    if any(digest(Path(path)) != value for path, value in api_files.items()):
        raise RuntimeError("API code changed during measurement; repeat using a new directory")
    (args.output_dir / "environment.json").write_text(
        json.dumps(
            {
                "artifact_version": manifest["artifact_version"],
                "api_source_sha256": api_files,
                "packages": {
                    name: version(name)
                    for name in (
                        "google-category-matcher",
                        "fastapi",
                        "pydantic",
                        "starlette",
                        "uvicorn",
                        "httpx",
                    )
                },
                "server_exit_code": process.returncode,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(resource, indent=2))


if __name__ == "__main__":
    main()
