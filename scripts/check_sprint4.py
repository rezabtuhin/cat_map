"""Real-model pilot/recovery checks in isolated state; write only a short summary."""

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import httpx

from category_matcher.alias_review import apply_alias_reviews, export_feedback, export_pilot_inputs
from category_matcher.artifacts import digest, resolve_bundle
from category_matcher.bundles import activate, copy_bundle, rollback, stage_build
from category_matcher.datasets import read_jsonl
from category_matcher.runtime import RuntimeStore, restore


@contextmanager
def serve(root, runtime, directory, name):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    environment = {k: v for k, v in os.environ.items() if not k.startswith("CATEGORY_MATCHER_API_")}
    environment.update(
        CATEGORY_MATCHER_API_ARTIFACTS=str(root),
        CATEGORY_MATCHER_API_RUNTIME_DB=str(runtime),
        CATEGORY_MATCHER_CHECK_FD=str(listener.fileno()),
    )
    code = (
        "import os, uvicorn; "
        "uvicorn.run('category_matcher.api:create_app', factory=True, "
        "fd=int(os.environ['CATEGORY_MATCHER_CHECK_FD']), workers=1, access_log=False)"
    )
    with (directory / f"{name}.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            pass_fds=(listener.fileno(),),
        )
        listener.close()
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=40) as client:
                started = time.monotonic()
                while True:
                    try:
                        ready = client.get("/health/ready")
                        if ready.status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if process.poll() is not None or time.monotonic() - started > 90:
                        raise RuntimeError((directory / f"{name}.log").read_text()[-2000:])
                    time.sleep(0.05)
                assert ready.json()["semantic_auto_acceptance_enabled"] is False
                yield client
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def match(client, category, **extra):
    response = client.post(
        "/v1/categories/match", json={"taxonomy_key": "google", "category": category, **extra}
    )
    response.raise_for_status()
    result = response.json()
    assert (result["status"] == "matched") == (result["category"] is not None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/sprint_4/checks.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing evidence; use a new --output path")
    original = resolve_bundle(Path("artifacts/google"))
    active_before = Path("artifacts/google/active.json").read_bytes()
    with tempfile.TemporaryDirectory(prefix="category-sprint4-") as temporary:
        directory = Path(temporary)
        root = directory / "artifacts"
        current = copy_bundle(original, root)
        first = activate(root, current.name)
        print("Rebuilding alternate retrieval-only profile for restart/rollback check", flush=True)
        alternate, _, _ = stage_build(
            Path("google_category_list.txt"),
            Path("data/google/aliases.jsonl"),
            Path("models/embedding_model.json"),
            root,
            None,
        )
        runtime = directory / "runtime.sqlite3"
        dataset = Path("data/google/evaluation/development.jsonl")
        reference = json.loads(
            Path("reports/sprint_2/revision_3/retrieval_comparison.json").read_text()
        )
        assert reference["dataset_sha256"] == digest(dataset)
        assert reference["artifact_version"] == original.name
        expected = {r["example_id"]: r for r in reference["examples"]["hybrid_rerank"]}
        agreements = 0
        with serve(root, runtime, directory, "primary") as client:
            print("Checking frozen CLI agreement and persistent feedback", flush=True)
            for i, row in enumerate(read_jsonl(dataset), 1):
                actual = match(
                    client,
                    row["query"],
                    category_id=row.get("category_id"),
                    context=row.get("context"),
                    parent_category_id=row.get("parent_category_id"),
                    language=row.get("language", "en"),
                )
                prior = expected[row["example_id"]]
                assert actual["status"] == prior["actual_status"]
                assert (actual["category"]["id"] if actual["category"] else None) == prior[
                    "accepted_id"
                ]
                assert (actual["method"], actual["reason_code"]) == (
                    prior["actual_method"],
                    prior["reason_code"],
                )
                alternatives = (
                    prior["candidate_ids"][:3]
                    if actual["status"] in {"provisional", "needs_clarification"}
                    else []
                )
                assert [c["id"] for c in actual["alternatives"]] == alternatives
                agreements += 1
                if i % 50 == 0:
                    print(f"CLI agreement {i}/200", flush=True)
            for filename in ("user_examples.jsonl", "user_examples_2.jsonl"):
                for row in read_jsonl(Path("data/google/evaluation") / filename):
                    match(client, row["query"])
            result = match(client, "Hoodies & Sweatshirts|Clothing")
            event_id = result["event_id"]
            assert result["category"] is None
            response = client.post(
                "/v1/categories/feedback",
                json={
                    "match_event_id": event_id,
                    "action": "correct",
                    "category_id": "212",
                    "reviewer": "synthetic integration check",
                    "note": "Not human review or a label approval",
                },
            )
            assert response.status_code == 201 and response.json()["review_status"] == "pending"
            try:
                activate(root, alternate.name)
            except ValueError as error:
                assert "stop the API" in str(error)
            else:
                raise AssertionError("Activation was allowed with resident models")
            store = RuntimeStore(runtime)
            backup = directory / "backup.sqlite3"
            store.backup(backup)
            packet = directory / "feedback.jsonl"
            export_feedback(store, packet)
            assert all(r["decision"]["state"] == "pending" for r in read_jsonl(packet))
            try:
                apply_alias_reviews(
                    store,
                    packet,
                    Path("data/google/aliases.jsonl"),
                    Path("google_category_list.txt"),
                    directory / "aliases.jsonl",
                )
            except ValueError as error:
                assert "No human alias approvals" in str(error)
            else:
                raise AssertionError("Unreviewed feedback promoted an alias")
            export = export_pilot_inputs(store, directory / "pilot.jsonl")
            assert export["independent_holdout_created"] is False
        restored = directory / "restored.sqlite3"
        restore(backup, restored)
        assert RuntimeStore(restored).events() == RuntimeStore(runtime).events()
        assert RuntimeStore(restored).feedback_rows() == RuntimeStore(runtime).feedback_rows()
        activate(root, alternate.name)
        with serve(root, runtime, directory, "alternate") as client:
            alternate_result = match(client, "Smart Phone")
            assert alternate_result["artifact_version"] == alternate.name
            assert alternate_result["method"] == "hybrid"
        assert rollback(root)["release_id"] == first["release_id"]
        with serve(root, restored, directory, "rollback") as client:
            recovered = match(client, "Hoodies & Sweatshirts|Clothing")
            assert (
                recovered["artifact_version"] == original.name
                and recovered["method"] == "hybrid+rerank"
            )
            assert recovered["category"] is None
            response = client.post(
                "/v1/categories/feedback",
                json={
                    "match_event_id": event_id,
                    "action": "reject",
                    "reviewer": "synthetic restore check",
                },
            )
            assert response.status_code == 201
        assert Path("artifacts/google/active.json").read_bytes() == active_before
        report = {
            "artifact_version": original.name,
            "cli_agreement": f"{agreements}/200",
            "user_examples_checked": 10,
            "feedback_persisted_pending": True,
            "activation_blocked_while_serving": True,
            "live_backup_and_restore": "passed",
            "alternate_profile_restart_and_rollback": "passed",
            "restored_event_feedback": "passed",
            "unreviewed_alias_promotion": "rejected",
            "pilot_export": "unlabeled development inputs only",
            "test_holdout_scored": False,
            "human_pilot": "pending",
            "qualification": "pending",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
