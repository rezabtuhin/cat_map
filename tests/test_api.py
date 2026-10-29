import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from uuid import UUID

import httpx
import numpy as np
import pytest
from pydantic import ValidationError

from category_matcher.api import create_app
from category_matcher.api.schemas import MatchResponse
from category_matcher.api.settings import APISettings
from category_matcher.matcher import Matcher

ROUTE = "/v1/categories/match"


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def factory(bundle, *, gate=None, threads=None, corrupt=False, crash=False):
    def initialize():
        matcher = Matcher(bundle)
        matcher.encode_queries = lambda texts: np.array([[1.0, 0.0, 0.0] for _ in texts])
        original = matcher.match
        if threads is not None:
            threads.append(("initialize", threading.get_ident()))
            close = matcher.close

            def tracked_close():
                threads.append(("close", threading.get_ident()))
                close()

            matcher.close = tracked_close

        def match(text="", **kwargs):
            if threads is not None:
                threads.append(("match", threading.get_ident()))
            if gate is not None and text == "slow":
                gate[0].set()
                assert gate[1].wait(3), "Test did not release its inference gate"
            if crash:
                raise RuntimeError("private internal failure detail")
            result = original(text, **kwargs)
            if corrupt:
                result.status = "provisional"
            return result

        matcher.match = match
        return matcher

    return initialize


@asynccontextmanager
async def client_for(app, ready=True):
    async with app.router.lifespan_context(app):
        if ready:
            await app.state.service.wait_ready(3)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client


async def wait_until(predicate):
    deadline = time.perf_counter() + 2
    while not predicate():
        assert time.perf_counter() < deadline, "Timed out waiting for test state"
        await asyncio.sleep(0.001)


def test_api_agrees_with_engine_and_owns_sqlite_on_one_thread(bundle, caplog):
    threads = []
    caplog.set_level("INFO", logger="category_matcher.api")
    app = create_app(APISettings(), matcher_factory=factory(bundle, threads=threads))
    baseline = factory(bundle)()
    cases = [
        ("Mobile Phones", "en"),
        ("cellphones", "en"),
        ("portable handsets", "en"),
        ("Collections", "en"),
        ("Erin Recommends", "en"),
        ("phone", "bn"),
    ]
    expected = [baseline.match(text, language=language) for text, language in cases]
    baseline.close()

    async def scenario():
        async with client_for(app) as client:
            for (text, language), reference in zip(cases, expected, strict=True):
                response = await client.post(
                    ROUTE, json={"taxonomy_key": "google", "category": text, "language": language}
                )
                assert response.status_code == 200, response.text
                result = response.json()
                assert result["status"] == reference.status
                assert result["category"] == reference.category
                assert result["reason_code"] == reference.reason_code
                assert result["artifact_version"] == "unit-test-bundle"
                UUID(result["event_id"])
                assert response.headers["X-Request-ID"] == result["event_id"]
                assert all(value >= 0 for value in result["timings"].values())
                if result["status"] != "matched":
                    assert result["category"] is None
            by_id = await client.post(ROUTE, json={"taxonomy_key": "google", "category_id": "3"})
            assert by_id.json()["category"]["id"] == "3"
            canonical = await client.get("/v1/taxonomies/google/categories/3")
            assert canonical.status_code == 200 and canonical.json()["parent_category_id"] == "2"
            assert (await client.get("/v1/taxonomies/meta/categories/3")).status_code == 404
            assert (await client.get("/v1/taxonomies/google/categories/missing")).status_code == 404
            taxonomy = (await client.get("/v1/taxonomies")).json()[0]
            assert taxonomy["category_count"] == 7
            assert (await client.get("/health/ready")).json()[
                "semantic_auto_acceptance_enabled"
            ] is False

    asyncio.run(scenario())
    assert {label for label, _ in threads} == {"initialize", "match", "close"}
    assert len({identifier for _, identifier in threads}) == 1
    assert threads[0][1] != threading.get_ident()
    records = [json.loads(r.message) for r in caplog.records if r.name == "category_matcher.api"]
    assert any(r["event"] == "match_completed" and r["stage"] == "matching" for r in records)
    assert all("category" not in r for r in records)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"taxonomy_key": "meta", "category": "phones"},
        {"taxonomy_key": "google", "category": "  "},
        {"taxonomy_key": "google", "category_id": 3},
        {"taxonomy_key": "google", "category": "phones", "force_match": True},
        {"taxonomy_key": "google", "category": "phones", "limit": True},
        {"taxonomy_key": "google", "category": "x" * 301},
        {"taxonomy_key": "google", "category": "phones", "context": "x" * 1001},
        {"taxonomy_key": "google", "category": "phones", "taxonomy_version": "old"},
        {"taxonomy_key": "google", "category": "Mobile Phones", "parent_category_id": "5"},
        {"taxonomy_key": "google", "category": "Mobile Phones", "category_id": "4"},
        {"taxonomy_key": "google", "category": "phones", "parent_category_id": "missing"},
    ],
)
def test_invalid_requests_fail_without_poisoning_readiness(bundle, payload):
    app = create_app(APISettings(), matcher_factory=factory(bundle))

    async def scenario():
        async with client_for(app) as client:
            response = await client.post(ROUTE, json=payload)
            assert response.status_code == 422, response.text
            assert response.json()["event_id"] == response.headers["X-Request-ID"]
            assert (await client.get("/health/ready")).status_code == 200

    asyncio.run(scenario())


def test_request_boundary_enforces_chunked_body_limits_media_type_and_json(bundle):
    app = create_app(APISettings(max_body_bytes=1024), matcher_factory=factory(bundle))

    async def chunks():
        yield b"x" * 600
        yield b"x" * 600

    async def scenario():
        async with client_for(app) as client:
            response = await client.post(
                ROUTE, content=chunks(), headers={"Content-Type": "application/json"}
            )
            assert response.status_code == 413
            assert (await client.post(ROUTE, content="text")).status_code == 415
            assert (
                await client.post(ROUTE, content="{", headers={"Content-Type": "application/json"})
            ).status_code == 422
            trimmed = await client.post(
                ROUTE, json={"taxonomy_key": "google", "category": "  Mobile Phones  "}
            )
            assert trimmed.json()["category"]["id"] == "3"

    asyncio.run(scenario())


def test_queue_is_bounded_and_health_and_lookup_remain_responsive(bundle):
    entered, release = threading.Event(), threading.Event()
    app = create_app(
        APISettings(queue_size=8, timeout_seconds=2),
        matcher_factory=factory(bundle, gate=(entered, release)),
    )

    async def scenario():
        async with client_for(app) as client:
            first = asyncio.create_task(
                client.post(ROUTE, json={"taxonomy_key": "google", "category": "slow"})
            )
            await wait_until(entered.is_set)
            pending = [
                asyncio.create_task(
                    client.post(ROUTE, json={"taxonomy_key": "google", "category": "Mobile Phones"})
                )
                for _ in range(8)
            ]
            await wait_until(lambda: len(app.state.service.pending) == 8)
            rejected = await client.post(
                ROUTE, json={"taxonomy_key": "google", "category": "Mobile Phones"}
            )
            assert rejected.status_code == 503 and rejected.json()["error"]["code"] == "queue_full"
            assert rejected.headers["Retry-After"] == "1"
            started = time.perf_counter()
            assert (await client.get("/health/live")).status_code == 200
            assert (await client.get("/v1/taxonomies/google/categories/3")).status_code == 200
            assert time.perf_counter() - started < 0.5
            assert (await client.get("/health/ready")).json()["pending_requests"] == 8
            release.set()
            results = await asyncio.gather(first, *pending)
            assert all(response.status_code == 200 for response in results)

    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_timeout_removes_queued_jobs_but_keeps_native_inference_slot(bundle):
    entered, release = threading.Event(), threading.Event()
    threads = []
    app = create_app(
        APISettings(queue_size=1, timeout_seconds=0.08),
        matcher_factory=factory(bundle, gate=(entered, release), threads=threads),
    )

    async def scenario():
        async with client_for(app) as client:
            first = asyncio.create_task(
                client.post(ROUTE, json={"taxonomy_key": "google", "category": "slow"})
            )
            await wait_until(entered.is_set)
            queued = asyncio.create_task(
                client.post(ROUTE, json={"taxonomy_key": "google", "category": "Mobile Phones"})
            )
            results = await asyncio.gather(first, queued)
            assert [r.status_code for r in results] == [504, 504]
            assert app.state.service.active is not None and len(app.state.service.pending) == 0
            assert len([label for label, _ in threads if label == "match"]) == 1
            release.set()
            await wait_until(lambda: app.state.service.active is None)
            response = await client.post(
                ROUTE, json={"taxonomy_key": "google", "category": "Mobile Phones"}
            )
            assert response.status_code == 200

    try:
        asyncio.run(scenario())
    finally:
        release.set()


@pytest.mark.parametrize("failure", ["crash", "assignment_contract"])
def test_inference_failure_never_returns_an_untrusted_assignment(bundle, failure):
    app = create_app(
        APISettings(),
        matcher_factory=factory(
            bundle, crash=failure == "crash", corrupt=failure == "assignment_contract"
        ),
    )

    async def scenario():
        async with client_for(app) as client:
            response = await client.post(
                ROUTE, json={"taxonomy_key": "google", "category": "Mobile Phones"}
            )
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "inference_failed"
            assert "private internal" not in response.text and "category" not in response.json()
            assert (await client.get("/health/ready")).status_code == 503
            assert (await client.get("/health/live")).status_code == 200

    asyncio.run(scenario())


def test_startup_failure_is_unready_with_live_process_and_no_internal_error_details():
    def fail():
        raise FileNotFoundError("private model path")

    app = create_app(APISettings(), matcher_factory=fail)

    async def scenario():
        async with client_for(app, ready=False) as client:
            await app.state.service.initialized.wait()
            assert (await client.get("/health/live")).status_code == 200
            ready = await client.get("/health/ready")
            assert ready.status_code == 503 and ready.json()["state"] == "failed"
            response = await client.post(
                ROUTE, json={"taxonomy_key": "google", "category": "Mobile Phones"}
            )
            assert response.status_code == 503 and "private model path" not in response.text

    asyncio.run(scenario())


def test_response_schema_rejects_assigned_categories_for_uncertain_outcomes():
    with pytest.raises(ValidationError, match="accepted matched"):
        MatchResponse(
            status="provisional",
            taxonomy_key="google",
            taxonomy_version="v1",
            artifact_version="a1",
            category={"id": "1", "label": "P", "path": "P"},
            alternatives=[],
            method="hybrid",
            reason_code="test",
            clarification=None,
            event_id="event",
            timings={"queue_wait_ms": 0.0, "inference_ms": 0.0, "total_ms": 0.0},
        )


def test_no_match_response_keeps_category_null_and_hides_candidates(bundle):
    def initialize():
        matcher = factory(bundle)()
        original = matcher.match

        def no_match(text, **kwargs):
            result = original(text, **kwargs)
            result.status = "no_match"
            result.category = None
            result.reason_code = "no_candidates"
            return result

        matcher.match = no_match
        return matcher

    app = create_app(APISettings(), matcher_factory=initialize)

    async def scenario():
        async with client_for(app) as client:
            response = await client.post(
                ROUTE, json={"taxonomy_key": "google", "category": "Mobile Phones"}
            )
            assert response.status_code == 200
            assert response.json()["status"] == "no_match"
            assert response.json()["category"] is None
            assert response.json()["alternatives"] == []
            assert response.json()["reason_code"] == "no_candidates"

    asyncio.run(scenario())


def test_noncanonical_backend_assignment_fails_closed(bundle):
    def initialize():
        matcher = factory(bundle)()
        original = matcher.match

        def wrong_path(text, **kwargs):
            result = original(text, **kwargs)
            result.category["path"] = "Invented canonical path"
            return result

        matcher.match = wrong_path
        return matcher

    app = create_app(APISettings(), matcher_factory=initialize)

    async def scenario():
        async with client_for(app) as client:
            response = await client.post(
                ROUTE, json={"taxonomy_key": "google", "category": "Mobile Phones"}
            )
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "inference_failed"
            assert "Invented canonical path" not in response.text
            assert (await client.get("/health/ready")).status_code == 503

    asyncio.run(scenario())


def test_startup_is_nonblocking_and_readiness_waits_for_initialization(bundle):
    release = threading.Event()

    def initialize():
        assert release.wait(2)
        return factory(bundle)()

    app = create_app(APISettings(), matcher_factory=initialize)

    async def scenario():
        async with client_for(app, ready=False) as client:
            assert (await client.get("/health/live")).status_code == 200
            assert (await client.get("/health/ready")).status_code == 503
            release.set()
            await app.state.service.wait_ready(3)
            assert (await client.get("/health/ready")).status_code == 200

    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_client_cancellation_frees_pending_capacity_and_shutdown_drains_owner_thread(bundle):
    entered, release = threading.Event(), threading.Event()
    app = create_app(
        APISettings(timeout_seconds=2), matcher_factory=factory(bundle, gate=(entered, release))
    )

    async def scenario():
        async with client_for(app) as client:
            active = asyncio.create_task(
                client.post(ROUTE, json={"taxonomy_key": "google", "category": "slow"})
            )
            await wait_until(entered.is_set)
            queued = asyncio.create_task(
                client.post(ROUTE, json={"taxonomy_key": "google", "category": "Mobile Phones"})
            )
            await wait_until(lambda: len(app.state.service.pending) == 1)
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
            assert len(app.state.service.pending) == 0
            active.cancel()
            with pytest.raises(asyncio.CancelledError):
                await active
            assert app.state.service.active is not None
            closing = asyncio.create_task(app.state.service.close())
            await wait_until(lambda: app.state.service.state == "stopping")
            assert not closing.done()
            release.set()
            await closing
            await app.state.service.close()  # lifespan cleanup is idempotent
            assert app.state.service.state == "stopped"

    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_draft_policy_cannot_enable_api_assignments(bundle, tmp_path):
    from category_matcher.decision import write_policy

    (bundle.parent / "active.json").write_text(json.dumps({"artifact_version": bundle.name}))
    policy = tmp_path / "draft.json"
    write_policy(
        policy,
        {
            "format_version": 1,
            "taxonomy_key": "google",
            "artifact_version": "unit-test-bundle",
            "thresholds": {
                "min_semantic": 0.5,
                "no_match_semantic": 0.1,
                "min_margin": 0,
                "fuzzy_support": 70,
            },
            "calibration": {"unreviewed_examples": 200},
            "qualification": {"passed": False},
        },
    )
    app = create_app(APISettings(artifacts=bundle.parent, policy=policy))

    async def scenario():
        async with client_for(app, ready=False) as client:
            await app.state.service.initialized.wait()
            assert (await client.get("/health/ready")).status_code == 503
            assert not app.state.service.policy_enabled

    asyncio.run(scenario())


def test_body_read_timeout_does_not_admit_inference(bundle):
    app = create_app(APISettings(timeout_seconds=0.02), matcher_factory=factory(bundle))

    async def chunks():
        await asyncio.sleep(0.08)
        yield b"{}"

    async def scenario():
        async with client_for(app) as client:
            result = await client.post(
                ROUTE, content=chunks(), headers={"Content-Type": "application/json"}
            )
            assert result.status_code == 504
            assert app.state.service.active is None

    asyncio.run(scenario())


def test_settings_reject_unknown_environment_and_invalid_limits(monkeypatch):
    monkeypatch.setenv("CATEGORY_MATCHER_API_QUEUE_SIZE", "4")
    assert APISettings.from_env().queue_size == 4
    monkeypatch.setenv("CATEGORY_MATCHER_API_UNKNOWN", "value")
    with pytest.raises(ValueError, match="Unknown API setting"):
        APISettings.from_env()
    for values in [{"queue_size": 0}, {"queue_size": True}, {"timeout_seconds": float("nan")}]:
        with pytest.raises(ValidationError):
            APISettings(**values)
