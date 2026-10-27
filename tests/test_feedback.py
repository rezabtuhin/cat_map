import asyncio
import json

import pytest
from test_api import client_for, factory

from category_matcher.alias_review import apply_alias_reviews, export_feedback, export_pilot_inputs
from category_matcher.api import create_app
from category_matcher.api.settings import APISettings
from category_matcher.artifacts import digest
from category_matcher.runtime import RuntimeStore

MATCH = "/v1/categories/match"
FEEDBACK = "/v1/categories/feedback"


def test_feedback_is_persistent_source_bound_and_never_changes_matching(bundle, tmp_path):
    settings = APISettings(runtime_db=tmp_path / "runtime.sqlite3")
    original = digest(bundle / "taxonomy.sqlite3")
    app = create_app(settings, matcher_factory=factory(bundle))

    async def scenario():
        async with client_for(app) as client:
            result = (
                await client.post(
                    MATCH, json={"taxonomy_key": "google", "category": "portable handsets"}
                )
            ).json()
            assert result["status"] == "provisional" and result["category"] is None
            feedback = await client.post(
                FEEDBACK,
                json={
                    "match_event_id": result["event_id"],
                    "action": "correct",
                    "category_id": "3",
                    "reviewer": "integration fixture",
                    "note": "Test feedback; not a human label",
                },
            )
            assert feedback.status_code == 201, feedback.text
            record = feedback.json()
            assert record["artifact_version"] == result["artifact_version"]
            assert record["match_event_id"] == result["event_id"]
            assert record["review_status"] == "pending"
            assert record["event_id"] == feedback.headers["X-Request-ID"]
            assert (
                await client.post(
                    MATCH, json={"taxonomy_key": "google", "category": "portable handsets"}
                )
            ).json()["category"] is None
        restarted = create_app(settings, matcher_factory=factory(bundle))
        async with client_for(restarted) as client:
            response = await client.post(
                FEEDBACK,
                json={
                    "match_event_id": result["event_id"],
                    "action": "reject",
                    "reviewer": "fixture",
                },
            )
            assert response.status_code == 201
        return result["event_id"]

    event_id = asyncio.run(scenario())
    store = RuntimeStore(settings.runtime_db)
    assert store.event(event_id)["response"]["category"] is None
    assert len(store.feedback_rows()) == 2
    assert digest(bundle / "taxonomy.sqlite3") == original


def test_feedback_validation_and_confirming_a_provisional_choice(bundle, tmp_path):
    settings = APISettings(runtime_db=tmp_path / "runtime.sqlite3")
    app = create_app(settings, matcher_factory=factory(bundle))

    async def scenario():
        async with client_for(app) as client:
            result = (
                await client.post(
                    MATCH,
                    json={
                        "taxonomy_key": "google",
                        "category": "portable handsets",
                        "parent_category_id": "2",
                    },
                )
            ).json()
            base = {"match_event_id": result["event_id"], "reviewer": "fixture"}
            cases = [
                ({**base, "action": "confirm"}, 422),
                ({**base, "action": "correct", "category_id": "missing"}, 422),
                ({**base, "action": "correct", "category_id": "6"}, 422),
                ({**base, "action": "reject", "category_id": "3"}, 422),
                ({**base, "action": "reject", "reviewer": " "}, 422),
                ({**base, "action": "reject", "match_event_id": "unknown"}, 404),
                ({**base, "action": "confirm", "category_id": "3"}, 201),
            ]
            for payload, status in cases:
                response = await client.post(FEEDBACK, json=payload)
                assert response.status_code == status, response.text
            with RuntimeStore(settings.runtime_db).connection() as connection, connection:
                connection.execute("UPDATE match_events SET taxonomy_version='old'")
            response = await client.post(FEEDBACK, json={**base, "action": "reject"})
            assert response.status_code == 409
            assert (await client.get("/health/ready")).status_code == 200
            assert (await client.post(FEEDBACK, content="text")).status_code == 415

    asyncio.run(scenario())


def test_failed_event_persistence_does_not_return_a_mapping(bundle, tmp_path):
    app = create_app(
        APISettings(runtime_db=tmp_path / "runtime.sqlite3"), matcher_factory=factory(bundle)
    )

    def fail(*args):
        raise OSError("private database path")

    async def scenario():
        async with client_for(app) as client:
            app.state.service.runtime.record_match = fail
            response = await client.post(
                MATCH, json={"taxonomy_key": "google", "category": "Mobile Phones"}
            )
            assert response.status_code == 503 and "private database path" not in response.text
            assert "category" not in response.json()
            assert (await client.get("/health/ready")).status_code == 503

    asyncio.run(scenario())


def test_runtime_path_cannot_be_inside_bundle_storage(bundle):
    settings = APISettings(artifacts=bundle.parent, runtime_db=bundle / "runtime.sqlite3")
    app = create_app(settings, matcher_factory=factory(bundle))

    async def scenario():
        async with client_for(app, ready=False) as client:
            await app.state.service.initialized.wait()
            assert (await client.get("/health/ready")).status_code == 503

    asyncio.run(scenario())
    assert not settings.runtime_db.exists()


@pytest.fixture
def pilot(bundle, tmp_path):
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    store.initialize()
    app = create_app(APISettings(runtime_db=store.path), matcher_factory=factory(bundle))

    async def scenario():
        async with client_for(app) as client:
            result = (
                await client.post(
                    MATCH, json={"taxonomy_key": "google", "category": "portable handsets"}
                )
            ).json()
            response = await client.post(
                FEEDBACK,
                json={
                    "match_event_id": result["event_id"],
                    "action": "correct",
                    "category_id": "3",
                    "reviewer": "synthetic fixture",
                },
            )
            assert response.status_code == 201

    asyncio.run(scenario())
    aliases = tmp_path / "aliases.jsonl"
    aliases.write_text("")
    packet = tmp_path / "reviews.jsonl"
    export_feedback(store, packet)
    return store, aliases, packet, bundle.parent / "source.txt"


def approval(packet):
    row = json.loads(packet.read_text())
    row["decision"] = {
        "state": "approved",
        "reviewer_kind": "human",
        "reviewer": "test reviewer",
        "reviewed_at": "2026-10-27T12:00:00+06:00",
        "reason": "Fixture only",
    }
    packet.write_text(json.dumps(row) + "\n")
    return row


def test_alias_promotion_requires_source_bound_human_review_and_new_file(pilot, tmp_path):
    store, aliases, packet, taxonomy = pilot
    output = tmp_path / "approved.jsonl"
    with pytest.raises(ValueError, match="No human"):
        apply_alias_reviews(store, packet, aliases, taxonomy, output)
    assert not output.exists()
    approval(packet)
    result = apply_alias_reviews(store, packet, aliases, taxonomy, output)
    assert result["approved"] == 1 and result["activation_required"]
    promoted = json.loads(output.read_text())
    assert promoted["alias"] == "portable handsets" and promoted["category_id"] == "3"
    assert aliases.read_text() == "" and store.feedback_rows()[0]["review_status"] == "pending"
    with pytest.raises(ValueError, match="new file"):
        apply_alias_reviews(store, packet, aliases, taxonomy, aliases)
    with pytest.raises(ValueError, match="already approved"):
        apply_alias_reviews(store, packet, output, taxonomy, tmp_path / "repeat.jsonl")


@pytest.mark.parametrize("mutation", ["input", "target", "reviewer", "timestamp", "collision"])
def test_alias_review_rejects_tampering_and_unsafe_decisions(pilot, tmp_path, mutation):
    store, aliases, packet, taxonomy = pilot
    row = approval(packet)
    if mutation == "input":
        row["event"]["request"]["category"] = "changed input"
    elif mutation == "target":
        row["proposal"]["category_id"] = "6"
    elif mutation == "reviewer":
        row["decision"]["reviewer_kind"] = "ai"
    elif mutation == "timestamp":
        row["decision"]["reviewed_at"] = "2026-10-27"
    else:
        aliases.write_text(
            json.dumps(
                {
                    "taxonomy_key": "google",
                    "alias": "portable handsets",
                    "category_id": "4",
                    "status": "approved",
                    "reviewer": "fixture",
                    "reviewed_at": "now",
                }
            )
            + "\n"
        )
    packet.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        apply_alias_reviews(store, packet, aliases, taxonomy, tmp_path / "bad.jsonl")
    assert not (tmp_path / "bad.jsonl").exists()


def test_exported_pilot_inputs_are_unlabeled_development_data(pilot, tmp_path):
    store, _, _, _ = pilot
    output = tmp_path / "pilot.jsonl"
    report = export_pilot_inputs(store, output)
    assert report["independent_holdout_created"] is False
    row = json.loads(output.read_text())
    assert row["dataset_role"] == "pilot_development_only"
    assert row["expected_status"] is None and row["review_status"] == "pending_review"


@pytest.mark.parametrize(
    "category,extra",
    [
        ("Collections", {}),
        ("Eco Friendly", {}),
        ("handsets|Electronics", {}),
        ("portable handsets", {"context": "phones"}),
        ("portable handsets", {"parent_category_id": "2"}),
    ],
)
def test_contextual_or_nonproduct_feedback_cannot_become_a_global_alias(
    bundle, tmp_path, category, extra
):
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    store.initialize()
    app = create_app(APISettings(runtime_db=store.path), matcher_factory=factory(bundle))

    async def scenario():
        async with client_for(app) as client:
            result = (
                await client.post(
                    MATCH, json={"taxonomy_key": "google", "category": category, **extra}
                )
            ).json()
            response = await client.post(
                FEEDBACK,
                json={
                    "match_event_id": result["event_id"],
                    "action": "correct",
                    "category_id": "3",
                    "reviewer": "fixture",
                },
            )
            assert response.status_code == 201

    asyncio.run(scenario())
    packet = tmp_path / "reviews.jsonl"
    export_feedback(store, packet)
    assert json.loads(packet.read_text())["proposal"]["alias"] is None
    approval(packet)
    aliases = tmp_path / "aliases.jsonl"
    aliases.write_text("")
    with pytest.raises(ValueError, match="not eligible"):
        apply_alias_reviews(
            store, packet, aliases, bundle.parent / "source.txt", tmp_path / "bad.jsonl"
        )
