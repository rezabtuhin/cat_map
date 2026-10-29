import json
import shutil

import numpy as np
import pytest

from category_matcher.api.service import initialize_matcher
from category_matcher.api.settings import APISettings
from category_matcher.bundles import (
    FileLease,
    activate,
    copy_bundle,
    load_active,
    load_release,
    rollback,
    stage_build,
)


@pytest.fixture
def managed_bundle(bundle):
    metadata = bundle / "manifest.json"
    manifest = json.loads(metadata.read_text())
    snapshot = bundle.parent / "models" / "embedding"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"test model weights")
    (snapshot / "config.json").write_text("{}")
    manifest["model"].update(
        {
            "name": "fixture",
            "revision": "a" * 40,
            "snapshot": str(snapshot),
            "query_instruction": "",
        }
    )
    metadata.write_text(json.dumps(manifest))
    destination = bundle.with_name(manifest["artifact_version"])
    bundle.rename(destination)
    return destination


def test_release_pins_models_and_manifest_and_rejects_tampering(managed_bundle):
    bundle, root = managed_bundle, managed_bundle.parent
    release = activate(root, bundle.name)
    assert load_active(root, require_release=True)[0] == bundle
    assert activate(root, bundle.name)["changed"] is False
    snapshot = json.loads((bundle / "manifest.json").read_text())["model"]["snapshot"]
    from pathlib import Path

    weights = Path(snapshot) / "model.safetensors"
    original = weights.read_bytes()
    weights.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="Model file checksum"):
        load_active(root)
    weights.write_bytes(original)
    manifest = bundle / "manifest.json"
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ValueError, match="manifest checksum"):
        load_release(root, release["release_id"])


def test_activation_is_blocked_while_service_owns_release_and_rollback_works(
    managed_bundle, monkeypatch
):
    root = managed_bundle.parent
    first = activate(root, managed_bundle.name)
    monkeypatch.setattr(
        "category_matcher.matcher.Matcher.encode_queries",
        lambda self, texts: np.array([[1.0, 0.0, 0.0]]),
    )
    resident = initialize_matcher(APISettings(artifacts=root))
    try:
        with pytest.raises(ValueError, match="stop the API"):
            activate(root, managed_bundle.name)
        assert resident.match("Mobile Phones").category["id"] == "3"
    finally:
        resident.close()
    second_bundle = root / "next-bundle"
    shutil.copytree(managed_bundle, second_bundle)
    document = json.loads((second_bundle / "manifest.json").read_text())
    document["artifact_version"] = second_bundle.name
    (second_bundle / "manifest.json").write_text(json.dumps(document))
    second = activate(root, second_bundle.name)
    assert second["previous_release_id"] == first["release_id"]
    assert load_active(root)[0] == second_bundle
    assert rollback(root)["release_id"] == first["release_id"]
    assert load_active(root)[0] == managed_bundle
    assert rollback(root)["release_id"] == second["release_id"]


def test_invalid_activation_leaves_pointer_unchanged(managed_bundle):
    root = managed_bundle.parent
    activate(root, managed_bundle.name)
    before = (root / "active.json").read_bytes()
    (managed_bundle / "extra-file").write_text("untracked payload")
    with pytest.raises(ValueError, match="inventory"):
        activate(root, managed_bundle.name)
    assert (root / "active.json").read_bytes() == before
    with pytest.raises(ValueError, match="identifier"):
        activate(root, "../outside")


def test_staging_build_does_not_activate_and_cleans_failed_build(
    managed_bundle, tmp_path, monkeypatch
):
    root = tmp_path / "staged"
    root.mkdir()
    pointer = root / "active.json"
    pointer.write_text('{"artifact_version":"current"}')
    before = pointer.read_bytes()

    def build_fixture(source, aliases, model, temporary, profile):
        staged = temporary / managed_bundle.name
        shutil.copytree(managed_bundle, staged)
        (temporary / "active.json").write_text("{}")
        return staged, json.loads((staged / "manifest.json").read_text())

    monkeypatch.setattr("category_matcher.bundles.build", build_fixture)
    destination, _, _ = stage_build(None, None, None, root, None)
    assert destination.name == managed_bundle.name and pointer.read_bytes() == before
    stage_build(None, None, None, root, None)
    assert pointer.read_bytes() == before

    def fail(*args):
        raise ValueError("failed build")

    monkeypatch.setattr("category_matcher.bundles.build", fail)
    with pytest.raises(ValueError, match="failed build"):
        stage_build(None, None, None, root, None)
    assert not list(root.glob(".staging-*"))


def test_imported_bundle_is_verified_without_activation(managed_bundle, tmp_path):
    root = tmp_path / "imported"
    copy_bundle(managed_bundle, root)
    assert not (root / "active.json").exists()
    with FileLease(root / ".service.lock", shared=True), pytest.raises(ValueError):
        activate(root, managed_bundle.name)


def test_release_policy_must_be_qualified_and_snapshot_is_immutable(managed_bundle, tmp_path):
    from test_decision import document

    from category_matcher.decision import QUALIFICATION_CHECKS, write_policy

    root = managed_bundle.parent
    first = activate(root, managed_bundle.name)
    policy = document()
    policy["artifact_version"] = managed_bundle.name
    path = tmp_path / "policy.json"
    write_policy(path, policy)
    with pytest.raises(ValueError, match="not independently validated"):
        activate(root, managed_bundle.name, policy=path)
    assert load_active(root)[2] is None
    policy["calibration"]["unreviewed_examples"] = 0
    policy["qualification"] = {
        "passed": True,
        "checks": dict.fromkeys(QUALIFICATION_CHECKS, True),
        "example_count": 200,
        "unreviewed_examples": 0,
        "independent_holdout": True,
        "dataset_sha256": "synthetic-independent-fixture",
    }
    write_policy(path, policy)
    second = activate(root, managed_bundle.name, policy=path)
    bound = load_active(root)[2]
    assert bound != path and bound.read_bytes() == path.read_bytes()
    path.write_text("changed external source")
    assert load_active(root)[2] == bound
    bound.write_text("corrupted snapshot")
    with pytest.raises(ValueError, match="Policy snapshot checksum"):
        load_active(root)
    assert rollback(root)["release_id"] == first["release_id"]
    assert second["release_id"] != first["release_id"]


def test_failed_pointer_commit_preserves_previous_active_release(managed_bundle, monkeypatch):
    root = managed_bundle.parent
    first = activate(root, managed_bundle.name)
    original = (root / "active.json").read_bytes()
    next_bundle = root / "next-generation"
    shutil.copytree(managed_bundle, next_bundle)
    value = json.loads((next_bundle / "manifest.json").read_text())
    value["artifact_version"] = next_bundle.name
    (next_bundle / "manifest.json").write_text(json.dumps(value))

    def fail(*args):
        raise OSError("simulated disk failure")

    monkeypatch.setattr("category_matcher.bundles.atomic_json", fail)
    with pytest.raises(OSError, match="simulated"):
        activate(root, next_bundle.name)
    assert (root / "active.json").read_bytes() == original
    assert load_active(root)[0] == managed_bundle
    assert json.loads(original)["release_id"] == first["release_id"]
