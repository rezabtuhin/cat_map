"""Staging and immutable release descriptors around the existing artifact builder."""

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import AbstractContextManager
from pathlib import Path

from category_matcher.artifacts import build, digest, load_manifest, resolve_bundle
from category_matcher.decision import DecisionPolicy
from category_matcher.storage import connect


class FileLease(AbstractContextManager):
    def __init__(self, path: Path, *, shared=False):
        self.path, self.shared, self.stream = path, shared, None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+")
        try:
            fcntl.flock(
                self.stream, (fcntl.LOCK_SH if self.shared else fcntl.LOCK_EX) | fcntl.LOCK_NB
            )
        except BlockingIOError as error:
            self.stream.close()
            self.stream = None
            raise ValueError(
                "Bundle is in use; stop the API and retry activation/rollback"
            ) from error
        return self

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def __exit__(self, *_):
        self.close()


def atomic_json(path, document):
    atomic_bytes(path, (json.dumps(document, sort_keys=True) + "\n").encode())


def atomic_bytes(path, content):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def member(root, name):
    if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Invalid bundle identifier")
    path = (root / name).resolve()
    if path.parent != root.resolve():
        raise ValueError("Bundle escapes artifact root")
    return path


def model_inventory(manifest):
    inventory = {}
    for role, metadata in (
        ("embedding", manifest["model"]),
        ("ranking", manifest.get("ranking_model")),
    ):
        if metadata is None:
            continue
        revision = metadata.get("revision")
        if not isinstance(revision, str) or len(revision) != 40:
            raise ValueError("Model must use an immutable revision")
        snapshot = Path(metadata["snapshot"])
        if not snapshot.is_dir():
            raise ValueError(f"Missing {role} model snapshot")
        files = {
            str(p.relative_to(snapshot)): digest(p)
            for p in sorted(snapshot.rglob("*"))
            if p.is_file()
        }
        if not files or not any(p.endswith((".safetensors", ".bin")) for p in files):
            raise ValueError(f"Missing {role} model weights")
        inventory[role] = {"name": metadata["name"], "revision": revision, "files": files}
    return inventory


def validate_policy(path, manifest):
    if path is None:
        return None
    policy = DecisionPolicy.load(path, artifact_version=manifest["artifact_version"])
    ranking = manifest.get("ranking_model")
    expected = "hybrid+rerank" if ranking is not None else "hybrid"
    if (
        policy.document.get("ranking_method") != expected
        or policy.document.get("ranking_model") != ranking
    ):
        raise ValueError("Policy is incompatible with the active ranking configuration")
    return policy


def validate_bundle(bundle):
    manifest = load_manifest(bundle)
    actual = {str(p.relative_to(bundle)) for p in bundle.rglob("*") if p.is_file()} - {
        "manifest.json"
    }
    if actual != set(manifest["checksums"]):
        raise ValueError("Artifact inventory does not match the manifest")
    if "taxonomy.sqlite3" not in actual or "aliases.json" not in actual:
        raise ValueError("Missing required bundle files")
    if manifest["artifact_version"] != bundle.name:
        raise ValueError("Artifact directory and manifest identity differ")
    connection = connect(bundle / "taxonomy.sqlite3", readonly=True)
    try:
        if (
            connection.execute("PRAGMA quick_check").fetchone()[0] != "ok"
            or connection.execute("PRAGMA foreign_key_check").fetchall()
        ):
            raise ValueError("Bundle database integrity check failed")
        count = connection.execute("SELECT COUNT(*) FROM category_embeddings").fetchone()[0]
        if count != manifest["embedding_rows"] or count != 2 * manifest["record_count"]:
            raise ValueError("Bundle embedding count mismatch")
        invalid = connection.execute(
            "SELECT COUNT(*) FROM category_embeddings "
            "WHERE taxonomy_key != ? OR taxonomy_version != ? OR vec_length(embedding) != ?",
            (
                manifest["taxonomy_key"],
                manifest["taxonomy_version"],
                manifest["model"]["dimension"],
            ),
        ).fetchone()[0]
        if invalid:
            raise ValueError("Bundle embedding identity or dimension mismatch")
    finally:
        connection.close()
    return manifest


def stage_build(source, aliases, model, root, ranking_profile):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".staging-", dir=root) as temporary:
        staged, manifest = build(source, aliases, model, Path(temporary), ranking_profile)
        validate_bundle(staged)
        inventory = model_inventory(manifest)
        destination = member(root, staged.name)
        with FileLease(root / ".staging.lock"):
            if destination.exists():
                existing = validate_bundle(destination)
                if existing["checksums"] != manifest["checksums"]:
                    raise ValueError("Rebuild changed bundle bytes under the same identity")
            else:
                staged.rename(destination)
        return destination, manifest, inventory


def create_release(root, bundle, policy_path=None):
    manifest = validate_bundle(bundle)
    validate_policy(policy_path, manifest)
    directory = root / ".releases"
    directory.mkdir(exist_ok=True)
    policy = None
    if policy_path is not None:
        value = policy_path.read_bytes()
        identifier = hashlib.sha256(value).hexdigest()
        destination = directory / f"policy-{identifier}.json"
        if destination.exists() and destination.read_bytes() != value:
            raise ValueError("Policy snapshot was modified")
        if not destination.exists():
            atomic_bytes(destination, value)
        policy = {"file": destination.name, "sha256": identifier}
    document = {
        "format_version": 1,
        "artifact_version": bundle.name,
        "manifest_sha256": digest(bundle / "manifest.json"),
        "model_files": model_inventory(manifest),
        "policy": policy,
    }
    content = (json.dumps(document, sort_keys=True) + "\n").encode()
    release_id = hashlib.sha256(content).hexdigest()
    destination = directory / f"{release_id}.json"
    if destination.exists() and destination.read_bytes() != content:
        raise ValueError("Release descriptor was modified")
    if not destination.exists():
        atomic_bytes(destination, content)
    return release_id


def load_release(root, release_id):
    if (
        not isinstance(release_id, str)
        or len(release_id) != 64
        or any(c not in "0123456789abcdef" for c in release_id)
    ):
        raise ValueError("Invalid release identifier")
    release_file = root / ".releases" / f"{release_id}.json"
    if digest(release_file) != release_id:
        raise ValueError("Release descriptor checksum mismatch")
    release = json.loads(release_file.read_text())
    if release.get("format_version") != 1:
        raise ValueError("Unsupported release descriptor")
    bundle = member(root, release["artifact_version"])
    if digest(bundle / "manifest.json") != release["manifest_sha256"]:
        raise ValueError("Release manifest checksum mismatch")
    manifest = validate_bundle(bundle)
    if model_inventory(manifest) != release["model_files"]:
        raise ValueError("Model file checksum mismatch")
    policy_path = None
    if release["policy"] is not None:
        policy_path = member(root / ".releases", release["policy"]["file"])
        if digest(policy_path) != release["policy"]["sha256"]:
            raise ValueError("Policy snapshot checksum mismatch")
        validate_policy(policy_path, manifest)
    return bundle, manifest, policy_path


def load_active(root, *, require_release=False):
    pointer = json.loads((root / "active.json").read_text())
    if "release_id" not in pointer:
        if require_release:
            raise ValueError("Activate the existing bundle to pin model files before serving")
        bundle = resolve_bundle(root)
        return bundle, load_manifest(bundle), None
    bundle, manifest, policy = load_release(root, pointer["release_id"])
    if bundle.name != pointer["artifact_version"]:
        raise ValueError("Active pointer mixes release and bundle generations")
    return bundle, manifest, policy


def activate(root, artifact_version=None, *, policy=None, release_id=None):
    root = root.resolve()
    with FileLease(root / ".service.lock"):
        pointer = root / "active.json"
        old = json.loads(pointer.read_text()) if pointer.exists() else {}
        if release_id is None:
            release_id = create_release(root, member(root, artifact_version), policy)
        bundle, _, _ = load_release(root, release_id)
        if old.get("release_id") == release_id:
            return {"artifact_version": bundle.name, "release_id": release_id, "changed": False}
        value = {"artifact_version": bundle.name, "release_id": release_id}
        if old.get("release_id"):
            value["previous_release_id"] = old["release_id"]
        atomic_json(pointer, value)
        return {**value, "changed": True}


def rollback(root, release_id=None):
    if release_id is None:
        release_id = json.loads((root / "active.json").read_text()).get("previous_release_id")
    if release_id is None:
        raise ValueError("No previous managed release; specify --release-id")
    return activate(root, release_id=release_id)


def copy_bundle(bundle, root):
    """Import an already built immutable bundle, without activation."""
    validate_bundle(bundle)
    root.mkdir(parents=True, exist_ok=True)
    destination = member(root, bundle.name)
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(prefix=".import-", dir=root) as temporary:
        staged = Path(temporary) / bundle.name
        shutil.copytree(bundle, staged)
        validate_bundle(staged)
        staged.rename(destination)
    return destination
