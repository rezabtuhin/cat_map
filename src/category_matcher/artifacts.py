"""Offline artifact construction and integrity checks for the Google prototype."""

import hashlib
import importlib.metadata
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from category_matcher.encoder import BGEEncoder
from category_matcher.normalization import NORMALIZATION_VERSION
from category_matcher.retrieval import RETRIEVAL_CONFIG, build_bm25
from category_matcher.storage import (
    SCHEMA_VERSION,
    CategoryStore,
    connect,
    create_database,
    vector_blob,
)
from category_matcher.taxonomy import read_google_taxonomy


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def implementation_fingerprint() -> dict[str, str]:
    directory = Path(__file__).parent
    names = (
        "domain.py",
        "normalization.py",
        "taxonomy.py",
        "storage.py",
        "retrieval.py",
        "encoder.py",
        "matcher.py",
        "query.py",
        "artifacts.py",
    )
    return {name: digest(directory / name) for name in names}


def read_aliases(path: Path, valid_ids: set[str]) -> tuple[list[dict], int]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    approved = []
    for row in records:
        if row.get("taxonomy_key") != "google" or row.get("category_id") not in valid_ids:
            raise ValueError("Alias references an unknown taxonomy or category")
        if not isinstance(row.get("alias"), str) or not row["alias"].strip():
            raise ValueError("Alias must contain nonempty text")
        if row.get("status") not in {"pending_review", "approved", "rejected"}:
            raise ValueError("Alias has an invalid review status")
        if row["status"] == "approved":
            if not row.get("reviewer") or not row.get("reviewed_at"):
                raise ValueError("Approved aliases require reviewer and reviewed_at fields")
            approved.append(row)
    return approved, len(records) - len(approved)


def build(source: Path, aliases_path: Path, model_path: Path, root: Path) -> tuple[Path, dict]:
    categories, audit = read_google_taxonomy(source)
    approved, inactive_aliases = read_aliases(aliases_path, {c.category_id for c in categories})
    model = json.loads(model_path.read_text())
    if not isinstance(model.get("revision"), str) or len(model["revision"]) != 40:
        raise ValueError("Download and pin an immutable model revision before building")
    version = audit["source_sha256"][:16]
    packages = {
        name: importlib.metadata.version(name)
        for name in (
            "bm25s",
            "numpy",
            "rapidfuzz",
            "sentence-transformers",
            "torch",
            "sqlite-vec",
            "transformers",
        )
    }
    identity = {
        "source_sha256": audit["source_sha256"],
        "aliases_sha256": digest(aliases_path),
        "model": {k: v for k, v in model.items() if k != "snapshot"},
        "normalization": NORMALIZATION_VERSION,
        "retrieval": RETRIEVAL_CONFIG,
        "schema_version": SCHEMA_VERSION,
        "packages": packages,
        "implementation_fingerprint": implementation_fingerprint(),
    }
    config_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    artifact_version = f"{version}-{config_hash}"
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / artifact_version
    if destination.exists():
        manifest = load_manifest(destination)
        activate(root, destination)
        return destination, manifest

    staging = Path(tempfile.mkdtemp(prefix=".building-", dir=root))
    try:
        connection = connect(staging / "taxonomy.sqlite3")
        try:
            create_database(connection, categories, version, model["dimension"])
            encoder = BGEEncoder(model)
            texts = [text for c in categories for text in (c.label, c.full_path)]
            vectors = encoder.encode_documents(texts, progress=True)
            connection.executemany(
                "INSERT INTO category_embeddings VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        "google",
                        version,
                        c.category_id,
                        view,
                        vector_blob(vectors[2 * i + offset], model["dimension"]),
                    )
                    for i, c in enumerate(categories)
                    for offset, view in enumerate(("label", "path"))
                ],
            )
            connection.commit()
            if connection.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError("Foreign-key check failed")
            extension_version = connection.execute("SELECT vec_version()").fetchone()[0]
            sqlite_version = connection.execute("SELECT sqlite_version()").fetchone()[0]
        finally:
            connection.close()
        build_bm25(categories, staging / "bm25")
        (staging / "aliases.json").write_text(json.dumps(approved, indent=2) + "\n")
        (staging / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        checksums = {
            str(p.relative_to(staging)): digest(p) for p in staging.rglob("*") if p.is_file()
        }
        manifest = {
            **identity,
            "taxonomy_key": "google",
            "taxonomy_version": version,
            "artifact_version": artifact_version,
            "model": model,
            "created_at": datetime.now(UTC).isoformat(),
            "record_count": len(categories),
            "embedding_rows": len(texts),
            "approved_alias_count": len(approved),
            "inactive_alias_count": inactive_aliases,
            "sqlite_version": sqlite_version,
            "sqlite_vec_version": extension_version,
            "checksums": checksums,
            "decision_policy": "sprint-1-provisional-only",
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(destination)
        load_manifest(destination)
        activate(root, destination)
        return destination, manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def activate(root: Path, destination: Path) -> None:
    pointer = root / "active.json"
    with tempfile.NamedTemporaryFile(mode="w", dir=root, delete=False) as stream:
        json.dump({"artifact_version": destination.name}, stream)
        temporary = Path(stream.name)
    temporary.replace(pointer)


def resolve_bundle(root: Path) -> Path:
    name = json.loads((root / "active.json").read_text())["artifact_version"]
    if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Invalid active artifact identifier")
    return root / name


def load_manifest(bundle: Path) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest.get("implementation_fingerprint") != implementation_fingerprint():
        raise ValueError("Matching implementation changed: rebuild artifacts before querying")
    if manifest["taxonomy_key"] != "google" or manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported taxonomy or schema version")
    for package, expected in manifest["packages"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(
                f"Package version changed: {package}; rebuild with the locked environment"
            )
    for relative, expected in manifest["checksums"].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle.resolve()) or digest(path) != expected:
            raise ValueError(f"Artifact integrity failure: {relative}")
    store = CategoryStore(
        bundle / "taxonomy.sqlite3",
        "google",
        manifest["taxonomy_version"],
        manifest["model"]["dimension"],
    )
    try:
        if len(store.categories) != manifest["record_count"]:
            raise ValueError("Category count differs from the manifest")
        if store.connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            raise ValueError("Database schema differs from the manifest")
        actual_version = store.connection.execute("SELECT vec_version()").fetchone()[0]
        if actual_version != manifest["sqlite_vec_version"]:
            raise ValueError("sqlite-vec version differs from the artifact build")
    finally:
        store.close()
    return manifest
