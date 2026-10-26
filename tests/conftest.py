import json
from pathlib import Path

import numpy as np
import pytest

from category_matcher.artifacts import digest, implementation_fingerprint
from category_matcher.retrieval import RETRIEVAL_CONFIG, build_bm25
from category_matcher.storage import SCHEMA_VERSION, connect, create_database, vector_blob
from category_matcher.taxonomy import read_google_taxonomy


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    source = tmp_path / "source.txt"
    source.write_text(
        "1 - Products\n2 - Products > Electronics\n"
        "3 - Products > Electronics > Mobile Phones\n"
        "4 - Products > Electronics > Mobile Phone Cases\n"
        "5 - Products > Pet Supplies\n6 - Products > Pet Supplies > Cat Toys\n"
        "7 - Products > Pet Supplies > Dog Food\n"
    )
    categories, audit = read_google_taxonomy(source)
    version = audit["source_sha256"][:16]
    result = tmp_path / "bundle"
    result.mkdir()
    connection = connect(result / "taxonomy.sqlite3")
    create_database(connection, categories, version, 3)
    vectors = {
        "1": [1, 1, 1],
        "2": [1, 0, 0.5],
        "3": [1, 0, 0],
        "4": [1, 0, 0.2],
        "5": [0, 1, 0.5],
        "6": [0, 1, 0],
        "7": [0, 1, 0.2],
    }
    connection.executemany(
        "INSERT INTO category_embeddings VALUES (?, ?, ?, ?, ?)",
        [
            (
                "google",
                version,
                c.category_id,
                view,
                vector_blob(np.array(vectors[c.category_id]), 3),
            )
            for c in categories
            for view in ("label", "path")
        ],
    )
    connection.commit()
    extension = connection.execute("SELECT vec_version()").fetchone()[0]
    connection.close()
    build_bm25(categories, result / "bm25")
    (result / "aliases.json").write_text(
        json.dumps(
            [
                {"alias": "cellphones", "category_id": "3"},
                {"alias": "conflicting alias", "category_id": "3"},
                {"alias": "conflicting alias", "category_id": "4"},
            ]
        )
    )
    manifest = {
        "taxonomy_key": "google",
        "taxonomy_version": version,
        "artifact_version": "unit-test-bundle",
        "implementation_fingerprint": implementation_fingerprint(),
        "schema_version": SCHEMA_VERSION,
        "sqlite_vec_version": extension,
        "record_count": 7,
        "embedding_rows": 14,
        "source_sha256": audit["source_sha256"],
        "model": {"dimension": 3},
        "retrieval": RETRIEVAL_CONFIG,
        "packages": {},
        "checksums": {
            str(p.relative_to(result)): digest(p) for p in result.rglob("*") if p.is_file()
        },
    }
    (result / "manifest.json").write_text(json.dumps(manifest))
    return result
