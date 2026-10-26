"""SQLite owns canonical records, scope relationships, and sqlite-vec vectors."""

import sqlite3
from pathlib import Path

import numpy as np
import sqlite_vec

from category_matcher.domain import Category, Hit

SCHEMA_VERSION = 1


def connect(path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    target = path.resolve().as_uri() + "?mode=ro" if readonly else str(path)
    connection = sqlite3.connect(target, uri=readonly, timeout=5)
    try:
        connection.enable_load_extension(True)
        try:
            sqlite_vec.load(connection)
        finally:
            connection.enable_load_extension(False)
        connection.execute("SELECT vec_version()").fetchone()
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.row_factory = sqlite3.Row
        return connection
    except BaseException:
        connection.close()
        raise


def vector_blob(vector: np.ndarray, dimension: int) -> bytes:
    vector = np.asarray(vector, dtype="<f4")
    if vector.shape != (dimension,) or not np.isfinite(vector).all():
        raise ValueError(f"Expected one finite {dimension}-dimensional vector")
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("A zero-norm embedding is invalid")
    return (vector / norm).astype("<f4").tobytes()


def create_database(
    connection: sqlite3.Connection,
    categories: list[Category],
    version: str,
    dimension: int,
) -> None:
    if not isinstance(dimension, int) or dimension < 1:
        raise ValueError("Invalid embedding dimension")
    connection.executescript(f"""
        CREATE TABLE categories (
          taxonomy_key TEXT NOT NULL, taxonomy_version TEXT NOT NULL,
          category_id TEXT NOT NULL, label TEXT NOT NULL, full_path TEXT NOT NULL,
          parent_id TEXT, depth INTEGER NOT NULL, is_leaf INTEGER NOT NULL,
          PRIMARY KEY (taxonomy_key, taxonomy_version, category_id),
          FOREIGN KEY (taxonomy_key, taxonomy_version, parent_id)
            REFERENCES categories(taxonomy_key, taxonomy_version, category_id)
            DEFERRABLE INITIALLY DEFERRED
        );
        CREATE TABLE category_closure (
          taxonomy_key TEXT NOT NULL, taxonomy_version TEXT NOT NULL,
          ancestor_id TEXT NOT NULL, descendant_id TEXT NOT NULL, distance INTEGER NOT NULL,
          PRIMARY KEY (taxonomy_key, taxonomy_version, ancestor_id, descendant_id),
          FOREIGN KEY (taxonomy_key, taxonomy_version, ancestor_id)
            REFERENCES categories(taxonomy_key, taxonomy_version, category_id),
          FOREIGN KEY (taxonomy_key, taxonomy_version, descendant_id)
            REFERENCES categories(taxonomy_key, taxonomy_version, category_id)
        );
        CREATE TABLE category_embeddings (
          taxonomy_key TEXT NOT NULL, taxonomy_version TEXT NOT NULL,
          category_id TEXT NOT NULL, view TEXT NOT NULL CHECK(view IN ('label', 'path')),
          embedding BLOB NOT NULL CHECK (
            typeof(embedding) = 'blob' AND length(embedding) = {dimension * 4}
            AND vec_type(embedding) = 'float32' AND vec_length(embedding) = {dimension}
          ),
          PRIMARY KEY (taxonomy_key, taxonomy_version, category_id, view),
          FOREIGN KEY (taxonomy_key, taxonomy_version, category_id)
            REFERENCES categories(taxonomy_key, taxonomy_version, category_id)
        );
        PRAGMA user_version={SCHEMA_VERSION};
    """)
    connection.executemany(
        "INSERT INTO categories VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "google",
                version,
                c.category_id,
                c.label,
                c.full_path,
                c.parent_id,
                c.depth,
                c.is_leaf,
            )
            for c in categories
        ],
    )
    by_id = {c.category_id: c for c in categories}
    closure = []
    for category in categories:
        ancestor: Category | None = category
        distance = 0
        while ancestor is not None:
            closure.append(
                ("google", version, ancestor.category_id, category.category_id, distance)
            )
            ancestor = by_id.get(ancestor.parent_id)
            distance += 1
    connection.executemany("INSERT INTO category_closure VALUES (?, ?, ?, ?, ?)", closure)
    connection.commit()


class CategoryStore:
    def __init__(self, path: Path, taxonomy_key: str, version: str, dimension: int):
        self.connection = connect(path, readonly=True)
        self.taxonomy_key = taxonomy_key
        self.version = version
        self.dimension = dimension
        self.categories = {
            row["category_id"]: Category(
                row["category_id"],
                row["label"],
                row["full_path"],
                row["parent_id"],
                row["depth"],
                bool(row["is_leaf"]),
            )
            for row in self.connection.execute(
                "SELECT * FROM categories WHERE taxonomy_key=? AND taxonomy_version=?",
                (taxonomy_key, version),
            )
        }

    def close(self) -> None:
        self.connection.close()

    def allowed_ids(self, parent_id: str | None = None) -> set[str]:
        if parent_id is None:
            return set(self.categories)
        if parent_id not in self.categories:
            raise ValueError(f"Unknown parent category ID: {parent_id}")
        return {
            row[0]
            for row in self.connection.execute(
                "SELECT descendant_id FROM category_closure "
                "WHERE taxonomy_key=? AND taxonomy_version=? AND ancestor_id=?",
                (self.taxonomy_key, self.version, parent_id),
            )
        }

    def semantic_search(
        self,
        vector: np.ndarray,
        limit: int,
        parent_id: str | None = None,
        excluded_ids: set[str] | None = None,
    ) -> list[Hit]:
        if parent_id is not None and parent_id not in self.categories:
            raise ValueError(f"Unknown parent category ID: {parent_id}")
        scope = (
            ""
            if parent_id is None
            else """
            AND EXISTS (
              SELECT 1 FROM category_closure AS c
              WHERE c.taxonomy_key=e.taxonomy_key AND c.taxonomy_version=e.taxonomy_version
                AND c.descendant_id=e.category_id AND c.ancestor_id=:parent_id
            )
        """
        )
        parameters = {
            "vector": vector_blob(vector, self.dimension),
            "taxonomy_key": self.taxonomy_key,
            "version": self.version,
            "parent_id": parent_id,
            "limit": limit,
        }
        if excluded_ids:
            placeholders = []
            for index, cid in enumerate(sorted(excluded_ids)):
                key = f"excluded_{index}"
                parameters[key] = cid
                placeholders.append(":" + key)
            scope += " AND category_id NOT IN (" + ",".join(placeholders) + ")"
        rows = self.connection.execute(
            f"""SELECT category_id, MIN(vec_distance_cosine(embedding, :vector)) AS distance
            FROM category_embeddings AS e
            WHERE taxonomy_key=:taxonomy_key AND taxonomy_version=:version {scope}
            GROUP BY category_id ORDER BY distance, category_id LIMIT :limit""",
            parameters,
        ).fetchall()
        return [Hit(row[0], 1.0 - row[1], "semantic", i) for i, row in enumerate(rows, 1)]
