"""Ordered migrations for writable runtime state; taxonomy schema stays version 1."""

import hashlib
import sqlite3
from datetime import UTC, datetime

MIGRATIONS = (
    (
        1,
        "match_events",
        (
            """CREATE TABLE match_events (
                event_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                taxonomy_key TEXT NOT NULL, taxonomy_version TEXT NOT NULL,
                artifact_version TEXT NOT NULL, request_json TEXT NOT NULL,
                response_json TEXT NOT NULL
            )""",
            "CREATE INDEX events_created ON match_events(created_at, event_id)",
        ),
    ),
    (
        2,
        "pending_feedback",
        (
            """CREATE TABLE feedback (
                feedback_id TEXT PRIMARY KEY,
                match_event_id TEXT NOT NULL REFERENCES match_events(event_id),
                created_at TEXT NOT NULL, action TEXT NOT NULL
                  CHECK(action IN ('confirm', 'correct', 'reject', 'needs_clarification')),
                category_json TEXT, reviewer TEXT NOT NULL, note TEXT NOT NULL,
                review_status TEXT NOT NULL DEFAULT 'pending' CHECK(review_status='pending'),
                request_id TEXT NOT NULL UNIQUE
            )""",
            "CREATE INDEX feedback_event ON feedback(match_event_id)",
        ),
    ),
)
LATEST_VERSION = MIGRATIONS[-1][0]


def checksum(statements):
    return hashlib.sha256("\n".join(statements).encode()).hexdigest()


def validate(connection):
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if not 0 <= version <= LATEST_VERSION:
        raise ValueError("Unsupported runtime database version")
    tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if version == 0:
        if tables:
            raise ValueError("Unrecognized database; refusing runtime migrations")
        return 0
    if "schema_migrations" not in tables:
        raise ValueError("Missing migration history")
    history = [
        tuple(row)
        for row in connection.execute(
            "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
        )
    ]
    expected = [(v, name, checksum(sql)) for v, name, sql in MIGRATIONS[:version]]
    if history != expected:
        raise ValueError("Migration history or checksum mismatch")
    return version


def migrate(connection: sqlite3.Connection, target: int = LATEST_VERSION):
    if type(target) is not int or not 1 <= target <= LATEST_VERSION:
        raise ValueError("Unsupported migration target")
    connection.execute("BEGIN IMMEDIATE")
    try:
        current = validate(connection)
        if target < current:
            raise ValueError("Runtime schema downgrades are not supported")
        if current == 0:
            connection.execute("""CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY, name TEXT NOT NULL,
                checksum TEXT NOT NULL, applied_at TEXT NOT NULL
            )""")
        for version, name, statements in MIGRATIONS[current:target]:
            for sql in statements:
                connection.execute(sql)
            connection.execute(
                "INSERT INTO schema_migrations VALUES (?, ?, ?, ?)",
                (version, name, checksum(statements), datetime.now(UTC).isoformat()),
            )
            connection.execute(f"PRAGMA user_version={version}")
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    return target
