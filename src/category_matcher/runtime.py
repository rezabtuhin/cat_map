"""Short-lived SQLite connections for match events and backups."""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from category_matcher.migrations import LATEST_VERSION, migrate, validate


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class RuntimeStore:
    def __init__(self, path: Path):
        self.path = path.resolve()

    @contextmanager
    def connection(self, *, readonly=False):
        target = self.path.as_uri() + "?mode=ro" if readonly else str(self.path)
        connection = sqlite3.connect(target, uri=readonly, timeout=2)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            migrate(connection)
            connection.execute("PRAGMA journal_mode=WAL")

    def record_match(self, request, response):
        with self.connection() as connection, connection:
            connection.execute(
                "INSERT INTO match_events VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    response["event_id"],
                    datetime.now(UTC).isoformat(),
                    response["taxonomy_key"],
                    response["taxonomy_version"],
                    response["artifact_version"],
                    json.dumps(request),
                    json.dumps(response),
                ),
            )

    def event(self, event_id):
        with self.connection(readonly=True) as connection:
            row = connection.execute(
                "SELECT * FROM match_events WHERE event_id=?", (event_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            **{k: row[k] for k in row.keys() if k not in {"request_json", "response_json"}},
            "request": json.loads(row["request_json"]),
            "response": json.loads(row["response_json"]),
        }

    def events(self):
        with self.connection(readonly=True) as connection:
            identifiers = [
                r[0]
                for r in connection.execute(
                    "SELECT event_id FROM match_events ORDER BY created_at, event_id"
                )
            ]
        return [self.event(identifier) for identifier in identifiers]

    def backup(self, output: Path):
        output = output.resolve()
        marker = Path(str(output) + ".sha256")
        if marker.exists():
            raise ValueError("Backup checksum already exists")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb"):
            pass
        try:
            with self.connection(readonly=True) as source:
                if validate(source) != LATEST_VERSION:
                    raise ValueError("Migrate runtime data before backing up")
                destination = sqlite3.connect(output)
                try:
                    source.backup(destination)
                finally:
                    destination.close()
            verify_database(output)
            with marker.open("x") as stream:
                stream.write(file_digest(output) + "\n")
        except BaseException:
            output.unlink(missing_ok=True)
            raise
        return {"backup": str(output), "sha256": file_digest(output)}


def verify_database(path):
    store = RuntimeStore(Path(path))
    with store.connection(readonly=True) as connection:
        validate(connection)
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Runtime integrity check failed")
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("Runtime foreign-key check failed")


def restore(backup: Path, output: Path):
    expected = Path(str(backup) + ".sha256").read_text().strip()
    if file_digest(backup) != expected:
        raise ValueError("Backup checksum mismatch")
    verify_database(backup)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb"):
        pass
    try:
        with RuntimeStore(backup).connection(readonly=True) as source:
            destination = sqlite3.connect(output)
            try:
                source.backup(destination)
            finally:
                destination.close()
        verify_database(output)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return {"restored": str(output), "schema_version": LATEST_VERSION}
