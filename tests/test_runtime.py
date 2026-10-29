import json
import sqlite3

import pytest

from category_matcher.migrations import migrate, validate
from category_matcher.runtime import RuntimeStore, restore


def event(store, identifier="event"):
    store.record_match(
        {"category": "phones", "context": None, "language": "en"},
        {
            "event_id": identifier,
            "taxonomy_key": "google",
            "taxonomy_version": "v1",
            "artifact_version": "a1",
            "category": None,
            "status": "provisional",
            "alternatives": [],
        },
    )


def test_runtime_upgrade_from_previous_schema_preserves_events(tmp_path):
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    with store.connection() as connection:
        assert migrate(connection, target=1) == 1
    event(store)
    store.initialize()
    store.initialize()
    assert store.event("event")["request"]["category"] == "phones"
    with store.connection() as connection:
        assert validate(connection) == 2
        assert connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 2
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


@pytest.mark.parametrize("corrupt", ["checksum", "future", "missing_history"])
def test_invalid_migration_state_is_rejected(tmp_path, corrupt):
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    store.initialize()
    with store.connection() as connection, connection:
        if corrupt == "checksum":
            connection.execute("UPDATE schema_migrations SET checksum='changed' WHERE version=1")
        elif corrupt == "future":
            connection.execute("PRAGMA user_version=99")
        else:
            connection.execute("DROP TABLE schema_migrations")
    with pytest.raises(ValueError):
        store.initialize()


def test_failed_upgrade_is_atomic(tmp_path, monkeypatch):
    import category_matcher.migrations as module

    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    with store.connection() as connection:
        migrate(connection, target=1)
    event(store)
    monkeypatch.setattr(
        module,
        "MIGRATIONS",
        (
            module.MIGRATIONS[0],
            (2, "failed", ("CREATE TABLE partial(value)", "NOT VALID SQL")),
        ),
    )
    with store.connection() as connection:
        with pytest.raises(sqlite3.OperationalError):
            migrate(connection)
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert (
            connection.execute("SELECT name FROM sqlite_master WHERE name='partial'").fetchone()
            is None
        )
    assert store.event("event") is not None


def test_runtime_refuses_unrecognized_database(tmp_path):
    store = RuntimeStore(tmp_path / "taxonomy.sqlite3")
    with store.connection() as connection, connection:
        connection.execute("CREATE TABLE categories(id)")
    with pytest.raises(ValueError, match="Unrecognized"):
        store.initialize()


def test_live_backup_ignores_uncommitted_writes_and_restore_never_overwrites(tmp_path):
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    store.initialize()
    event(store)
    backup, restored = tmp_path / "backup.sqlite3", tmp_path / "restored.sqlite3"
    with store.connection() as writer:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute(
            "UPDATE match_events SET request_json=?", (json.dumps({"category": "uncommitted"}),)
        )
        store.backup(backup)
        writer.rollback()
    restore(backup, restored)
    assert RuntimeStore(restored).event("event") == store.event("event")
    with pytest.raises(FileExistsError):
        restore(backup, restored)
    with pytest.raises(FileExistsError):
        store.backup(store.path)
    backup.write_bytes(backup.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        restore(backup, tmp_path / "bad.sqlite3")
    assert not (tmp_path / "bad.sqlite3").exists()
