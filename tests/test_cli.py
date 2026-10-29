import json

import numpy as np

from category_matcher.cli import main
from category_matcher.decision import write_policy


def test_cli_serializes_confirmed_lookup_and_rejects_empty_input(bundle, monkeypatch, capsys):
    (bundle.parent / "active.json").write_text(json.dumps({"artifact_version": bundle.name}))
    arguments = ["category-matcher", "--artifacts", str(bundle.parent), "match"]
    monkeypatch.setattr("sys.argv", arguments + ["Mobile Phones"])
    assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "matched" and result["category"]["id"] == "3"
    monkeypatch.setattr("sys.argv", arguments + [" "])
    assert main() == 2
    assert "nonempty" in capsys.readouterr().err


def test_cli_draft_policy_is_preview_only_and_normal_loading_fails(bundle, monkeypatch, capsys):
    (bundle.parent / "active.json").write_text(json.dumps({"artifact_version": bundle.name}))
    path = bundle.parent / "policy.json"
    write_policy(
        path,
        {
            "format_version": 1,
            "taxonomy_key": "google",
            "artifact_version": "unit-test-bundle",
            "ranking_method": "hybrid",
            "ranking_model": None,
            "thresholds": {
                "min_semantic": 0.5,
                "min_margin": 0,
                "no_match_semantic": 0.3,
                "fuzzy_support": 75,
            },
            "calibration": {
                "example_count": 200,
                "unreviewed_examples": 200,
                "dataset_sha256": "draft",
            },
            "qualification": {"passed": False},
        },
    )
    monkeypatch.setattr(
        "category_matcher.matcher.Matcher.encode_queries",
        lambda self, texts: np.array([[1.0, 0.0, 0.0]]),
    )
    arguments = ["category-matcher", "--artifacts", str(bundle.parent), "match", "mobil phones"]
    monkeypatch.setattr("sys.argv", arguments + ["--experimental-policy", str(path)])
    assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "provisional" and result["category"] is None
    assert result["decision_preview"]["status"] == "matched"
    monkeypatch.setattr("sys.argv", arguments + ["--policy", str(path)])
    assert main() == 2
    assert "not independently validated" in capsys.readouterr().err


def test_runtime_cli_migrate_backup_restore_and_exports_without_models(
    tmp_path, monkeypatch, capsys
):
    runtime = tmp_path / "runtime.sqlite3"
    base = ["category-matcher", "--runtime-db", str(runtime)]
    for args in (
        ["runtime-migrate"],
        ["runtime-backup", "--output", str(tmp_path / "backup.sqlite3")],
        [
            "runtime-restore",
            "--backup",
            str(tmp_path / "backup.sqlite3"),
            "--output",
            str(tmp_path / "restored.sqlite3"),
        ],
        ["pilot-export", "--output", str(tmp_path / "pilot.jsonl")],
    ):
        monkeypatch.setattr("sys.argv", base + args)
        assert main() == 0
        assert isinstance(json.loads(capsys.readouterr().out), dict)
    monkeypatch.setattr(
        "sys.argv",
        base
        + [
            "runtime-restore",
            "--backup",
            str(tmp_path / "backup.sqlite3"),
            "--output",
            str(runtime),
        ],
    )
    assert main() == 2
    assert "exists" in capsys.readouterr().err.lower()
