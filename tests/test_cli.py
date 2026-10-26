import json

from category_matcher.cli import main


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
