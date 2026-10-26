import json

import pytest

from category_matcher.artifacts import digest
from category_matcher.datasets import is_reviewed, read_jsonl
from category_matcher.review import apply_reviews, capture_inputs, export_reviews


def packet(bundle, tmp_path):
    taxonomy = bundle.parent / "source.txt"
    dataset = tmp_path / "dataset.jsonl"
    row = {
        "example_id": "real-1",
        "taxonomy_key": "google",
        "taxonomy_version": digest(taxonomy)[:16],
        "query": "portable handsets",
        "context": "",
        "language": "en",
        "parent_category_id": "2",
        "dataset_role": "development",
        "source": "ai_label_proposal_for_real_input",
        "expected_status": "matched",
        "acceptable_category_ids": ["3"],
        "review_status": "pending_review",
        "reviewer": None,
        "reviewed_at": None,
    }
    dataset.write_text(json.dumps(row) + "\n")
    reviews = tmp_path / "reviews.jsonl"
    export_reviews(dataset, taxonomy, reviews)
    return taxonomy, dataset, reviews


def approve(reviews):
    rows = read_jsonl(reviews)
    rows[0]["decision"].update(
        {
            "state": "approved",
            "expected_status": "matched",
            "acceptable_category_ids": ["3"],
            "reviewer_kind": "human",
            "reviewer": "test reviewer",
            "reviewed_at": "2026-10-26T10:00:00Z",
            "reason": "Product intent checked against the canonical paths.",
        }
    )
    reviews.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


def test_export_does_not_approve_and_apply_preserves_origin_and_split(bundle, tmp_path):
    taxonomy, dataset, reviews = packet(bundle, tmp_path)
    before = dataset.read_bytes()
    assert read_jsonl(reviews)[0]["decision"]["state"] == "pending"
    with pytest.raises(ValueError, match="No human review"):
        apply_reviews(dataset, taxonomy, reviews, tmp_path / "empty.jsonl")
    approve(reviews)
    output = tmp_path / "reviewed.jsonl"
    result = apply_reviews(dataset, taxonomy, reviews, output)
    row = read_jsonl(output)[0]
    assert result["unreviewed"] == 0 and is_reviewed(row)
    assert row["source"] == "ai_label_proposal_for_real_input"
    assert row["dataset_role"] == "development"
    assert not result["independent_holdout_created"]
    assert dataset.read_bytes() == before
    with pytest.raises(FileExistsError):
        apply_reviews(dataset, taxonomy, reviews, output)


@pytest.mark.parametrize(
    "mutation", ["query", "source", "cohort", "human", "timestamp", "scope", "ids"]
)
def test_review_import_rejects_stale_or_invalid_decisions(bundle, tmp_path, mutation):
    taxonomy, dataset, reviews = packet(bundle, tmp_path)
    rows = approve(reviews)
    if mutation == "query":
        rows[0]["input"]["query"] = "changed input"
    elif mutation == "source":
        dataset.write_text(dataset.read_text() + "\n")
    elif mutation == "cohort":
        rows.append(rows[0])
    elif mutation == "human":
        rows[0]["decision"]["reviewer_kind"] = "ai"
    elif mutation == "timestamp":
        rows[0]["decision"]["reviewed_at"] = "2026-10-26"
    elif mutation == "scope":
        rows[0]["decision"]["acceptable_category_ids"] = ["7"]
    else:
        rows[0]["decision"]["acceptable_category_ids"] = ["missing"]
    reviews.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        apply_reviews(dataset, taxonomy, reviews, tmp_path / "output.jsonl")
    assert not (tmp_path / "output.jsonl").exists()


def test_capture_creates_unlabeled_records_not_fake_clarification_labels(bundle, tmp_path):
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text(json.dumps({"query": "Erin Recommends", "product_description": None}))
    output = tmp_path / "captured.jsonl"
    result = capture_inputs(inputs, bundle.parent / "source.txt", output, "user-message")
    row = read_jsonl(output)[0]
    assert result["labeled"] == 0 and row["expected_status"] is None
    assert not is_reviewed(row)
    assert row["dataset_role"] == "unassigned_collected_input"
    inputs.write_text(inputs.read_text() + "\n" + inputs.read_text())
    with pytest.raises(ValueError, match="Duplicate"):
        capture_inputs(inputs, bundle.parent / "source.txt", tmp_path / "duplicates.jsonl", "user")


def test_review_cli_does_not_require_an_active_model_artifact(
    bundle, tmp_path, monkeypatch, capsys
):
    from category_matcher.cli import main

    taxonomy, dataset, _ = packet(bundle, tmp_path)
    output = tmp_path / "cli-reviews.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "category-matcher",
            "--artifacts",
            str(tmp_path / "absent"),
            "review-export",
            "--dataset",
            str(dataset),
            "--taxonomy",
            str(taxonomy),
            "--output",
            str(output),
        ],
    )
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["decisions_approved_by_export"] == 0


def test_review_export_supports_id_only_canonical_examples(bundle, tmp_path):
    taxonomy, dataset, _ = packet(bundle, tmp_path)
    row = read_jsonl(dataset)[0]
    row.update({"query": "", "category_id": "3"})
    dataset.write_text(json.dumps(row) + "\n")
    output = tmp_path / "id-review.jsonl"
    assert export_reviews(dataset, taxonomy, output)["examples"] == 1
    assert read_jsonl(output)[0]["input"]["category_id"] == "3"


def test_capture_makes_product_description_available_to_the_matcher_as_context(bundle, tmp_path):
    inputs = tmp_path / "description.jsonl"
    inputs.write_text(
        json.dumps(
            {
                "query": "Erin Recommends",
                "product_description": "portable handsets",
                "parent_category_id": "2",
            }
        )
    )
    output = tmp_path / "described.jsonl"
    capture_inputs(inputs, bundle.parent / "source.txt", output, "user")
    assert read_jsonl(output)[0]["context"] == "portable handsets"
