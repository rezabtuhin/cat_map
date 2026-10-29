"""Collect unlabeled inputs and apply explicitly supplied, source-bound human decisions."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from category_matcher.artifacts import digest
from category_matcher.datasets import is_reviewed, read_jsonl
from category_matcher.normalization import exact_key
from category_matcher.taxonomy import read_google_taxonomy

STATUSES = {"matched", "needs_clarification", "no_match", "unsupported_language"}
INPUT_FIELDS = (
    "query",
    "context",
    "parent_category_id",
    "category_id",
    "language",
    "product_description",
)


def review_input(row):
    fields = INPUT_FIELDS + (("source_ancestors",) if "source_ancestors" in row else ())
    return {key: row.get(key) for key in fields}


def row_hash(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def write_new(path: Path, rows: list[dict]) -> None:
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(content)


def load_source(dataset: Path, taxonomy: Path):
    categories, audit = read_google_taxonomy(taxonomy)
    cats = {c.category_id: c for c in categories}
    rows = read_jsonl(dataset)
    seen = set()
    for row in rows:
        identifier = row["example_id"]
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("Missing or duplicate example ID")
        seen.add(identifier)
        if (row["taxonomy_key"], row["taxonomy_version"]) != (
            "google",
            audit["source_sha256"][:16],
        ):
            raise ValueError("Review dataset belongs to a different taxonomy/version")
        category_id = row.get("category_id")
        if category_id is not None and (
            not isinstance(category_id, str) or category_id not in cats
        ):
            raise ValueError("Unknown explicit category ID")
        if not isinstance(row["query"], str) or (not row["query"].strip() and category_id is None):
            raise ValueError("Review inputs require nonempty query text or an explicit category ID")
        if not all(cid in cats for cid in row.get("acceptable_category_ids", [])):
            raise ValueError("Draft proposal contains an unknown category ID")
    return rows, cats, audit


def export_reviews(dataset: Path, taxonomy: Path, output: Path) -> dict:
    rows, cats, audit = load_source(dataset, taxonomy)
    source_hash = digest(dataset)
    packet = []
    for row in rows:
        ids = row.get("acceptable_category_ids", [])
        packet.append(
            {
                "review_format_version": 1,
                "example_id": row["example_id"],
                "source_dataset_sha256": source_hash,
                "taxonomy_sha256": audit["source_sha256"],
                "source_row_sha256": row_hash(row),
                "input": review_input(row),
                "proposal": {
                    "expected_status": row.get("expected_status"),
                    "acceptable_category_ids": ids,
                    "canonical_paths": {cid: cats[cid].full_path for cid in ids},
                    "label_origin": row.get("source"),
                    "existing_review_status": row.get("review_status"),
                },
                "decision": {
                    "state": "pending",
                    "expected_status": None,
                    "acceptable_category_ids": [],
                    "reviewer_kind": None,
                    "reviewer": None,
                    "reviewed_at": None,
                    "reason": None,
                },
            }
        )
    write_new(output, packet)
    return {
        "output": str(output),
        "examples": len(rows),
        "decisions_approved_by_export": 0,
        "source_dataset_sha256": source_hash,
    }


def apply_reviews(dataset: Path, taxonomy: Path, reviews: Path, output: Path) -> dict:
    rows, cats, audit = load_source(dataset, taxonomy)
    packet = read_jsonl(reviews)
    by_id = {row["example_id"]: row for row in packet}
    if len(by_id) != len(packet) or set(by_id) != {row["example_id"] for row in rows}:
        raise ValueError(
            "Review packet must contain the entire unchanged example cohort exactly once"
        )
    source_hash = digest(dataset)
    approved, updated = 0, []
    for row in rows:
        review = by_id[row["example_id"]]
        if (
            review.get("review_format_version") != 1
            or review.get("source_dataset_sha256") != source_hash
            or review.get("taxonomy_sha256") != audit["source_sha256"]
            or review.get("source_row_sha256") != row_hash(row)
            or review.get("input") != review_input(row)
        ):
            raise ValueError("Stale or changed review input/source binding")
        decision = review["decision"]
        copy = dict(row)
        if decision["state"] == "approved":
            status, ids = decision["expected_status"], decision["acceptable_category_ids"]
            if (
                not isinstance(status, str)
                or status not in STATUSES
                or not isinstance(ids, list)
                or bool(ids) != (status == "matched")
                or any(not isinstance(cid, str) or cid not in cats for cid in ids)
                or len(ids) != len(set(ids))
            ):
                raise ValueError("Invalid reviewed status or category IDs")
            if decision.get("reviewer_kind") != "human" or any(
                not isinstance(decision.get(key), str) or not decision[key].strip()
                for key in ("reviewer", "reason", "reviewed_at")
            ):
                raise ValueError(
                    "Approval requires an explicit human reviewer, reason, and timestamp"
                )
            try:
                timestamp = datetime.fromisoformat(decision["reviewed_at"].replace("Z", "+00:00"))
                if timestamp.tzinfo is None:
                    raise ValueError("Timezone required")
            except ValueError as error:
                raise ValueError(
                    "Review timestamp must include a date, time, and timezone"
                ) from error
            parent = row.get("parent_category_id")
            if parent is not None and (
                parent not in cats
                or any(
                    cid != parent
                    and not cats[cid].full_path.startswith(cats[parent].full_path + " > ")
                    for cid in ids
                )
            ):
                raise ValueError("Reviewed category contradicts the explicit parent constraint")
            copy.update(
                {
                    "expected_status": status,
                    "acceptable_category_ids": ids,
                    "review_status": "approved",
                    "reviewer": decision["reviewer"],
                    "reviewed_at": decision["reviewed_at"],
                    "review_reason": decision["reason"],
                    "review_provenance": {
                        "kind": "declared_human_review",
                        "packet_sha256": digest(reviews),
                        "source_dataset_sha256": source_hash,
                    },
                }
            )
            approved += 1
        elif decision["state"] != "pending":
            raise ValueError("Review state must be pending or approved")
        updated.append(copy)
    if not approved:
        raise ValueError("No human review decisions supplied; nothing to apply")
    write_new(output, updated)
    return {
        "output": str(output),
        "new_decisions": approved,
        "examples": len(updated),
        "unreviewed": sum(not is_reviewed(row) for row in updated),
        "independent_holdout_created": False,
    }


def capture_inputs(inputs: Path, taxonomy: Path, output: Path, source_name: str) -> dict:
    if not source_name.strip():
        raise ValueError("A collection source name is required")
    categories, audit = read_google_taxonomy(taxonomy)
    valid_ids = {category.category_id for category in categories}
    now = datetime.now(UTC).isoformat()
    rows, seen = [], set()
    for item in read_jsonl(inputs):
        query, context = item.get("query"), item.get("context", "")
        if not isinstance(context, str):
            raise ValueError("Context must be text")
        description = item.get("product_description")
        if description is not None:
            if not isinstance(description, str):
                raise ValueError("Product description must be text")
            if context and description and context.strip() != description.strip():
                raise ValueError(
                    "Put product descriptions in context; use parent_category_id for scope"
                )
            context = context or description
        if (
            not isinstance(query, str)
            or not query.strip()
            or len(query) > 300
            or not isinstance(context, str)
            or len(context) > 1000
        ):
            raise ValueError(
                "Input requires a category query up to 300 and context up to 1000 characters"
            )
        parent = item.get("parent_category_id")
        language = item.get("language", "en")
        if parent is not None and (not isinstance(parent, str) or parent not in valid_ids):
            raise ValueError("Unknown parent category ID")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("Language must be a nonempty language code")
        key = (exact_key(query), exact_key(context), parent, language)
        if key in seen:
            raise ValueError("Duplicate collected input")
        seen.add(key)
        identifier = "collected-" + row_hash({"key": key})[:16]
        rows.append(
            {
                "example_id": identifier,
                "taxonomy_key": "google",
                "taxonomy_version": audit["source_sha256"][:16],
                "query": query.strip(),
                "context": context.strip(),
                "parent_category_id": item.get("parent_category_id"),
                "language": item.get("language", "en"),
                "product_description": item.get("product_description"),
                "input_family": item.get("input_family") or exact_key(query),
                "slice": "collected_unlabeled",
                "source": "declared_user_supplied_input",
                "dataset_role": "unassigned_collected_input",
                "collection_provenance": {"source_name": source_name, "received_at": now},
                "expected_status": None,
                "acceptable_category_ids": [],
                "review_status": "pending_review",
                "reviewer": None,
                "reviewed_at": None,
            }
        )
    write_new(output, rows)
    return {
        "output": str(output),
        "examples": len(rows),
        "labeled": 0,
        "independent_holdout_created": False,
    }


def export_pilot_inputs(store, output):
    events = {}
    for event in store.events():
        request = event["request"]
        key = (
            exact_key(request.get("category") or ""),
            exact_key(request.get("context") or ""),
            request.get("category_id"),
            request.get("parent_category_id"),
            request.get("language", "en"),
            tuple(request.get("source_ancestors") or []),
        )
        events.setdefault(key, event)
    rows = [
        {
            "example_id": "pilot-" + event["event_id"],
            "taxonomy_key": event["taxonomy_key"],
            "taxonomy_version": event["taxonomy_version"],
            "query": event["request"].get("category") or "",
            "context": event["request"].get("context") or "",
            "parent_category_id": event["request"].get("parent_category_id"),
            "category_id": event["request"].get("category_id"),
            "language": event["request"].get("language", "en"),
            "source_ancestors": event["request"].get("source_ancestors", []),
            "input_family": exact_key(
                event["request"].get("category") or event["request"]["category_id"]
            ),
            "slice": "pilot_input",
            "source": "pilot_input_with_model_output_already_inspected",
            "dataset_role": "pilot_development_only",
            "review_status": "pending_review",
            "expected_status": None,
            "acceptable_category_ids": [],
            "collection_provenance": {
                "match_event_id": event["event_id"],
                "artifact_version": event["artifact_version"],
            },
        }
        for event in events.values()
    ]
    write_new(output, rows)
    return {"output": str(output), "examples": len(rows), "independent_holdout_created": False}
