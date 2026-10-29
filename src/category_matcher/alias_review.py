"""Source-bound human review of pilot feedback; approvals write a new alias file."""

import hashlib
import json
from datetime import datetime

from category_matcher.artifacts import digest, read_aliases
from category_matcher.normalization import exact_key
from category_matcher.query import (
    is_attribute_only,
    is_audience_only,
    is_merchandising_label,
    is_placeholder_label,
)
from category_matcher.review import write_new
from category_matcher.taxonomy import read_google_taxonomy


def snapshot_hash(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


def packets(store):
    for feedback in store.feedback_rows():
        event = store.event(feedback["match_event_id"])
        source = {"feedback": feedback, "event": event}
        request = event["request"]
        eligible = (
            feedback["action"] in {"confirm", "correct"}
            and feedback["category"] is not None
            and request.get("language", "en") == "en"
            and not request.get("context")
            and not request.get("source_ancestors")
            and not request.get("parent_category_id")
            and not request.get("category_id")
            and bool(request.get("category"))
            and "|" not in request["category"]
            and not is_merchandising_label(request["category"])
            and not is_attribute_only(request["category"])
            and not is_audience_only(request["category"])
            and not is_placeholder_label(request["category"])
        )
        yield {
            "format_version": 1,
            "feedback_id": feedback["feedback_id"],
            "source_sha256": snapshot_hash(source),
            **source,
            "proposal": {
                "alias": request.get("category") if eligible else None,
                "category_id": feedback["category"]["id"] if eligible else None,
            },
            "decision": {
                "state": "pending",
                "reviewer_kind": None,
                "reviewer": None,
                "reviewed_at": None,
                "reason": None,
            },
        }


def export_feedback(store, output):
    rows = list(packets(store))
    write_new(output, rows)
    return {"output": str(output), "feedback": len(rows), "approved": 0}


def apply_alias_reviews(store, reviews, aliases, taxonomy, output):
    if output.resolve() == aliases.resolve():
        raise ValueError("Write reviewed aliases to a new file")
    categories, audit = read_google_taxonomy(taxonomy)
    by_id = {c.category_id: c for c in categories}
    read_aliases(aliases, set(by_id))
    existing = [json.loads(line) for line in aliases.read_text().splitlines() if line.strip()]
    expected = {p["feedback_id"]: p for p in packets(store)}
    approved, seen = [], set()
    keys = {}
    for category in categories:
        for text in (category.label, category.full_path):
            keys.setdefault(exact_key(text), set()).add(category.category_id)
    for row in existing:
        if row["status"] == "approved":
            keys.setdefault(exact_key(row["alias"]), set()).add(row["category_id"])
    for packet in [json.loads(line) for line in reviews.read_text().splitlines() if line.strip()]:
        identifier = packet["feedback_id"]
        if identifier in seen or identifier not in expected:
            raise ValueError("Unknown or duplicate feedback review")
        seen.add(identifier)
        reference = expected[identifier]
        if {k: v for k, v in packet.items() if k != "decision"} != {
            k: v for k, v in reference.items() if k != "decision"
        }:
            raise ValueError("Changed or stale feedback review binding")
        decision = packet["decision"]
        if decision["state"] in {"pending", "rejected"}:
            continue
        if (
            decision["state"] != "approved"
            or decision.get("reviewer_kind") != "human"
            or any(
                not isinstance(decision.get(k), str) or not decision[k].strip()
                for k in ("reviewer", "reviewed_at", "reason")
            )
        ):
            raise ValueError(
                "Alias approval requires an explicit human reviewer, timestamp, and reason"
            )
        timestamp = datetime.fromisoformat(decision["reviewed_at"].replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            raise ValueError("Review timestamp requires a timezone")
        event = packet["event"]
        alias, cid = packet["proposal"]["alias"], packet["proposal"]["category_id"]
        if (
            not alias
            or cid not in by_id
            or event["taxonomy_key"] != "google"
            or event["taxonomy_version"] != audit["source_sha256"][:16]
            or packet["feedback"]["category"]
            != {
                "id": cid,
                "label": by_id[cid].label,
                "path": by_id[cid].full_path,
            }
        ):
            raise ValueError("Feedback is not eligible for a global alias in this taxonomy")
        targets = keys.setdefault(exact_key(alias), set())
        if targets and targets != {cid}:
            raise ValueError("Alias conflicts with a canonical label or approved alias")
        if any(
            exact_key(r["alias"]) == exact_key(alias)
            and r["category_id"] == cid
            and r["status"] == "approved"
            for r in existing + approved
        ):
            raise ValueError("Alias is already approved")
        targets.add(cid)
        approved.append(
            {
                "taxonomy_key": "google",
                "alias": alias,
                "category_id": cid,
                "status": "approved",
                "reviewer": decision["reviewer"],
                "reviewed_at": decision["reviewed_at"],
                "source": "declared_human_review_of_pilot_feedback",
                "review_reason": decision["reason"],
                "feedback_id": identifier,
                "match_event_id": event["event_id"],
                "artifact_version": event["artifact_version"],
                "review_packet_sha256": digest(reviews),
            }
        )
    if not approved:
        raise ValueError("No human alias approvals supplied")
    # Retire draft entries for newly approved keys instead of retaining conflicting proposals.
    promoted = {exact_key(r["alias"]) for r in approved}
    rows = [
        r for r in existing if r["status"] == "approved" or exact_key(r["alias"]) not in promoted
    ]
    write_new(output, rows + approved)
    return {"output": str(output), "approved": len(approved), "activation_required": True}


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
            "slice": "pilot_feedback",
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
