"""Feedback application rules; storage is injected and feedback stays unreviewed."""

from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4


class FeedbackRepository(Protocol):
    def event(self, event_id: str) -> dict | None: ...
    def add_feedback(self, record: dict) -> None: ...


class FeedbackError(ValueError):
    def __init__(self, code, message, status=422):
        super().__init__(message)
        self.code, self.status = code, status


def submit_feedback(repository: FeedbackRepository, payload, event_id, categories, version):
    event = repository.event(payload["match_event_id"])
    if event is None:
        raise FeedbackError("match_event_not_found", "Unknown match event ID", 404)
    if event["taxonomy_key"] != "google" or event["taxonomy_version"] != version:
        raise FeedbackError("feedback_version_unavailable", "Original taxonomy is not active", 409)
    assigned = event["response"]["category"]
    category_id = payload["category_id"]
    action = payload["action"]
    if action == "confirm":
        offered = [assigned] if assigned else event["response"]["alternatives"]
        if category_id is None and assigned:
            category_id = assigned["id"]
        if category_id is None or category_id not in {c["id"] for c in offered}:
            raise FeedbackError("invalid_confirmation", "Choose an assigned or suggested category")
    if action in {"confirm", "correct"}:
        if category_id not in categories:
            raise FeedbackError("unknown_feedback_category", "Choose a canonical category ID")
        selected = categories[category_id]
        parent_id = event["request"].get("parent_category_id")
        if parent_id is not None and (
            parent_id not in categories
            or (
                category_id != parent_id
                and not selected.full_path.startswith(categories[parent_id].full_path + " > ")
            )
        ):
            raise FeedbackError(
                "feedback_scope_conflict", "Category contradicts the original scope"
            )
        category = {"id": selected.category_id, "label": selected.label, "path": selected.full_path}
    else:
        if category_id is not None:
            raise FeedbackError("invalid_feedback_category", "This action cannot assign a category")
        category = None
    record = {
        "feedback_id": str(uuid4()),
        "match_event_id": event["event_id"],
        "created_at": datetime.now(UTC).isoformat(),
        "action": action,
        "category": category,
        "reviewer": payload["reviewer"],
        "note": payload["note"],
        "event_id": event_id,
        "review_status": "pending",
        "taxonomy_key": event["taxonomy_key"],
        "taxonomy_version": event["taxonomy_version"],
        "artifact_version": event["artifact_version"],
    }
    repository.add_feedback(record)
    return record
