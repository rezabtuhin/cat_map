"""Framework-independent records shared by the prototype components."""

from dataclasses import dataclass, field
from typing import Literal, Protocol


@dataclass(frozen=True)
class Category:
    category_id: str
    label: str
    full_path: str
    parent_id: str | None
    depth: int
    is_leaf: bool


@dataclass(frozen=True)
class Hit:
    category_id: str
    score: float
    source: str
    rank: int


@dataclass
class MatchResult:
    status: Literal[
        "matched", "provisional", "unsupported_language", "needs_clarification", "no_match"
    ]
    taxonomy_key: str
    taxonomy_version: str
    artifact_version: str
    method: str
    category: dict | None = None
    candidates: list[dict] = field(default_factory=list)
    reason_code: str = "uncalibrated_retrieval"
    ranking_model: dict | None = None
    policy_id: str | None = None
    decision_preview: dict | None = None
    clarification: str | None = None


class CandidateReranker(Protocol):
    def rank(self, query: str, candidates: list[dict]) -> list[dict]: ...
