"""HTTP contracts keep accepted assignments separate from suggestions."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class MatchRequest(StrictModel):
    taxonomy_key: Literal["google"]
    taxonomy_version: str | None = Field(default=None, min_length=1, max_length=128)
    category: str | None = Field(default=None, max_length=300)
    category_id: str | None = Field(default=None, min_length=1, max_length=128)
    parent_category_id: str | None = Field(default=None, min_length=1, max_length=128)
    context: str | None = Field(default=None, max_length=1000)
    language: str = Field(
        default="en", min_length=2, max_length=32, pattern=r"^[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$"
    )
    limit: int = Field(default=3, ge=1, le=10)

    @field_validator(
        "category",
        "context",
        "category_id",
        "parent_category_id",
        "taxonomy_version",
        mode="before",
    )
    @classmethod
    def trim_text(cls, value):
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def require_input(self):
        if not self.category and self.category_id is None:
            raise ValueError("Provide category text or an explicit category_id")
        return self


class CanonicalCategory(StrictModel):
    id: str
    label: str
    path: str


class Timings(StrictModel):
    queue_wait_ms: float = Field(ge=0)
    inference_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)


class MatchResponse(StrictModel):
    status: Literal[
        "matched", "provisional", "needs_clarification", "no_match", "unsupported_language"
    ]
    taxonomy_key: Literal["google"]
    taxonomy_version: str
    artifact_version: str
    category: CanonicalCategory | None
    alternatives: list[CanonicalCategory]
    method: str
    reason_code: str
    clarification: str | None
    event_id: str
    timings: Timings

    @model_validator(mode="after")
    def assignment_contract(self):
        if (self.status == "matched") != (self.category is not None):
            raise ValueError("Only an accepted matched result may contain an assigned category")
        return self


class CategoryResponse(CanonicalCategory):
    taxonomy_key: Literal["google"]
    taxonomy_version: str
    artifact_version: str
    parent_category_id: str | None
    depth: int
    is_leaf: bool


class TaxonomyInfo(StrictModel):
    taxonomy_key: Literal["google"]
    taxonomy_version: str
    artifact_version: str
    languages: list[str]
    category_count: int


class ErrorDetail(StrictModel):
    code: str
    message: str
    details: list[dict] | None = None


class ErrorResponse(StrictModel):
    error: ErrorDetail
    event_id: str
