"""Validated local-service settings, separate from quality-affecting artifacts."""

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class APISettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    artifacts: Path = Path("artifacts/google")
    policy: Path | None = None
    runtime_db: Path = Path("var/category_matcher.sqlite3")
    queue_size: int = Field(default=8, ge=1, le=32)
    timeout_seconds: float = Field(default=30, gt=0, le=120)
    max_body_bytes: int = Field(default=16384, ge=1024, le=65536)

    @classmethod
    def from_env(cls):
        fields = {
            "ARTIFACTS": ("artifacts", Path),
            "POLICY": ("policy", Path),
            "RUNTIME_DB": ("runtime_db", Path),
            "QUEUE_SIZE": ("queue_size", int),
            "TIMEOUT_SECONDS": ("timeout_seconds", float),
            "MAX_BODY_BYTES": ("max_body_bytes", int),
        }
        values = {}
        for key, value in os.environ.items():
            if not key.startswith("CATEGORY_MATCHER_API_"):
                continue
            suffix = key.removeprefix("CATEGORY_MATCHER_API_")
            if suffix not in fields:
                raise ValueError(f"Unknown API setting: {key}")
            name, convert = fields[suffix]
            values[name] = convert(value)
        return cls.model_validate(values)
