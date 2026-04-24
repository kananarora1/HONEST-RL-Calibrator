"""Unified problem schema shared across all ingested datasets.

Every ingestion script in ``data/ingestion/`` emits records conforming to
``UnifiedProblem``. The schema is deliberately domain-agnostic at the top
level: domain-specific payloads (code test cases, math answer types,
logic attribute grids) live inside ``verification_metadata``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Union

from pydantic import BaseModel, Field
from typing_extensions import Literal


Domain = Literal["math", "code", "logic"]


class UnifiedProblem(BaseModel):
    """Canonical record format for a single reasoning problem."""

    problem_id: str = Field(
        ...,
        min_length=1,
        description="Stable, deterministic ID (e.g. 'hendrycks_math_algebra_42').",
    )
    domain: Domain = Field(..., description="Reasoning domain of the problem.")
    difficulty: int = Field(
        ...,
        ge=1,
        le=5,
        description="Integer difficulty rating on a 1-5 scale.",
    )
    source: str = Field(
        ...,
        min_length=1,
        description="Originating dataset tag (e.g. 'hendrycks_math', 'mbpp').",
    )
    question: str = Field(
        ...,
        min_length=1,
        description="Problem text as presented to the model.",
    )
    canonical_answer: Union[str, Dict[str, Any]] = Field(
        ...,
        description="Ground-truth answer. String for math/code, dict for logic grids.",
    )
    verification_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Source-specific data the domain verifier needs.",
    )
    raw_source_entry: Dict[str, Any] = Field(
        default_factory=dict,
        description="Original dataset row, preserved for debugging and traceability.",
    )

    model_config = {"extra": "forbid"}

    def to_jsonl(self) -> str:
        """Serialize to a single-line JSON string suitable for JSONL files."""
        return self.model_dump_json()

    @classmethod
    def from_jsonl(cls, line: str) -> "UnifiedProblem":
        """Parse a single JSONL line back into a ``UnifiedProblem`` instance."""
        return cls.model_validate(json.loads(line))
