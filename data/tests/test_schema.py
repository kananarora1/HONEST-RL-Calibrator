"""Tests for the unified problem schema."""

import json

import pytest
from pydantic import ValidationError

from data.schema import UnifiedProblem


def _math_problem() -> UnifiedProblem:
    return UnifiedProblem(
        problem_id="hendrycks_math_algebra_42",
        domain="math",
        difficulty=3,
        source="hendrycks_math",
        question="Solve for x: 2x + 3 = 11.",
        canonical_answer="4",
        verification_metadata={"answer_type": "integer"},
        raw_source_entry={"level": "Level 3", "type": "Algebra"},
    )


def _code_problem() -> UnifiedProblem:
    return UnifiedProblem(
        problem_id="mbpp_0123",
        domain="code",
        difficulty=2,
        source="mbpp",
        question="Write a function add(a, b) that returns a + b.",
        canonical_answer="def add(a, b):\n    return a + b\n",
        verification_metadata={
            "tests": [
                "assert add(1, 2) == 3",
                "assert add(-1, 1) == 0",
            ],
            "entry_point": "add",
        },
        raw_source_entry={"task_id": 123},
    )


def _logic_problem() -> UnifiedProblem:
    return UnifiedProblem(
        problem_id="zebralogic_regen_000017",
        domain="logic",
        difficulty=4,
        source="zebralogic_regen",
        question="Three houses, three colors. House 1 is red...",
        canonical_answer={
            "house_1": {"color": "red", "pet": "cat"},
            "house_2": {"color": "blue", "pet": "dog"},
            "house_3": {"color": "green", "pet": "fish"},
        },
        verification_metadata={"n_houses": 3, "attributes": ["color", "pet"]},
        raw_source_entry={"seed": 17},
    )


class TestValidConstruction:
    def test_math_problem_constructs(self):
        p = _math_problem()
        assert p.domain == "math"
        assert p.difficulty == 3
        assert p.canonical_answer == "4"
        assert p.verification_metadata["answer_type"] == "integer"

    def test_code_problem_constructs(self):
        p = _code_problem()
        assert p.domain == "code"
        assert isinstance(p.canonical_answer, str)
        assert "tests" in p.verification_metadata
        assert len(p.verification_metadata["tests"]) == 2

    def test_logic_problem_constructs_with_dict_answer(self):
        p = _logic_problem()
        assert p.domain == "logic"
        assert isinstance(p.canonical_answer, dict)
        assert p.canonical_answer["house_1"]["color"] == "red"

    def test_defaults_for_optional_dicts(self):
        p = UnifiedProblem(
            problem_id="procedural_math_0",
            domain="math",
            difficulty=1,
            source="procedural",
            question="2 + 2 = ?",
            canonical_answer="4",
        )
        assert p.verification_metadata == {}
        assert p.raw_source_entry == {}


class TestRoundTripSerialization:
    @pytest.mark.parametrize(
        "factory", [_math_problem, _code_problem, _logic_problem]
    )
    def test_round_trip_preserves_all_fields(self, factory):
        original = factory()
        line = original.to_jsonl()

        # jsonl contract: exactly one line, valid JSON.
        assert "\n" not in line
        assert isinstance(json.loads(line), dict)

        restored = UnifiedProblem.from_jsonl(line)
        assert restored == original
        assert restored.model_dump() == original.model_dump()

    def test_round_trip_preserves_nested_dict_answer(self):
        original = _logic_problem()
        restored = UnifiedProblem.from_jsonl(original.to_jsonl())
        assert restored.canonical_answer == original.canonical_answer
        assert restored.raw_source_entry == original.raw_source_entry


class TestValidationFailures:
    def test_invalid_domain_rejected(self):
        with pytest.raises(ValidationError):
            UnifiedProblem(
                problem_id="x",
                domain="physics",  # type: ignore[arg-type]
                difficulty=1,
                source="s",
                question="q",
                canonical_answer="a",
            )

    @pytest.mark.parametrize("bad_difficulty", [0, -1, 6, 10])
    def test_difficulty_out_of_range_rejected(self, bad_difficulty):
        with pytest.raises(ValidationError):
            UnifiedProblem(
                problem_id="x",
                domain="math",
                difficulty=bad_difficulty,
                source="s",
                question="q",
                canonical_answer="a",
            )

    @pytest.mark.parametrize(
        "missing_field",
        [
            "problem_id",
            "domain",
            "difficulty",
            "source",
            "question",
            "canonical_answer",
        ],
    )
    def test_missing_required_field_rejected(self, missing_field):
        payload = {
            "problem_id": "x",
            "domain": "math",
            "difficulty": 1,
            "source": "s",
            "question": "q",
            "canonical_answer": "a",
        }
        payload.pop(missing_field)
        with pytest.raises(ValidationError):
            UnifiedProblem(**payload)

    def test_empty_problem_id_rejected(self):
        with pytest.raises(ValidationError):
            UnifiedProblem(
                problem_id="",
                domain="math",
                difficulty=1,
                source="s",
                question="q",
                canonical_answer="a",
            )

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            UnifiedProblem(
                problem_id="x",
                domain="math",
                difficulty=1,
                source="s",
                question="q",
                canonical_answer="a",
                unknown_field="oops",
            )
