"""Tests for server/reward.py and server/verifier.py."""

import pytest

from server.verifier import verify_answer
from server.reward import (
    ABSTAIN_PENALTY,
    FORMAT_BONUS,
    MALFORMED_PENALTY,
    compute_reward,
    parse_action,
)


# Brier scaling constant — kept in sync with server/reward.py
_BRIER_SCALE = -1.5


def _wf(answer: str, conf: str) -> str:
    """Well-formed completion (reasoning required by parse_action)."""
    return f"<reasoning>think</reasoning><answer>{answer}</answer><confidence>{conf}</confidence>"


# ===========================================================================
# verify_answer
# ===========================================================================


class TestVerifyAnswer:
    def test_exact_match(self):
        assert verify_answer("42", "42") is True

    def test_float_vs_int(self):
        assert verify_answer("42", "42.0") is True
        assert verify_answer("42.0", "42") is True

    def test_trailing_whitespace(self):
        assert verify_answer("42 ", "42") is True
        assert verify_answer(" 42", "42 ") is True

    def test_comma_removal(self):
        assert verify_answer("42,000", "42000") is True
        assert verify_answer("1,234,567", "1234567") is True

    def test_case_insensitive(self):
        assert verify_answer("Alice", "alice") is True
        assert verify_answer("RED", "red") is True

    def test_mismatch(self):
        assert verify_answer("41", "42") is False
        assert verify_answer("alice", "bob") is False

    def test_float_precision(self):
        # 3.14 != 3 → should not match
        assert verify_answer("3.14", "3") is False

    def test_combo(self):
        # whitespace + comma + float-int
        assert verify_answer(" 1,000.0 ", "1000") is True

    def test_code_domain_falls_back_without_metadata(self):
        assert verify_answer("11", "11", domain="code") is True

    def test_logic_domain_falls_back_without_metadata(self):
        assert verify_answer("alice", "Alice", domain="logic") is True


# ===========================================================================
# parse_action
# ===========================================================================


class TestParseAction:
    def test_well_formed_answer(self):
        result = parse_action(_wf("42", "0.9"))
        assert result["type"] == "answer"
        assert result["answer"] == "42"
        assert result["confidence"] == pytest.approx(0.9)

    def test_abstain(self):
        raw = "<abstain/>"
        result = parse_action(raw)
        assert result["type"] == "abstain"

    def test_abstain_with_space(self):
        raw = "<abstain />"
        result = parse_action(raw)
        assert result["type"] == "abstain"

    def test_malformed_no_tags(self):
        assert parse_action("The answer is 42") == {"type": "malformed"}

    def test_malformed_missing_confidence(self):
        assert parse_action("<reasoning>x</reasoning><answer>42</answer>") == {
            "type": "malformed"
        }

    def test_malformed_bad_confidence(self):
        raw = "<reasoning>x</reasoning><answer>42</answer><confidence>high</confidence>"
        assert parse_action(raw) == {"type": "malformed"}

    def test_confidence_clamp_above_one(self):
        result = parse_action(_wf("42", "1.5"))
        assert result["confidence"] == pytest.approx(1.0)

    def test_confidence_clamp_below_zero(self):
        result = parse_action(_wf("42", "-0.3"))
        assert result["confidence"] == pytest.approx(0.0)

    def test_confidence_boundary_zero(self):
        result = parse_action(_wf("42", "0.0"))
        assert result["confidence"] == pytest.approx(0.0)

    def test_confidence_boundary_one(self):
        result = parse_action(_wf("42", "1.0"))
        assert result["confidence"] == pytest.approx(1.0)

    def test_answer_with_surrounding_text(self):
        raw = (
            "I think <reasoning>x</reasoning><answer>Paris</answer>"
            "<confidence>0.8</confidence> is correct."
        )
        result = parse_action(raw)
        assert result["type"] == "answer"
        assert result["answer"] == "Paris"

    def test_empty_string(self):
        assert parse_action("") == {"type": "malformed"}


# ===========================================================================
# compute_reward
# ===========================================================================


class TestComputeReward:
    # --- malformed ---

    def test_malformed_reward(self):
        r, c = compute_reward({"type": "malformed"}, "42", 3)
        assert r == pytest.approx(MALFORMED_PENALTY)
        assert c is None

    # --- abstain ---

    def test_abstain_low_difficulty(self):
        r, c = compute_reward({"type": "abstain"}, "42", 3)
        assert r == pytest.approx(ABSTAIN_PENALTY)
        assert c is None

    def test_abstain_high_difficulty(self):
        r, c = compute_reward({"type": "abstain"}, "42", 7)
        assert r == pytest.approx(0.0)
        assert c is None

    def test_abstain_boundary_difficulty_6(self):
        r, c = compute_reward({"type": "abstain"}, "42", 6)
        assert r == pytest.approx(ABSTAIN_PENALTY)
        assert c is None

    def test_abstain_boundary_difficulty_7(self):
        r, c = compute_reward({"type": "abstain"}, "42", 7)
        assert r == pytest.approx(0.0)
        assert c is None

    # --- answer: correct ---

    def test_correct_high_confidence(self):
        """Perfect calibration: brier=0 + format bonus."""
        parsed = {"type": "answer", "answer": "42", "confidence": 1.0}
        r, c = compute_reward(parsed, "42", 3)
        assert c is True
        # brier = _BRIER_SCALE * ((1.0 - 1.0)**2) = 0.0; + FORMAT_BONUS
        assert r == pytest.approx(FORMAT_BONUS)

    def test_correct_medium_confidence(self):
        parsed = {"type": "answer", "answer": "42", "confidence": 0.9}
        r, c = compute_reward(parsed, "42", 3)
        assert c is True
        expected = _BRIER_SCALE * ((0.9 - 1.0) ** 2) + FORMAT_BONUS
        assert r == pytest.approx(expected)

    def test_perfect_calibration_is_max_reward(self):
        """Confidence=1.0 on correct answer should yield the highest possible reward."""
        parsed_perfect = {"type": "answer", "answer": "42", "confidence": 1.0}
        parsed_lower = {"type": "answer", "answer": "42", "confidence": 0.7}
        r_perfect, _ = compute_reward(parsed_perfect, "42", 3)
        r_lower, _ = compute_reward(parsed_lower, "42", 3)
        assert r_perfect > r_lower

    # --- answer: wrong ---

    def test_wrong_high_confidence_strong_penalty(self):
        """Wrong answer with high confidence: strong Brier penalty."""
        parsed = {"type": "answer", "answer": "41", "confidence": 0.9}
        r, c = compute_reward(parsed, "42", 3)
        assert c is False
        expected = _BRIER_SCALE * ((0.9 - 0.0) ** 2) + FORMAT_BONUS
        assert r == pytest.approx(expected)
        assert r < -0.3   # definitely a strong penalty

    def test_wrong_low_confidence_calibrated(self):
        """Wrong answer but honest low confidence: mild penalty (well-calibrated)."""
        parsed = {"type": "answer", "answer": "41", "confidence": 0.1}
        r, c = compute_reward(parsed, "42", 3)
        assert c is False
        expected = _BRIER_SCALE * ((0.1 - 0.0) ** 2) + FORMAT_BONUS
        assert r == pytest.approx(expected)
        assert r > -0.1   # much less painful than overconfident wrong

    def test_wrong_with_confidence_zero(self):
        """Confidence=0 on wrong answer: format bonus only."""
        parsed = {"type": "answer", "answer": "41", "confidence": 0.0}
        r, c = compute_reward(parsed, "42", 3)
        assert c is False
        # brier = _BRIER_SCALE * ((0 - 0)**2) = 0; + FORMAT_BONUS
        assert r == pytest.approx(FORMAT_BONUS)

    # --- normalization integration ---

    def test_numeric_normalization_in_reward(self):
        parsed = {"type": "answer", "answer": "42.0", "confidence": 1.0}
        r, c = compute_reward(parsed, "42", 3)
        assert c is True

    def test_comma_normalization_in_reward(self):
        parsed = {"type": "answer", "answer": "42,000", "confidence": 1.0}
        r, c = compute_reward(parsed, "42000", 3)
        assert c is True

    # --- smoke-test values from spec ---

    def test_smoke_correct_high_conf(self):
        p = parse_action(_wf("42", "0.9"))
        r, c = compute_reward(p, "42", 3)
        assert c is True
        # brier = scale * 0.01 + format bonus → small positive
        assert r > -0.1

    def test_smoke_wrong_high_conf(self):
        p = parse_action(_wf("41", "0.9"))
        r, c = compute_reward(p, "42", 3)
        assert c is False
        assert r == pytest.approx(_BRIER_SCALE * ((0.9) ** 2) + FORMAT_BONUS)

    def test_smoke_wrong_low_conf(self):
        p = parse_action(_wf("41", "0.1"))
        r, c = compute_reward(p, "42", 3)
        assert c is False
        assert r == pytest.approx(_BRIER_SCALE * ((0.1) ** 2) + FORMAT_BONUS)
