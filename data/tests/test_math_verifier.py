"""Tests for the math answer verifier."""

import pytest

from data.verifiers.math_verifier import verify_math_answer


class TestExactAndStringMatches:
    def test_identical_integer_strings(self):
        assert verify_math_answer("17", "17") is True

    def test_identical_latex_fractions(self):
        assert verify_math_answer(r"\frac{1}{2}", r"\frac{1}{2}") is True

    def test_whitespace_differences(self):
        assert verify_math_answer("  42 ", "42") is True

    def test_strips_boxed_wrapper(self):
        assert verify_math_answer(r"\boxed{17}", "17") is True

    def test_strips_dollar_wrapper(self):
        assert verify_math_answer("$17$", "17") is True

    def test_strips_both_wrappers(self):
        assert verify_math_answer(r"$\boxed{\frac{1}{2}}$", r"\frac{1}{2}") is True


class TestSymbolicEquivalence:
    def test_fraction_equals_decimal(self):
        assert verify_math_answer("1/2", "0.5") is True

    def test_latex_fraction_equals_decimal(self):
        assert verify_math_answer(r"\frac{1}{2}", "0.5") is True

    def test_surd_latex_vs_sympy(self):
        assert verify_math_answer(r"2\sqrt{3}", "2*sqrt(3)") is True

    def test_integer_vs_float(self):
        assert verify_math_answer("17", "17.0") is True

    def test_negative_integer(self):
        assert verify_math_answer("-3", "-3") is True

    def test_negative_fraction_vs_decimal(self):
        assert verify_math_answer(r"-\frac{1}{4}", "-0.25") is True

    def test_algebraically_equal_products(self):
        assert verify_math_answer("2*3", "6") is True

    def test_latex_sqrt_over_latex_fraction(self):
        assert verify_math_answer(r"\frac{\sqrt{2}}{2}", r"\frac{1}{\sqrt{2}}") is True


class TestNegativeCases:
    def test_different_integers(self):
        assert verify_math_answer("17", "18") is False

    def test_different_fractions(self):
        assert verify_math_answer("1/2", "1/3") is False

    def test_sign_flip(self):
        assert verify_math_answer("3", "-3") is False


class TestMalformedInput:
    @pytest.mark.parametrize(
        "bad, good",
        [
            ("???", "17"),
            ("\\frac{1}{", "0.5"),  # truncated LaTeX
            ("42 elephants", "42"),
            ("", "17"),
            ("17", ""),
        ],
    )
    def test_malformed_returns_false_not_raise(self, bad, good):
        # Should never raise, should return False for these malformed inputs.
        result = verify_math_answer(bad, good)
        assert result is False

    def test_none_inputs_do_not_crash(self):
        # Pydantic would never pass None, but the verifier must be defensive.
        assert verify_math_answer(None, None) is False  # type: ignore[arg-type]
        assert verify_math_answer(None, "17") is False  # type: ignore[arg-type]
        assert verify_math_answer("17", None) is False  # type: ignore[arg-type]

    def test_non_string_numeric_inputs(self):
        # Defensive: should coerce to str before comparing.
        assert verify_math_answer(17, "17") is True  # type: ignore[arg-type]
