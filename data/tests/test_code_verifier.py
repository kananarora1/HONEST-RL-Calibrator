"""Tests for the code answer verifier."""

import time

import pytest

from data.verifiers.code_verifier import verify_code_answer


# ---------------------------------------------------------------------------
# execute_and_assert (MBPP-style)
# ---------------------------------------------------------------------------


class TestExecuteAndAssert:
    def test_correct_solution_passes(self):
        meta = {
            "verification_type": "execute_and_assert",
            "test_list": [
                "assert add(1, 2) == 3",
                "assert add(-1, 1) == 0",
                "assert add(0, 0) == 0",
            ],
        }
        code = "def add(a, b):\n    return a + b\n"
        assert verify_code_answer(code, meta) is True

    def test_buggy_solution_fails(self):
        meta = {
            "verification_type": "execute_and_assert",
            "test_list": ["assert add(1, 2) == 3"],
        }
        code = "def add(a, b):\n    return a - b\n"  # bug
        assert verify_code_answer(code, meta) is False

    def test_syntax_error_returns_false(self):
        meta = {
            "verification_type": "execute_and_assert",
            "test_list": ["assert add(1, 2) == 3"],
        }
        code = "def add(a, b:\n    return a + b"  # broken syntax
        assert verify_code_answer(code, meta) is False

    def test_runtime_error_returns_false(self):
        meta = {
            "verification_type": "execute_and_assert",
            "test_list": ["assert boom() == 1"],
        }
        code = "def boom():\n    raise RuntimeError('nope')\n"
        assert verify_code_answer(code, meta) is False

    def test_infinite_loop_times_out(self):
        meta = {
            "verification_type": "execute_and_assert",
            "test_list": ["assert spin() == 1"],
        }
        code = "def spin():\n    while True:\n        pass\n"
        start = time.monotonic()
        result = verify_code_answer(code, meta, timeout_seconds=2)
        elapsed = time.monotonic() - start
        assert result is False
        # Must return promptly — the test itself must not hang.
        assert elapsed < 6, f"verifier hung for {elapsed:.1f}s"

    def test_missing_test_list_returns_false(self):
        meta = {"verification_type": "execute_and_assert", "test_list": []}
        code = "def add(a, b):\n    return a + b\n"
        assert verify_code_answer(code, meta) is False

    def test_test_imports_are_executed(self):
        meta = {
            "verification_type": "execute_and_assert",
            "test_imports": ["import math"],
            "test_list": ["assert sqrt2() == math.sqrt(2)"],
        }
        code = "import math\ndef sqrt2():\n    return math.sqrt(2)\n"
        assert verify_code_answer(code, meta) is True


# ---------------------------------------------------------------------------
# stdin_stdout (APPS-style)
# ---------------------------------------------------------------------------


class TestStdinStdout:
    def test_echo_program_passes(self):
        meta = {
            "verification_type": "stdin_stdout",
            "inputs": ["hello\n"],
            "outputs": ["hello\n"],
        }
        code = "import sys\nprint(sys.stdin.read().strip())\n"
        assert verify_code_answer(code, meta) is True

    def test_multiple_cases_all_pass(self):
        meta = {
            "verification_type": "stdin_stdout",
            "inputs": ["3\n4\n", "10\n20\n"],
            "outputs": ["7\n", "30\n"],
        }
        code = (
            "import sys\n"
            "nums = [int(x) for x in sys.stdin.read().split()]\n"
            "print(sum(nums))\n"
        )
        assert verify_code_answer(code, meta) is True

    def test_wrong_output_fails(self):
        meta = {
            "verification_type": "stdin_stdout",
            "inputs": ["3\n4\n"],
            "outputs": ["7\n"],
        }
        code = "import sys\nprint(99)\n"
        assert verify_code_answer(code, meta) is False

    def test_normalizes_trailing_whitespace(self):
        meta = {
            "verification_type": "stdin_stdout",
            "inputs": ["1\n"],
            "outputs": ["42\n\n\n"],  # trailing blank lines should be stripped
        }
        code = "print(42)\n"
        assert verify_code_answer(code, meta) is True

    def test_empty_io_lists_fail(self):
        meta = {
            "verification_type": "stdin_stdout",
            "inputs": [],
            "outputs": [],
        }
        code = "print('anything')\n"
        assert verify_code_answer(code, meta) is False

    def test_mismatched_io_lengths_fail(self):
        meta = {
            "verification_type": "stdin_stdout",
            "inputs": ["1\n", "2\n"],
            "outputs": ["1\n"],  # length mismatch
        }
        code = "import sys\nprint(sys.stdin.read().strip())\n"
        assert verify_code_answer(code, meta) is False


# ---------------------------------------------------------------------------
# Defensive / routing behavior
# ---------------------------------------------------------------------------


class TestDefensive:
    def test_unknown_verification_type_returns_false(self):
        meta = {"verification_type": "nonsense", "test_list": []}
        assert verify_code_answer("print(1)", meta) is False

    def test_non_string_code_returns_false(self):
        meta = {
            "verification_type": "execute_and_assert",
            "test_list": ["assert True"],
        }
        assert verify_code_answer(None, meta) is False  # type: ignore[arg-type]
        assert verify_code_answer(123, meta) is False  # type: ignore[arg-type]
        assert verify_code_answer("", meta) is False

    def test_non_dict_metadata_returns_false(self):
        assert verify_code_answer("print(1)", None) is False  # type: ignore[arg-type]
        assert verify_code_answer("print(1)", "bad") is False  # type: ignore[arg-type]
