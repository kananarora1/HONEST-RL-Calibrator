import re

import pytest

from server.generators.math_gen import generate


CANONICAL_ANSWER = re.compile(r"-?\d+")


@pytest.mark.parametrize("difficulty", [1, 2, 3, 4, 5])
def test_returns_valid_tuple(difficulty):
    question, answer = generate(difficulty, seed=42)
    assert isinstance(question, str) and question
    assert isinstance(answer, str) and answer
    assert CANONICAL_ANSWER.fullmatch(answer), f"non-canonical answer: {answer!r}"
    # Canonical form: no leading zeros (except for "0" itself), no spaces, no commas
    assert " " not in answer
    assert "," not in answer
    assert "." not in answer


@pytest.mark.parametrize("difficulty", [1, 2, 3, 4, 5])
def test_seeded_is_reproducible(difficulty):
    first = generate(difficulty, seed=123)
    second = generate(difficulty, seed=123)
    assert first == second


@pytest.mark.parametrize("difficulty", [1, 2, 3, 4, 5])
def test_different_seeds_vary(difficulty):
    outputs = {generate(difficulty, seed=s) for s in range(25)}
    assert len(outputs) > 1, "seeded generation across different seeds should vary"


def test_unseeded_varies():
    outputs = {generate(3) for _ in range(30)}
    assert len(outputs) > 1, "unseeded generation should vary across calls"


def test_invalid_difficulty_raises():
    with pytest.raises(ValueError):
        generate(0)
    with pytest.raises(ValueError):
        generate(6)


def test_level_1_correctness():
    pattern = re.compile(r"(-?\d+) ([+\-]) (-?\d+)")
    for s in range(100):
        q, a = generate(1, seed=s)
        m = pattern.fullmatch(q)
        assert m, f"unexpected level-1 format: {q!r}"
        x, op, y = int(m.group(1)), m.group(2), int(m.group(3))
        expected = x + y if op == "+" else x - y
        assert int(a) == expected, f"{q} → got {a}, expected {expected}"


def test_level_2_correctness():
    pattern = re.compile(r"(-?\d+) ([+\-*]) (-?\d+)")
    for s in range(100):
        q, a = generate(2, seed=s)
        m = pattern.fullmatch(q)
        assert m, f"unexpected level-2 format: {q!r}"
        x, op, y = int(m.group(1)), m.group(2), int(m.group(3))
        if op == "+":
            expected = x + y
        elif op == "-":
            expected = x - y
        else:
            expected = x * y
        assert int(a) == expected, f"{q} → got {a}, expected {expected}"


def test_level_3_correctness():
    pattern = re.compile(r"\((-?\d+) ([+\-*]) (-?\d+)\) ([+\-*]) (-?\d+)")
    for s in range(100):
        q, a = generate(3, seed=s)
        m = pattern.fullmatch(q)
        assert m, f"unexpected level-3 format: {q!r}"
        x, op1, y, op2, z = (
            int(m.group(1)),
            m.group(2),
            int(m.group(3)),
            m.group(4),
            int(m.group(5)),
        )
        inner = {"+": x + y, "-": x - y, "*": x * y}[op1]
        expected = {"+": inner + z, "-": inner - z, "*": inner * z}[op2]
        assert int(a) == expected, f"{q} → got {a}, expected {expected}"


def test_level_4_correctness():
    pattern = re.compile(r"(\d+)\^(\d+) mod (\d+)")
    for s in range(100):
        q, a = generate(4, seed=s)
        m = pattern.fullmatch(q)
        assert m, f"unexpected level-4 format: {q!r}"
        base, exponent, modulus = int(m.group(1)), int(m.group(2)), int(m.group(3))
        assert int(a) == pow(base, exponent, modulus), f"{q} → got {a}"
