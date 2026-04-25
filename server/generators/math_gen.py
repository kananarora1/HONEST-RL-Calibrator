import random
from typing import Callable, List, Optional, Tuple


def generate(difficulty: int, seed: Optional[int] = None) -> Tuple[str, str]:
    """Generate a math problem at the given difficulty.

    Returns (question, canonical_answer). The answer is always a normalized
    integer string: no leading zeros, no trailing ".0", no commas or spaces.
    A leading '-' is used for negatives.
    """
    rng = random.Random(seed) if seed is not None else random
    if difficulty == 1:
        return _level_1(rng)
    if difficulty == 2:
        return _level_2(rng)
    if difficulty == 3:
        return _level_3(rng)
    if difficulty == 4:
        return _level_4(rng)
    if difficulty == 5:
        return _level_5(rng)
    raise ValueError(f"difficulty must be in 1..5, got {difficulty}")


def _normalize(n: int) -> str:
    return str(int(n))


def _apply(op: str, a: int, b: int) -> int:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    raise ValueError(f"Unknown operator: {op}")


def _level_1(rng) -> Tuple[str, str]:
    a = rng.randint(1, 9)
    b = rng.randint(1, 9)
    # Only single-digit addition for the absolute easiest level
    return f"{a} + {b}", _normalize(_apply("+", a, b))


def _level_2(rng) -> Tuple[str, str]:
    op = rng.choice(["+", "-"])
    # Small numbers, no multiplication
    a = rng.randint(10, 30)
    b = rng.randint(1, 15)
    return f"{a} {op} {b}", _normalize(_apply(op, a, b))


def _level_3(rng) -> Tuple[str, str]:
    """Compound two-operator expression of the form ``(a op1 b) op2 c``.

    Operators are independently sampled from ``{+, -, *}``. When multiplication
    is involved the relevant operand is kept small so the final integer answer
    stays in a comfortable range (worst case roughly |30 * 12 * 12| = 4320).
    """
    op1 = rng.choice(["+", "-", "*"])
    op2 = rng.choice(["+", "-", "*"])
    a = rng.randint(10, 30)
    b = rng.randint(2, 12) if op1 == "*" else rng.randint(10, 30)
    c = rng.randint(2, 12) if op2 == "*" else rng.randint(2, 20)
    inner = _apply(op1, a, b)
    outer = _apply(op2, inner, c)
    return f"({a} {op1} {b}) {op2} {c}", _normalize(outer)


def _level_4(rng) -> Tuple[str, str]:
    base = rng.randint(2, 12)
    exponent = rng.randint(3, 15)
    modulus = rng.randint(3, 50)
    return f"{base}^{exponent} mod {modulus}", _normalize(pow(base, exponent, modulus))


def _word_store(rng) -> Tuple[str, int]:
    day1 = rng.randint(10, 50)
    multiplier = rng.randint(2, 5)
    q = (
        f"A store sold {day1} apples on Monday and {multiplier} times as many "
        f"on Tuesday. How many apples did the store sell in total?"
    )
    return q, day1 + day1 * multiplier


def _word_travel(rng) -> Tuple[str, int]:
    speed = rng.randint(30, 70)
    hours = rng.randint(2, 8)
    extra = rng.randint(10, 100)
    q = (
        f"A car drives at {speed} miles per hour for {hours} hours, then travels "
        f"another {extra} miles. How many total miles did the car travel?"
    )
    return q, speed * hours + extra


def _word_savings(rng) -> Tuple[str, int]:
    weekly = rng.randint(20, 100)
    weeks = rng.randint(4, 12)
    spent = rng.randint(10, weekly * weeks // 3)
    q = (
        f"Alex saves ${weekly} per week for {weeks} weeks, then spends ${spent}. "
        f"How much money does Alex have left?"
    )
    return q, weekly * weeks - spent


def _word_baskets(rng) -> Tuple[str, int]:
    baskets = rng.randint(3, 8)
    per_basket = rng.randint(4, 12)
    given_away = rng.randint(1, baskets * per_basket // 2)
    q = (
        f"There are {baskets} baskets with {per_basket} apples in each. "
        f"If {given_away} apples are given away, how many apples remain?"
    )
    return q, baskets * per_basket - given_away


def _word_classroom(rng) -> Tuple[str, int]:
    rows = rng.randint(4, 10)
    per_row = rng.randint(3, 8)
    absent = rng.randint(1, rows * per_row // 3)
    q = (
        f"A classroom has {rows} rows of {per_row} desks. On Monday, "
        f"{absent} students were absent. If every other desk was filled, "
        f"how many students were present?"
    )
    return q, rows * per_row - absent


_WORD_TEMPLATES: List[Callable] = [
    _word_store,
    _word_travel,
    _word_savings,
    _word_baskets,
    _word_classroom,
]


def _level_5(rng) -> Tuple[str, str]:
    template = rng.choice(_WORD_TEMPLATES)
    question, answer = template(rng)
    return question, _normalize(answer)
