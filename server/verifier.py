"""Answer verification with normalization."""

import math


def _normalize(s: str) -> str:
    """Normalize a string for comparison.

    Steps:
    1. Strip surrounding whitespace
    2. Lowercase
    3. Remove commas (handles "42,000" -> "42000")
    4. If the result parses as a float whose fractional part is zero,
       reduce to a plain integer string (handles "42.0" -> "42").
    """
    s = s.strip().lower().replace(",", "")
    try:
        f = float(s)
        # Guard: inf/nan strings parse as valid floats but are not real answers
        if not math.isfinite(f):
            return s  # keep as-is — will fail comparison
        # Guard: very large floats overflow int()
        if abs(f) > 1e15:
            return f"{f:g}"
        if f == int(f):
            return str(int(f))
        # Keep as float string but strip trailing zeros for consistency
        return f"{f:g}"
    except ValueError:
        pass
    return s


def verify_answer(agent_answer: str, ground_truth: str) -> bool:
    """Return True if agent_answer matches ground_truth after normalization."""
    # Reject empty answers — they should never reach here, but be safe
    if not agent_answer or not agent_answer.strip():
        return False
    try:
        return _normalize(agent_answer) == _normalize(ground_truth)
    except (OverflowError, ValueError):
        return False
