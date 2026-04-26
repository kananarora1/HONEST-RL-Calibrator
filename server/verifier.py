"""Answer verification with domain-specific routing."""

import math
from typing import Any, Dict, Optional

from data.verifiers.code_verifier import verify_code_answer
from data.verifiers.logic_verifier import verify_logic_answer
from data.verifiers.math_verifier import verify_math_answer


def verify_answer(
    agent_answer: str,
    ground_truth: str,
    domain: Optional[str] = None,
    verification_metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Return True if agent_answer matches ground_truth, using domain-specific rules."""
    if not agent_answer or not agent_answer.strip():
        return False
        
    verification_metadata = verification_metadata or {}

    try:
        if domain == "math":
            return verify_math_answer(agent_answer, ground_truth)
        elif domain == "code":
            # Environment-style code tasks use scalar/string answers, while
            # dataset-style code tasks provide executable verification metadata.
            if not verification_metadata:
                return _normalize(agent_answer) == _normalize(ground_truth)
            return verify_code_answer(agent_answer, verification_metadata)
        elif domain == "logic":
            # Environment-style logic tasks are often direct strings (e.g. a name/color),
            # whereas dataset-style tasks use structured canonical JSON solutions.
            if not verification_metadata:
                return _normalize(agent_answer) == _normalize(ground_truth)
            passed, _acc = verify_logic_answer(agent_answer, ground_truth, verification_metadata)
            return passed
        else:
            # Fallback to simple normalization if domain is unknown or None
            return _normalize(agent_answer) == _normalize(ground_truth)
    except Exception:
        # Defensive fallback
        try:
            return _normalize(agent_answer) == _normalize(ground_truth)
        except Exception:
            return False


def _normalize(s: str) -> str:
    """Normalize a string for fallback comparison."""
    s = s.strip().lower().replace(",", "")
    try:
        f = float(s)
        if not math.isfinite(f):
            return s
        if abs(f) > 1e15:
            return f"{f:g}"
        if f == int(f):
            return str(int(f))
        return f"{f:g}"
    except ValueError:
        pass
    return s


# ---------------------------------------------------------------------------
# MCQ-aware verification (OOD only)
#
# The core `verify_answer` above is used by the training reward pipeline and
# must stay untouched. For OOD MCQ datasets (MMLU / AGIEval) the ground-truth
# label can be stored either as a letter ("A"-"E") or as a string index
# ("0"-"4"), while the model may emit either convention as well. The helpers
# below canonicalize both sides to the same index space so that a correct
# pick is not flagged wrong purely due to representation.
# ---------------------------------------------------------------------------

_LETTER_TO_IDX = {"A": "0", "B": "1", "C": "2", "D": "3", "E": "4"}


def _canonicalize_mcq(s: Optional[str]) -> str:
    """Best-effort canonicalization of an MCQ token to a string index.

    Examples:
        "A"   -> "0"      "(B)" -> "1"    "C." -> "2"
        "0"   -> "0"      "3"   -> "3"
        "option D is correct" -> "3"  (takes first A-E letter token)
        "banana" -> "banana" (fallback: lower-cased, unchanged)
    """
    if s is None:
        return ""
    raw = str(s).strip()
    if not raw:
        return ""

    token = raw.strip().strip("()[]{}.,:;\"' \t").upper()
    head = token.split()[0] if token.split() else token
    head = head.strip("()[]{}.,:;\"' \t")

    if head in _LETTER_TO_IDX:
        return _LETTER_TO_IDX[head]
    if head.isdigit() and len(head) <= 2:
        return head

    for ch in token:
        if ch in _LETTER_TO_IDX:
            return _LETTER_TO_IDX[ch]

    return raw.strip().lower()


def verify_mcq(agent_answer: Optional[str], ground_truth: Optional[str]) -> bool:
    """Return True iff the two MCQ tokens canonicalize to the same choice.

    Safe for OOD where ground-truth may be "A"-"E" or "0"-"4". Never raises;
    returns False on any unparseable input.
    """
    try:
        return (
            _canonicalize_mcq(agent_answer) != ""
            and _canonicalize_mcq(agent_answer) == _canonicalize_mcq(ground_truth)
        )
    except Exception:
        return False