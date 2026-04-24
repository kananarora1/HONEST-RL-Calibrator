"""Verifier for math problems.

Exposes :func:`verify_math_answer`, which checks whether a model's string
answer is mathematically equivalent to a canonical answer. Uses SymPy's
LaTeX parser and symbolic simplification; falls back to normalized
string comparison on any parse failure.
"""

from __future__ import annotations

import re
from typing import Optional


_BOXED_RE = re.compile(r"\\boxed\s*\{")
_DOLLAR_RE = re.compile(r"^\$+|\$+$")


def _strip_boxed(s: str) -> str:
    """Strip a surrounding ``\\boxed{...}`` wrapper using brace balancing."""
    m = _BOXED_RE.search(s)
    if not m:
        return s
    start = m.end()
    depth = 1
    i = start
    while i < len(s) and depth > 0:
        ch = s[i]
        if ch == "\\" and i + 1 < len(s):
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                inner = s[start:i]
                # Only strip if \boxed{} wraps (essentially) the whole string.
                rest = s[:m.start()].strip() + s[i + 1:].strip()
                return inner if rest == "" else s
        i += 1
    return s


def _strip_wrappers(s: str) -> str:
    s = s.strip()
    # Strip surrounding $...$ (possibly repeated $$).
    prev = None
    while prev != s:
        prev = s
        s = _DOLLAR_RE.sub("", s).strip()
    s = _strip_boxed(s).strip()
    # Strip \text{...} wrappers and \left/\right spacing that break sympify.
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", " ")
    s = s.replace("\\left", "").replace("\\right", "")
    return s.strip()


def _normalize_string(s: str) -> str:
    return str(s).lower().replace(",", "").replace(" ", "").strip()


def _to_expr(s: str):
    """Best-effort conversion of a string to a SymPy expression.

    Picks the parser by shape: strings containing a backslash are LaTeX
    (``\\frac``, ``\\sqrt``, etc.); everything else is parsed as a Python
    expression via ``sympify``. Using ``parse_latex`` on non-LaTeX input
    misinterprets tokens like ``sqrt`` as ``s*q*r*t``.
    """
    from sympy import sympify

    parsers = []
    if "\\" in s:
        parsers.append("latex")
        parsers.append("sympify")
    else:
        parsers.append("sympify")
        parsers.append("latex")

    for parser in parsers:
        if parser == "latex":
            try:
                from sympy.parsing.latex import parse_latex

                return parse_latex(s)
            except Exception:
                continue
        else:
            try:
                return sympify(s, rational=False)
            except Exception:
                continue
    return None


def verify_math_answer(model_answer: str, canonical_answer: str) -> bool:
    """Return ``True`` iff ``model_answer`` is mathematically equivalent to
    ``canonical_answer``.

    The check order is:

    1. Exact normalized string match (cheap short-circuit).
    2. Symbolic equality via ``simplify(a - b) == 0``.
    3. Normalized string comparison (lowercase, no commas/spaces).

    Any unexpected exception falls through to the string comparison.
    """
    try:
        a_raw = "" if model_answer is None else str(model_answer)
        b_raw = "" if canonical_answer is None else str(canonical_answer)

        a = _strip_wrappers(a_raw)
        b = _strip_wrappers(b_raw)

        if a == b and a != "":
            return True

        ea = _to_expr(a)
        eb = _to_expr(b)
        if ea is not None and eb is not None:
            try:
                from sympy import simplify

                diff = simplify(ea - eb)
                # diff == 0 may return a sympy Boolean; coerce defensively.
                if bool(diff == 0):
                    return True
            except Exception:
                pass

        return _normalize_string(a) == _normalize_string(b) and _normalize_string(a) != ""
    except Exception:
        try:
            return _normalize_string(model_answer) == _normalize_string(canonical_answer)
        except Exception:
            return False
