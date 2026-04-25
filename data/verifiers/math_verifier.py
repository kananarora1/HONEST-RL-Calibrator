"""Verifier for math problems.

Exposes :func:`verify_math_answer`, which checks whether a model's string
answer is mathematically equivalent to a canonical answer.

Equivalence strategy (first hit wins):

1. Exact match after stripping ``\\boxed{...}``, ``$...$`` and other
   cosmetic LaTeX wrappers.
2. Symbolic equality via SymPy (``simplify(a - b) == 0``) — used when
   SymPy is importable.
3. Numeric equality via a safe LaTeX-to-Python translation: ``\\frac``,
   ``\\sqrt``, ``\\cdot``, ``\\times``, ``\\pi`` are converted and the
   expression is evaluated in a whitelisted sandbox with ``math.sqrt``.
   This path does not need SymPy and covers the common math-answer
   shapes (fractions, decimals, surds, simple products).
4. Last-resort normalized string comparison.

On any unexpected exception, the function falls through to the string
comparison and never raises.
"""

from __future__ import annotations

import math
import re
from typing import Optional


_BOXED_RE = re.compile(r"\\boxed\s*\{")
_DOLLAR_RE = re.compile(r"^\$+|\$+$")
_FRAC_RE = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
_SQRT_RE = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
_CDOT_RE = re.compile(r"\\cdot|\\times")
_PI_RE = re.compile(r"\\pi")
_SAFE_NUMERIC_RE = re.compile(r"^[\d\s\+\-\*\/\(\)\.]+$")


# ---------------------------------------------------------------------------
# Wrapper stripping
# ---------------------------------------------------------------------------


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
                rest = s[: m.start()].strip() + s[i + 1 :].strip()
                return inner if rest == "" else s
        i += 1
    return s


def _strip_wrappers(s: str) -> str:
    s = s.strip()
    prev = None
    while prev != s:
        prev = s
        s = _DOLLAR_RE.sub("", s).strip()
    s = _strip_boxed(s).strip()
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", " ")
    s = s.replace("\\left", "").replace("\\right", "")
    return s.strip()


def _normalize_string(s) -> str:
    return str(s).lower().replace(",", "").replace(" ", "").strip()


# ---------------------------------------------------------------------------
# SymPy path (optional — only used when sympy is installed)
# ---------------------------------------------------------------------------


def _to_expr_sympy(s: str):
    """Parse ``s`` into a SymPy expression, or return ``None``."""
    try:
        from sympy import sympify  # type: ignore[import-not-found]
    except ImportError:
        return None

    if "\\" in s:
        try:
            from sympy.parsing.latex import parse_latex  # type: ignore[import-not-found]

            return parse_latex(s)
        except Exception:
            return None

    try:
        return sympify(s, rational=False)
    except Exception:
        return None


def _sympy_equal(a: str, b: str) -> Optional[bool]:
    """Return True/False if SymPy can decide, else None (unknown)."""
    ea = _to_expr_sympy(a)
    eb = _to_expr_sympy(b)
    if ea is None or eb is None:
        return None
    try:
        from sympy import simplify  # type: ignore[import-not-found]

        diff = simplify(ea - eb)
        return bool(diff == 0)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Numeric path (no external deps — pure Python + math)
# ---------------------------------------------------------------------------


def _latex_to_python(s: str) -> str:
    """Translate a small subset of LaTeX into a Python expression.

    Handles ``\\frac{a}{b}``, ``\\sqrt{x}``, ``\\cdot`` / ``\\times`` and
    ``\\pi``. Iterated until fixed-point so nested forms like
    ``\\frac{\\sqrt{2}}{2}`` collapse correctly.
    """
    prev = None
    iterations = 0
    while prev != s and iterations < 8:
        prev = s
        s = _FRAC_RE.sub(r"((\1)/(\2))", s)
        s = _SQRT_RE.sub(r"sqrt(\1)", s)
        iterations += 1
    s = _CDOT_RE.sub("*", s)
    s = _PI_RE.sub(repr(math.pi), s)

    # Insert implicit multiplication: "2(" -> "2*(", "2sqrt" -> "2*sqrt",
    # ")(" -> ")*(", ")sqrt" -> ")*sqrt".
    s = re.sub(r"(\d)(\()", r"\1*\2", s)
    s = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", s)
    s = re.sub(r"(\))(\()", r"\1*\2", s)
    s = re.sub(r"(\))([a-zA-Z])", r"\1*\2", s)
    return s


def _to_float(s: str) -> Optional[float]:
    """Best-effort conversion of ``s`` to a ``float``.

    Accepts Python-style arithmetic plus the single whitelisted identifier
    ``sqrt`` (bound to :func:`math.sqrt`). Any other identifier or
    symbol causes a ``None`` return, so the evaluator cannot be used as
    an attack surface.
    """
    if not s:
        return None
    expr = _latex_to_python(s)

    # Whitelist check: after removing the literal token "sqrt", the
    # remainder must be arithmetic characters only. This blocks any
    # attempt to reference names, attributes, dunders, strings, etc.
    residual = expr.replace("sqrt", "")
    if not _SAFE_NUMERIC_RE.match(residual):
        return None
    if not residual.strip():
        return None

    try:
        value = eval(  # noqa: S307 — whitelisted input only
            expr,
            {"__builtins__": {}},
            {"sqrt": math.sqrt},
        )
    except Exception:
        return None

    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _numeric_equal(a: str, b: str, rel_tol: float = 1e-9) -> Optional[bool]:
    """Numeric equality with relative tolerance.

    Returns ``None`` if either side cannot be evaluated numerically, so
    the caller can fall through to another check.
    """
    fa = _to_float(a)
    fb = _to_float(b)
    if fa is None or fb is None:
        return None
    scale = max(1.0, abs(fa), abs(fb))
    return abs(fa - fb) <= rel_tol * scale


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def verify_math_answer(model_answer: str, canonical_answer: str) -> bool:
    """Return ``True`` iff ``model_answer`` is mathematically equivalent
    to ``canonical_answer``. Never raises."""
    try:
        if model_answer is None or canonical_answer is None:
            return False

        a = _strip_wrappers(str(model_answer))
        b = _strip_wrappers(str(canonical_answer))
        if not a or not b:
            return False
        if a == b:
            return True

        sym = _sympy_equal(a, b)
        if sym is True:
            return True

        num = _numeric_equal(a, b)
        if num is True:
            return True
        if num is False:
            return False

        # If SymPy parsed both sides and said they differ, trust it.
        if sym is False:
            return False

        na, nb = _normalize_string(a), _normalize_string(b)
        return na != "" and na == nb
    except Exception:
        try:
            if model_answer is None or canonical_answer is None:
                return False
            na = _normalize_string(model_answer)
            nb = _normalize_string(canonical_answer)
            return na != "" and na == nb
        except Exception:
            return False
