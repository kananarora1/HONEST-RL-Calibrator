"""Verifier for code problems.

Exposes :func:`verify_code_answer`, which executes candidate Python
solutions in an isolated subprocess and returns ``True`` iff every test
case in the provided verification metadata passes.

The verifier supports two verification styles:

* ``execute_and_assert`` — MBPP-style: run the candidate code followed
  by a list of ``assert`` statements; success iff the subprocess exits 0.
* ``stdin_stdout`` — APPS-style: for each input/output pair, run the
  candidate code as a subprocess with the input on stdin and compare the
  (normalized) stdout to the expected output.

Safety notes:

* The model's code is *never* imported, ``exec``'d, or ``eval``'d in the
  parent process — it is always executed in a fresh subprocess via a
  temp file, with a wall-clock timeout.
* On POSIX, a ``preexec_fn`` sets soft RLIMITs on CPU time and address
  space to cap runaway solutions. These are best-effort — the parent
  ``subprocess.run(timeout=...)`` is the authoritative kill switch.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024  # 512 MB
_CPU_LIMIT_SECONDS = 15  # >= subprocess timeout; parent timeout is authoritative.


def _set_child_limits() -> None:  # pragma: no cover — runs in child
    """Best-effort rlimits for child processes on POSIX systems."""
    try:
        import resource

        try:
            resource.setrlimit(
                resource.RLIMIT_CPU, (_CPU_LIMIT_SECONDS, _CPU_LIMIT_SECONDS)
            )
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(
                resource.RLIMIT_AS, (_MEMORY_LIMIT_BYTES, _MEMORY_LIMIT_BYTES)
            )
        except (ValueError, OSError):
            pass
    except Exception:
        pass


def _run_python(
    script_path: Path, stdin: str, timeout_seconds: int
) -> Optional[subprocess.CompletedProcess]:
    """Run ``script_path`` as a fresh Python subprocess.

    Returns the :class:`CompletedProcess` on success, or ``None`` on
    timeout. Any other failure propagates to the caller's try/except.
    """
    preexec = _set_child_limits if os.name == "posix" else None
    try:
        return subprocess.run(
            [sys.executable, "-I", str(script_path)],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            preexec_fn=preexec,
        )
    except subprocess.TimeoutExpired:
        return None


def _normalize_output(s: Any) -> str:
    """Normalize stdout/expected output for comparison.

    APPS sometimes stores outputs as lists (for multi-line expected
    output); coerce to a single string with Unix line endings, trim
    trailing whitespace per line, and strip leading/trailing whitespace.
    """
    if s is None:
        return ""
    if isinstance(s, list):
        text = "\n".join(str(x) for x in s)
    else:
        text = str(s)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    # Drop trailing empty lines for forgiving comparison.
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines).strip()


def _coerce_stdin(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(x) for x in value)
    return str(value)


def _verify_execute_and_assert(
    model_code: str, metadata: Dict[str, Any], timeout_seconds: int
) -> bool:
    tests: List[str] = list(metadata.get("test_list") or [])
    test_imports: List[str] = list(metadata.get("test_imports") or [])
    if not tests:
        return False

    script = "\n".join(test_imports) + "\n" + model_code + "\n\n" + "\n".join(tests) + "\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "candidate.py"
        script_path.write_text(script, encoding="utf-8")
        result = _run_python(script_path, stdin="", timeout_seconds=timeout_seconds)

    if result is None:
        return False
    return result.returncode == 0


def _verify_stdin_stdout(
    model_code: str, metadata: Dict[str, Any], timeout_seconds: int
) -> bool:
    inputs = metadata.get("inputs") or []
    outputs = metadata.get("outputs") or []
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return False
    if not inputs or len(inputs) != len(outputs):
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "candidate.py"
        script_path.write_text(model_code, encoding="utf-8")

        for stdin_value, expected in zip(inputs, outputs):
            result = _run_python(
                script_path,
                stdin=_coerce_stdin(stdin_value),
                timeout_seconds=timeout_seconds,
            )
            if result is None or result.returncode != 0:
                return False
            if _normalize_output(result.stdout) != _normalize_output(expected):
                return False
    return True


def verify_code_answer(
    model_code: str,
    verification_metadata: Dict[str, Any],
    timeout_seconds: int = 5,
) -> bool:
    """Return ``True`` iff ``model_code`` passes every test in the metadata.

    Any exception (syntax errors, missing imports, runtime errors in the
    candidate code, infrastructure failures) is caught and reported as
    ``False`` — this function is designed never to raise.
    """
    try:
        if not isinstance(model_code, str) or not model_code.strip():
            return False
        if not isinstance(verification_metadata, dict):
            return False

        vtype = verification_metadata.get("verification_type")
        if vtype == "execute_and_assert":
            return _verify_execute_and_assert(
                model_code, verification_metadata, timeout_seconds
            )
        if vtype == "stdin_stdout":
            return _verify_stdin_stdout(
                model_code, verification_metadata, timeout_seconds
            )
        return False
    except Exception:
        return False
