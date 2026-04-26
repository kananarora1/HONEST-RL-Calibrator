"""Ingest the Hendrycks MATH dataset and emit unified JSONL records.

Loads every subject config of ``EleutherAI/hendrycks_math`` (7 subjects),
extracts the ``\\boxed{...}`` answer from each solution using a
brace-balanced scanner, and writes one :class:`UnifiedProblem` per line to
``data/processed/math.jsonl``.

Run directly::

    python -m data.ingestion.ingest_hendrycks_math
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

from data.schema import UnifiedProblem


_BOXED_PREFIX = re.compile(r"\\boxed\s*\{")
_LEVEL_RE = re.compile(r"Level\s+([1-5])")

_SUBJECT_CONFIGS: Tuple[str, ...] = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)

_DATASET_CANDIDATES: Tuple[Tuple[str, bool], ...] = (
    # (dataset_name, needs_per_config_load)
    ("hendrycks/competition_math", False),
    ("EleutherAI/hendrycks_math", True),
)


def extract_boxed(text: str) -> Optional[str]:
    """Return the contents of the last ``\\boxed{...}`` in ``text``.

    Uses a brace-balanced scanner (not a pure regex) because LaTeX
    answers like ``\\boxed{\\frac{1}{\\sqrt{2}}}`` contain nested braces
    that standard regex engines cannot match. The prefix is located with
    a regex, then we walk forward counting brace depth to find the
    matching closer.
    """
    if not text:
        return None

    last: Optional[str] = None
    for match in _BOXED_PREFIX.finditer(text):
        start = match.end()  # index just past the opening '{'
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "\\" and i + 1 < len(text):
                # skip escaped char (e.g. \{ or \})
                i += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last = text[start:i]
                    break
            i += 1
    return last


def _level_to_int(level: str) -> Optional[int]:
    if not isinstance(level, str):
        return None
    m = _LEVEL_RE.search(level)
    return int(m.group(1)) if m else None


def _load_dataset_rows() -> Iterator[Tuple[str, dict]]:
    """Yield ``(split, row)`` tuples across all splits we ingest.

    Tries the canonical dataset names in order; the first one that loads
    cleanly wins.
    """
    from datasets import load_dataset  # type: ignore[import-not-found]  # heavy dep

    last_error: Optional[Exception] = None
    for name, per_config in _DATASET_CANDIDATES:
        try:
            if per_config:
                for cfg in _SUBJECT_CONFIGS:
                    ds = load_dataset(name, cfg)
                    for split in ("train", "test"):
                        if split in ds:
                            for row in ds[split]:
                                yield split, row
            else:
                ds = load_dataset(name)
                for split in ("train", "test"):
                    if split in ds:
                        for row in ds[split]:
                            yield split, row
            return
        except Exception as exc:  # noqa: BLE001 — we want to try the next
            last_error = exc
            print(
                f"[ingest_hendrycks_math] {name} not available ({exc}); trying next candidate…",
                file=sys.stderr,
            )
    raise RuntimeError("No Hendrycks MATH dataset could be loaded") from last_error


def _row_to_problem(
    split: str, index: int, row: dict
) -> Optional[UnifiedProblem]:
    problem_text = (row.get("problem") or "").strip()
    solution = row.get("solution") or ""
    subject = (row.get("type") or "unknown").strip()
    level = _level_to_int(row.get("level") or "")
    boxed = extract_boxed(solution)

    if not problem_text or boxed is None or level is None:
        return None

    subject_slug = re.sub(r"[^a-z0-9]+", "_", subject.lower()).strip("_") or "unknown"
    problem_id = f"hendrycks_math_{subject_slug}_{split}_{index:05d}"

    return UnifiedProblem(
        problem_id=problem_id,
        domain="math",
        difficulty=level,
        source="hendrycks_math",
        question=problem_text,
        canonical_answer=boxed,
        verification_metadata={
            "answer_type": "latex",
            "subject": subject,
            "split": split,
        },
        raw_source_entry={
            "problem": problem_text,
            "level": row.get("level"),
            "type": subject,
            "solution": solution,
        },
    )


def ingest(
    rows: Optional[Iterable[Tuple[str, dict]]] = None,
    output_path: Optional[Path] = None,
) -> dict:
    """Run ingestion and return a summary dict.

    ``rows`` is injectable for testing; when omitted, the HF dataset is
    loaded. ``output_path`` defaults to ``data/processed/math.jsonl``.
    """
    repo_root = Path(__file__).resolve().parents[2]
    out = output_path or (repo_root / "data" / "processed" / "math.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    source = rows if rows is not None else _load_dataset_rows()

    n_written = 0
    n_skipped = 0
    per_difficulty: Counter = Counter()
    per_subject: Counter = Counter()
    skip_reasons: Counter = Counter()

    # Separate index per split so IDs stay stable across re-runs.
    split_counters: Counter = Counter()

    with out.open("w", encoding="utf-8") as fh:
        for split, row in source:
            idx = split_counters[split]
            split_counters[split] += 1

            problem = _row_to_problem(split, idx, row)
            if problem is None:
                n_skipped += 1
                if not (row.get("problem") or "").strip():
                    skip_reasons["empty_problem"] += 1
                elif _level_to_int(row.get("level") or "") is None:
                    skip_reasons["unparseable_level"] += 1
                elif extract_boxed(row.get("solution") or "") is None:
                    skip_reasons["no_boxed_answer"] += 1
                else:
                    skip_reasons["other"] += 1
                continue

            fh.write(problem.to_jsonl() + "\n")
            n_written += 1
            per_difficulty[problem.difficulty] += 1
            per_subject[problem.verification_metadata["subject"]] += 1

    summary = {
        "written": n_written,
        "skipped": n_skipped,
        "skip_reasons": dict(skip_reasons),
        "per_difficulty": dict(sorted(per_difficulty.items())),
        "per_subject": dict(sorted(per_subject.items())),
        "output_path": str(out),
    }
    return summary


def _print_summary(summary: dict) -> None:
    print("=" * 60)
    print(f"Wrote: {summary['written']} problems -> {summary['output_path']}")
    print(f"Skipped: {summary['skipped']}  (reasons: {summary['skip_reasons']})")
    print("\nPer difficulty:")
    for lvl, n in summary["per_difficulty"].items():
        print(f"  Level {lvl}: {n}")
    print("\nPer subject:")
    for subj, n in summary["per_subject"].items():
        print(f"  {subj}: {n}")
    print("=" * 60)


def _sanity_check_boxed_regex() -> None:
    cases = [
        (r"\boxed{17}", "17"),
        (r"\boxed{\frac{3}{4}}", r"\frac{3}{4}"),
        (r"\boxed{2\sqrt{3}}", r"2\sqrt{3}"),
        (r"\boxed{\frac{1}{\sqrt{2}}}", r"\frac{1}{\sqrt{2}}"),
    ]
    for src, expected in cases:
        got = extract_boxed(src)
        assert got == expected, f"extract_boxed({src!r}) -> {got!r}, expected {expected!r}"


if __name__ == "__main__":
    _sanity_check_boxed_regex()
    summary = ingest()
    _print_summary(summary)
    # Surface summary as JSON on the last line for easy downstream capture.
    print(json.dumps({k: v for k, v in summary.items() if k != "output_path"}))
