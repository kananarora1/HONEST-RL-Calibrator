"""Ingest the APPS competitive-programming dataset into unified JSONL.

Streams ``codeparrot/apps`` (the full dataset is ~10 GB, so streaming is
required) and writes :class:`UnifiedProblem` records to
``data/processed/code_apps.jsonl``. Progress is flushed every 500
records, and the script skips already-written ``problem_id``s on restart
so it can resume after a failure.

Difficulty mapping:

* ``introductory`` -> 3
* ``interview``    -> 4
* ``competition``  -> 5

Run directly::

    python -m data.ingestion.ingest_apps
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

from data.schema import UnifiedProblem


_DATASET_NAME = "codeparrot/apps"
_DIFFICULTY_MAP = {"introductory": 3, "interview": 4, "competition": 5}
_CHECKPOINT_INTERVAL = 500


def _load_rows_streaming() -> Iterator[Tuple[str, dict]]:
    from datasets import load_dataset  # type: ignore[import-not-found]

    for split in ("train", "test"):
        try:
            ds = load_dataset(
                _DATASET_NAME, split=split, streaming=True, trust_remote_code=True
            )
        except Exception as exc:
            print(f"[ingest_apps] failed to open split {split}: {exc}", file=sys.stderr)
            continue
        for row in ds:
            yield split, row


def _normalize_io(raw_io: str) -> Optional[Tuple[list, list]]:
    """Parse APPS's ``input_output`` JSON blob into ``(inputs, outputs)``.

    Returns ``None`` if the blob is missing, empty, or malformed, or if
    the shapes don't line up.
    """
    if not raw_io or not isinstance(raw_io, str):
        return None
    try:
        parsed = json.loads(raw_io)
    except (json.JSONDecodeError, ValueError):
        return None
    inputs = parsed.get("inputs")
    outputs = parsed.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return None
    if not inputs or len(inputs) != len(outputs):
        return None
    return inputs, outputs


def _first_solution(raw_solutions: str) -> Optional[str]:
    if not raw_solutions or not isinstance(raw_solutions, str):
        return None
    try:
        parsed = json.loads(raw_solutions)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    first = parsed[0]
    return first if isinstance(first, str) and first.strip() else None


def _build_question(question: str, starter_code: str) -> str:
    q = (question or "").strip()
    starter = (starter_code or "").strip()
    if starter:
        return (
            f"{q}\n\nUse the following starter code:\n"
            f"```python\n{starter}\n```"
        )
    return q


def _row_to_problem(split: str, row: dict) -> Optional[UnifiedProblem]:
    difficulty_str = (row.get("difficulty") or "").strip().lower()
    difficulty = _DIFFICULTY_MAP.get(difficulty_str)
    if difficulty is None:
        return None

    io = _normalize_io(row.get("input_output") or "")
    if io is None:
        return None
    inputs, outputs = io

    solution = _first_solution(row.get("solutions") or "")
    if solution is None:
        return None

    question_text = _build_question(row.get("question") or "", row.get("starter_code") or "")
    if not question_text:
        return None

    raw_pid = row.get("problem_id")
    try:
        pid_int = int(raw_pid)
        pid_str = f"{pid_int:05d}"
    except (TypeError, ValueError):
        pid_str = str(raw_pid) if raw_pid is not None else "unknown"

    problem_id = f"apps_{split}_{pid_str}"

    return UnifiedProblem(
        problem_id=problem_id,
        domain="code",
        difficulty=difficulty,
        source="apps",
        question=question_text,
        canonical_answer=solution,
        verification_metadata={
            "inputs": inputs,
            "outputs": outputs,
            "verification_type": "stdin_stdout",
            "split": split,
            "apps_difficulty": difficulty_str,
        },
        raw_source_entry={
            "problem_id": raw_pid,
            "difficulty": difficulty_str,
            "url": row.get("url"),
            "starter_code": row.get("starter_code"),
        },
    )


def _load_seen_ids(path: Path) -> set:
    seen: set = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
                pid = rec.get("problem_id")
                if isinstance(pid, str):
                    seen.add(pid)
            except (json.JSONDecodeError, ValueError):
                continue
    return seen


def ingest(
    rows: Optional[Iterable[Tuple[str, dict]]] = None,
    output_path: Optional[Path] = None,
    checkpoint_interval: int = _CHECKPOINT_INTERVAL,
) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    out = output_path or (repo_root / "data" / "processed" / "code_apps.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    seen = _load_seen_ids(out)
    if seen:
        print(
            f"[ingest_apps] resuming: {len(seen)} problems already in {out}",
            file=sys.stderr,
        )

    source = rows if rows is not None else _load_rows_streaming()

    n_written = 0
    n_skipped = 0
    n_resumed_skipped = 0
    per_difficulty: Counter = Counter()
    per_apps_difficulty: Counter = Counter()
    per_split: Counter = Counter()

    mode = "a" if seen else "w"
    with out.open(mode, encoding="utf-8") as fh:
        for split, row in source:
            problem = _row_to_problem(split, row)
            if problem is None:
                n_skipped += 1
                continue
            if problem.problem_id in seen:
                n_resumed_skipped += 1
                continue

            fh.write(problem.to_jsonl() + "\n")
            seen.add(problem.problem_id)
            n_written += 1
            per_difficulty[problem.difficulty] += 1
            per_apps_difficulty[
                problem.verification_metadata["apps_difficulty"]
            ] += 1
            per_split[split] += 1

            if n_written % checkpoint_interval == 0:
                fh.flush()
                print(
                    f"[ingest_apps] checkpoint: {n_written} written "
                    f"(skipped {n_skipped}, resumed-skipped {n_resumed_skipped})",
                    file=sys.stderr,
                )

    summary = {
        "written": n_written,
        "skipped": n_skipped,
        "resumed_skipped": n_resumed_skipped,
        "per_difficulty": dict(sorted(per_difficulty.items())),
        "per_apps_difficulty": dict(per_apps_difficulty),
        "per_split": dict(per_split),
        "output_path": str(out),
    }
    return summary


def _print_summary(summary: dict) -> None:
    print("=" * 60)
    print(f"Wrote: {summary['written']} problems -> {summary['output_path']}")
    print(
        f"Skipped: {summary['skipped']} "
        f"(malformed); Resumed-skipped: {summary['resumed_skipped']}"
    )
    print(f"Per difficulty: {summary['per_difficulty']}")
    print(f"Per APPS difficulty: {summary['per_apps_difficulty']}")
    print(f"Per split: {summary['per_split']}")
    print("=" * 60)


if __name__ == "__main__":
    summary = ingest()
    _print_summary(summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "output_path"}))
