"""Ingest the APPS competitive-programming dataset into unified JSONL.

The Hugging Face dataset card ``codeparrot/apps`` still ships a Python
loading script (``apps.py``). Recent ``datasets`` versions reject those
scripts, so we **stream the published JSONL shards directly** from the Hub
(``train.jsonl`` / ``test.jsonl``) via HTTPS — same rows, no
``trust_remote_code`` / no ``load_dataset("codeparrot/apps", ...)``.

The full train split is large (~10 GB); ingestion streams line-by-line.

Each JSONL row uses the numeric field ``id`` (not ``problem_id``) as the
stable problem key. Rows whose ``input_output`` lists are both empty are
skipped (they are not stdin/stdout verifiable with this pipeline).

If a previous run wrote ``apps_*_unknown`` ids, delete ``code_apps.jsonl``
and re-ingest.

Difficulty mapping:

* ``introductory`` -> 3
* ``interview``    -> 4
* ``competition``  -> 5

Run directly::

    python -m data.ingestion.ingest_apps
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Tuple

from data.schema import UnifiedProblem


_HUB_JSONL_BASE = (
    "https://huggingface.co/datasets/codeparrot/apps/resolve/main"
)
_DIFFICULTY_MAP = {"introductory": 3, "interview": 4, "competition": 5}
_CHECKPOINT_INTERVAL = 500


def _open_apps_jsonl(split: str):
    """Return a binary HTTP response for ``{split}.jsonl`` (caller must close)."""
    url = f"{_HUB_JSONL_BASE}/{split}.jsonl"
    headers = {"User-Agent": "HONEST-RL-Calibrator-ingest/1.0 (APPS JSONL stream)"}
    token = (os.environ.get("HF_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=600)


def _load_rows_streaming() -> Iterator[Tuple[str, dict]]:
    for split in ("train", "test"):
        try:
            with _open_apps_jsonl(split) as resp:
                while True:
                    raw = resp.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        yield split, row
        except urllib.error.HTTPError as exc:
            print(
                f"[ingest_apps] HTTP {exc.code} opening {split}.jsonl: {exc.reason}",
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"[ingest_apps] failed to stream split {split}: {exc}", file=sys.stderr)


def _parse_input_output_blob(raw_io: Any) -> Optional[dict]:
    """Return a dict with ``inputs`` / ``outputs`` lists, or ``None``."""
    if raw_io is None:
        return None
    if isinstance(raw_io, dict):
        return raw_io
    if isinstance(raw_io, str):
        if not raw_io.strip():
            return None
        try:
            parsed = json.loads(raw_io)
        except (json.JSONDecodeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _resolve_io_pairs(blob: dict) -> Optional[Tuple[List[Any], List[Any]]]:
    """Return ``(inputs, outputs)`` for stdin_stdout verification, or ``None``.

    Rows with both lists empty (many APPS ``interview`` / ``competition``
    generator tasks) are skipped: they would require a custom checker or an
    expensive reference run at ingest time.
    """
    inputs = blob.get("inputs")
    outputs = blob.get("outputs")
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

    solution = _first_solution(row.get("solutions") or "")
    if solution is None:
        return None

    blob = _parse_input_output_blob(row.get("input_output"))
    if blob is None:
        return None
    io_pair = _resolve_io_pairs(blob)
    if io_pair is None:
        return None
    inputs, outputs = io_pair

    question_text = _build_question(row.get("question") or "", row.get("starter_code") or "")
    if not question_text:
        return None

    # Hub JSONL uses ``id``; older HF dataset dicts used ``problem_id``.
    raw_pid = row.get("problem_id")
    if raw_pid is None:
        raw_pid = row.get("id")
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
    bad_unknown = sum(1 for pid in seen if str(pid).endswith("_unknown"))
    if bad_unknown:
        print(
            f"[ingest_apps] WARNING: {bad_unknown} stale id(s) ending in '_unknown' "
            f"(Hub JSONL uses `id`, not `problem_id`).\n"
            f"  Fix: rm -f {out} && PYTHONPATH=. python -m data.ingestion.ingest_apps",
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
