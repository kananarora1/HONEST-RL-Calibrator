"""Ingest the MBPP (Mostly Basic Python Problems) dataset into unified JSONL.

Loads ``google-research-datasets/mbpp`` (prefers the ``sanitized`` config
when available), converts each problem into a :class:`UnifiedProblem`,
and writes to ``data/processed/code_mbpp.jsonl``.

Difficulty is set heuristically: problems whose description mentions
"recursion" or whose ``test_list`` has more than three asserts are
labeled 2; the rest are labeled 1.

Run directly::

    python -m data.ingestion.ingest_mbpp
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

from data.schema import UnifiedProblem


_DATASET_NAME = "google-research-datasets/mbpp"


def _difficulty_for(text: str, tests: list) -> int:
    if "recursion" in text.lower() or len(tests) > 3:
        return 2
    return 1


def _build_question(text: str, tests: list) -> str:
    joined_tests = "\n".join(tests) if tests else ""
    return (
        "Write a Python function that satisfies the following description. "
        "Return only the function definition(s); the function name and "
        "signature must match the tests below.\n\n"
        f"Description:\n{text.strip()}\n\n"
        f"Tests:\n{joined_tests}"
    )


def _load_rows() -> Iterator[Tuple[str, dict]]:
    from datasets import (  # type: ignore[import-not-found]
        get_dataset_config_names,
        load_dataset,
    )

    try:
        configs = get_dataset_config_names(_DATASET_NAME)
    except Exception:
        configs = []
    cfg = (
        "sanitized"
        if "sanitized" in configs
        else ("full" if "full" in configs else (configs[0] if configs else None))
    )
    ds = load_dataset(_DATASET_NAME, cfg) if cfg else load_dataset(_DATASET_NAME)
    for split in ds:
        for row in ds[split]:
            yield split, row


def _row_to_problem(
    split: str, index: int, row: dict
) -> Optional[UnifiedProblem]:
    # `sanitized` uses `prompt`; `full` uses `text`.
    text = (row.get("prompt") or row.get("text") or "").strip()
    code = (row.get("code") or "").strip()
    tests = list(row.get("test_list") or [])
    test_imports = list(row.get("test_imports") or [])
    task_id = row.get("task_id")

    if not text or not code or not tests:
        return None

    problem_id = f"mbpp_{split}_{task_id if task_id is not None else index:05d}"

    return UnifiedProblem(
        problem_id=problem_id,
        domain="code",
        difficulty=_difficulty_for(text, tests),
        source="mbpp",
        question=_build_question(text, tests),
        canonical_answer=code,
        verification_metadata={
            "test_list": tests,
            "test_imports": test_imports,
            "verification_type": "execute_and_assert",
            "split": split,
        },
        raw_source_entry={
            "task_id": task_id,
            "text": text,
            "code": code,
            "test_list": tests,
            "test_imports": test_imports,
        },
    )


def ingest(
    rows: Optional[Iterable[Tuple[str, dict]]] = None,
    output_path: Optional[Path] = None,
) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    out = output_path or (repo_root / "data" / "processed" / "code_mbpp.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    source = rows if rows is not None else _load_rows()

    n_written = 0
    n_skipped = 0
    per_difficulty: Counter = Counter()
    per_split: Counter = Counter()
    split_counters: Counter = Counter()

    with out.open("w", encoding="utf-8") as fh:
        for split, row in source:
            idx = split_counters[split]
            split_counters[split] += 1

            problem = _row_to_problem(split, idx, row)
            if problem is None:
                n_skipped += 1
                continue

            fh.write(problem.to_jsonl() + "\n")
            n_written += 1
            per_difficulty[problem.difficulty] += 1
            per_split[split] += 1

    summary = {
        "written": n_written,
        "skipped": n_skipped,
        "per_difficulty": dict(sorted(per_difficulty.items())),
        "per_split": dict(per_split),
        "output_path": str(out),
    }
    return summary


def _print_summary(summary: dict) -> None:
    print("=" * 60)
    print(f"Wrote: {summary['written']} problems -> {summary['output_path']}")
    print(f"Skipped: {summary['skipped']}")
    print(f"Per difficulty: {summary['per_difficulty']}")
    print(f"Per split: {summary['per_split']}")
    print("=" * 60)


if __name__ == "__main__":
    summary = ingest()
    _print_summary(summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "output_path"}))
