"""Fetch OOD evaluation data from HuggingFace datasets.

Downloads two small slices (~50 Q&A pairs each) from publicly available
HuggingFace datasets:
  - Medical: MMLU professional_medicine subset
  - Legal:   AGIEval LSAT-LR (logical reasoning) subset

Outputs:
  eval/ood/medqa_sample.jsonl   — 50 medical MCQ pairs
  eval/ood/lsat_sample.jsonl    — 50 LSAT logical-reasoning MCQ pairs

Each line: {"question": str, "answer": str, "domain": "medical"|"legal", "source": str}

Usage:
    python eval/ood/fetch_ood_data.py
    python eval/ood/fetch_ood_data.py --n 100 --seed 99
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# MCQ answer index -> letter
_IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def _fetch_mmlu_medicine(n: int, seed: int) -> list[dict]:
    """Pull n questions from MMLU professional_medicine (HF datasets)."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install datasets: pip install datasets")

    print(f"Fetching MMLU professional_medicine ({n} samples)...")
    # validation split has ~149 examples — small enough to load fully
    ds = load_dataset("cais/mmlu", "professional_medicine", split="validation")
    ds = ds.shuffle(seed=seed)

    records = []
    for row in ds:
        if len(records) >= n:
            break
        choices = row.get("choices", [])
        answer_idx = row.get("answer", 0)  # integer index 0-3

        if not choices or answer_idx not in range(len(choices)):
            continue

        # Format as a multiple-choice question string
        options_str = "\n".join(
            f"({_IDX_TO_LETTER.get(i, str(i))}) {c}"
            for i, c in enumerate(choices)
        )
        question = f"{row['question']}\n\nOptions:\n{options_str}"
        answer   = _IDX_TO_LETTER.get(answer_idx, str(answer_idx))

        records.append({
            "question": question,
            "answer":   answer,
            "domain":   "medical",
            "source":   "mmlu/professional_medicine",
        })

    print(f"  -> {len(records)} medical questions fetched.")
    return records


def _fetch_agieval_lsat(n: int, seed: int) -> list[dict]:
    """Pull n questions from AGIEval LSAT-LR (logical reasoning)."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install datasets: pip install datasets")

    print(f"Fetching AGIEval lsat-lr ({n} samples)...")
    try:
        ds = load_dataset("dmayhem93/agieval-lsat-lr", split="test")
    except Exception as e:
        # Fallback: try the main agieval repo
        print(f"  Primary fetch failed ({e}), trying fallback...")
        try:
            ds = load_dataset("hails/agieval-lsat-lr", split="test")
        except Exception as e2:
            print(f"  Fallback also failed ({e2}). Trying MMLU LSAT...")
            return _fetch_mmlu_lsat_fallback(n, seed)

    ds = ds.shuffle(seed=seed)

    records = []
    for row in ds:
        if len(records) >= n:
            break

        # AGIEval format: question, options list, label (letter)
        question_text = row.get("query") or row.get("question", "")
        options       = row.get("choices") or row.get("options", [])
        label         = row.get("gold") or row.get("answer", "")

        if not question_text or not options:
            continue

        # Build readable options
        options_str = "\n".join(
            f"({_IDX_TO_LETTER.get(i, str(i))}) {opt}"
            for i, opt in enumerate(options)
        )
        question_full = f"{question_text}\n\nOptions:\n{options_str}"

        # Normalise label: could be int index or letter string
        if isinstance(label, int):
            label = _IDX_TO_LETTER.get(label, str(label))
        elif isinstance(label, list) and label:
            label = str(label[0])

        records.append({
            "question": question_full,
            "answer":   str(label).strip().upper(),
            "domain":   "legal",
            "source":   "agieval/lsat-lr",
        })

    print(f"  -> {len(records)} legal questions fetched.")
    return records


def _fetch_mmlu_lsat_fallback(n: int, seed: int) -> list[dict]:
    """Fallback: use MMLU law subset if AGIEval is unavailable."""
    from datasets import load_dataset
    print("  Using MMLU professional_law as legal fallback...")
    ds = load_dataset("cais/mmlu", "professional_law", split="test")
    ds = ds.shuffle(seed=seed)

    records = []
    for row in ds:
        if len(records) >= n:
            break
        choices    = row.get("choices", [])
        answer_idx = row.get("answer", 0)
        if not choices or answer_idx not in range(len(choices)):
            continue
        options_str = "\n".join(
            f"({_IDX_TO_LETTER.get(i, str(i))}) {c}"
            for i, c in enumerate(choices)
        )
        question = f"{row['question']}\n\nOptions:\n{options_str}"
        records.append({
            "question": question,
            "answer":   _IDX_TO_LETTER.get(answer_idx, str(answer_idx)),
            "domain":   "legal",
            "source":   "mmlu/professional_law",
        })
    print(f"  -> {len(records)} legal (fallback) questions fetched.")
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Written {len(records)} records -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch OOD evaluation data")
    parser.add_argument("--n",    type=int, default=50, help="Samples per domain")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).parent),
                        help="Output directory for jsonl files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    medical_records = _fetch_mmlu_medicine(args.n, args.seed)
    legal_records   = _fetch_agieval_lsat(args.n, args.seed)

    write_jsonl(medical_records, out_dir / "medqa_sample.jsonl")
    write_jsonl(legal_records,   out_dir / "lsat_sample.jsonl")

    print(f"\nDone. {len(medical_records)} medical + {len(legal_records)} legal questions ready.")
    print("Next step: python eval/full_eval.py --baseline-results eval/baseline_results.json")


if __name__ == "__main__":
    main()
