"""Regenerate ZebraLogic-style constraint-satisfaction puzzles.

Generates fresh zebra-logic puzzles using a Z3-based approach to avoid
contamination from the published ZebraLogic benchmark (WildEval/ZeroEval).

Algorithm
---------
1. Pick a grid size (N houses × M features) and sample a random ground-truth
   solution (a bijection from houses to feature values for every feature).
2. Enumerate a rich set of candidate clues (Found_At, Left_Of, Right_Of,
   Side_By_Side, Not_At) that are *true* of the solution.
3. Greedily remove clues (in random order) while checking via Z3 that the
   remaining set still uniquely determines the solution.
4. Format the minimal clue set as a natural-language puzzle.

Difficulty mapping (matches ZeroEval's log-search-space thresholds)
-------------------------------------------------------------------
* Difficulty 3: 3×3 and 3×4  (target 500 total)
* Difficulty 4: 4×4 and 4×5  (target 500 total)
* Difficulty 5: 5×5 and 6×6  (target 300 total — slow to generate)

Run directly::

    python -m data.ingestion.regenerate_zebralogic

Source attribution: puzzle generation approach inspired by
  WildEval/ZeroEval (https://github.com/WildEval/ZeroEval, Apache-2.0).
  No code was copied; the Z3 uniqueness-check and clue vocabulary are
  original implementations.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from data.schema import UnifiedProblem


# ---------------------------------------------------------------------------
# Attribute vocabulary
# ---------------------------------------------------------------------------

FEATURE_POOLS: Dict[str, List[str]] = {
    "Name":       ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"],
    "Pet":        ["cat", "dog", "fish", "bird", "rabbit", "hamster"],
    "Drink":      ["tea", "coffee", "milk", "juice", "water", "soda"],
    "Color":      ["red", "blue", "green", "yellow", "white", "purple"],
    "Job":        ["doctor", "teacher", "engineer", "artist", "chef", "lawyer"],
    "Sport":      ["soccer", "tennis", "swimming", "cycling", "chess", "golf"],
    "Music":      ["jazz", "rock", "pop", "classical", "blues", "country"],
    "Transport":  ["car", "bike", "train", "bus", "plane", "boat"],
}

FEATURE_ORDER = list(FEATURE_POOLS.keys())


# ---------------------------------------------------------------------------
# Solution generation
# ---------------------------------------------------------------------------

def _sample_solution(n_houses: int, features: List[str]) -> Dict[int, Dict[str, str]]:
    """Return a random ground-truth assignment.

    Returns a dict: house_index (1-indexed) → {feature: value}.
    """
    solution: Dict[int, Dict[str, str]] = {h: {} for h in range(1, n_houses + 1)}
    for feat in features:
        pool = FEATURE_POOLS[feat]
        values = random.sample(pool, n_houses)
        for h, val in zip(range(1, n_houses + 1), values):
            solution[h][feat] = val
    return solution


# ---------------------------------------------------------------------------
# Clue generation
# ---------------------------------------------------------------------------

def _enumerate_clues(
    solution: Dict[int, Dict[str, str]], features: List[str]
) -> List[Tuple[str, ...]]:
    """Enumerate all clues that are *true* of the solution."""
    n = len(solution)
    clues: List[Tuple[str, ...]] = []

    # Build reverse maps: (feat, val) → house
    pos: Dict[Tuple[str, str], int] = {}
    for h, attrs in solution.items():
        for feat, val in attrs.items():
            pos[(feat, val)] = h

    # Found_At: (feat, val) is at house h
    for (feat, val), h in pos.items():
        clues.append(("Found_At", feat, val, str(h)))

    # Left_Of / Right_Of: (feat1, val1) is immediately left/right of (feat2, val2)
    for (f1, v1), h1 in pos.items():
        for (f2, v2), h2 in pos.items():
            if f1 == f2:
                continue
            if h2 == h1 + 1:
                clues.append(("Left_Of", f1, v1, f2, v2))
            if h2 == h1 - 1:
                clues.append(("Right_Of", f1, v1, f2, v2))

    # Side_By_Side: (feat1, val1) and (feat2, val2) are neighbours (|h1-h2|==1)
    for (f1, v1), h1 in pos.items():
        for (f2, v2), h2 in pos.items():
            if f1 >= f2:  # avoid duplicates
                continue
            if abs(h1 - h2) == 1:
                clues.append(("Side_By_Side", f1, v1, f2, v2))

    # Not_At: (feat, val) is NOT at house h  — only generate a few
    # (these add information density without exploding the clue set)
    for (feat, val), h_true in pos.items():
        for h_wrong in range(1, n + 1):
            if h_wrong != h_true:
                clues.append(("Not_At", feat, val, str(h_wrong)))

    return clues


# ---------------------------------------------------------------------------
# Z3 uniqueness check
# ---------------------------------------------------------------------------

def _build_z3_solver(
    n_houses: int, features: List[str], feature_values: Dict[str, List[str]]
):
    """Return a fresh Z3 solver with global uniqueness constraints."""
    import z3  # type: ignore[import-not-found]

    # Variable x[(h, feat, val)] ∈ {0,1}: 1 iff house h has feature=val
    x: Dict[Tuple, Any] = {}
    for h in range(1, n_houses + 1):
        for feat in features:
            for val in feature_values[feat]:
                x[(h, feat, val)] = z3.Bool(f"x_{h}_{feat}_{val}")

    s = z3.Solver()

    # Each house has exactly one value per feature
    for h in range(1, n_houses + 1):
        for feat in features:
            vals = feature_values[feat]
            s.add(z3.PbEq([(x[(h, feat, v)], 1) for v in vals], 1))

    # Each value appears in exactly one house per feature
    for feat in features:
        for val in feature_values[feat]:
            s.add(z3.PbEq([(x[(h, feat, val)], 1) for h in range(1, n_houses + 1)], 1))

    return s, x


def _clue_to_z3(
    clue: Tuple, n_houses: int, x: Dict
):
    """Convert a clue tuple to a Z3 expression."""
    import z3  # type: ignore[import-not-found]

    kind = clue[0]
    if kind == "Found_At":
        _, feat, val, h_str = clue
        h = int(h_str)
        return x[(h, feat, val)]

    if kind == "Left_Of":
        _, f1, v1, f2, v2 = clue
        # ∃h: x[h,f1,v1] ∧ x[h+1,f2,v2]
        terms = []
        for h in range(1, n_houses):
            terms.append(z3.And(x[(h, f1, v1)], x[(h + 1, f2, v2)]))
        return z3.Or(*terms) if terms else z3.BoolVal(False)

    if kind == "Right_Of":
        _, f1, v1, f2, v2 = clue
        terms = []
        for h in range(2, n_houses + 1):
            terms.append(z3.And(x[(h, f1, v1)], x[(h - 1, f2, v2)]))
        return z3.Or(*terms) if terms else z3.BoolVal(False)

    if kind == "Side_By_Side":
        _, f1, v1, f2, v2 = clue
        terms = []
        for h in range(1, n_houses):
            terms.append(z3.And(x[(h, f1, v1)], x[(h + 1, f2, v2)]))
            terms.append(z3.And(x[(h + 1, f1, v1)], x[(h, f2, v2)]))
        return z3.Or(*terms) if terms else z3.BoolVal(False)

    if kind == "Not_At":
        _, feat, val, h_str = clue
        h = int(h_str)
        return z3.Not(x[(h, feat, val)])

    raise ValueError(f"Unknown clue kind: {kind}")


def _is_unique(
    clues: List[Tuple],
    solution: Dict[int, Dict[str, str]],
    n_houses: int,
    features: List[str],
    feature_values: Dict[str, List[str]],
    timeout_ms: int = 5000,
) -> bool:
    """Return True iff clues uniquely determine the solution (no other solution exists)."""
    import z3  # type: ignore[import-not-found]

    s, x = _build_z3_solver(n_houses, features, feature_values)

    # Add all clues as constraints
    for clue in clues:
        z3expr = _clue_to_z3(clue, n_houses, x)
        s.add(z3expr)

    # Encode "not the known solution" to search for a second solution
    not_solution_terms = []
    for h, attrs in solution.items():
        for feat, val in attrs.items():
            not_solution_terms.append(z3.Not(x[(h, feat, val)]))

    s.set("timeout", timeout_ms)
    result = s.check(z3.Or(*not_solution_terms))
    return result == z3.unsat  # unsat → no other solution → unique


# ---------------------------------------------------------------------------
# Clue minimization (greedy)
# ---------------------------------------------------------------------------

def _minimize_clues(
    all_clues: List[Tuple],
    solution: Dict[int, Dict[str, str]],
    n_houses: int,
    features: List[str],
    feature_values: Dict[str, List[str]],
) -> List[Tuple]:
    """Greedily remove clues while maintaining uniqueness."""
    clues = list(all_clues)
    random.shuffle(clues)

    i = 0
    while i < len(clues):
        candidate = clues[:i] + clues[i + 1:]
        if _is_unique(candidate, solution, n_houses, features, feature_values):
            clues = candidate  # drop clue at i (don't increment i)
        else:
            i += 1

    return clues


# ---------------------------------------------------------------------------
# Natural-language formatting
# ---------------------------------------------------------------------------

def _clue_to_text(clue: Tuple) -> str:
    kind = clue[0]
    if kind == "Found_At":
        _, feat, val, h = clue
        return f"The person with {feat} '{val}' lives in House {h}."
    if kind == "Left_Of":
        _, f1, v1, f2, v2 = clue
        return (f"The person with {f1} '{v1}' lives "
                f"immediately to the left of the person with {f2} '{v2}'.")
    if kind == "Right_Of":
        _, f1, v1, f2, v2 = clue
        return (f"The person with {f1} '{v1}' lives "
                f"immediately to the right of the person with {f2} '{v2}'.")
    if kind == "Side_By_Side":
        _, f1, v1, f2, v2 = clue
        return (f"The person with {f1} '{v1}' lives "
                f"next to the person with {f2} '{v2}'.")
    if kind == "Not_At":
        _, feat, val, h = clue
        return f"The person with {feat} '{val}' does NOT live in House {h}."
    return str(clue)


def _format_question(
    n_houses: int,
    features: List[str],
    feature_values: Dict[str, List[str]],
    clues: List[Tuple],
) -> str:
    """Build the full natural-language puzzle prompt."""
    house_range = f"Houses 1 through {n_houses}"
    feature_list = ", ".join(features)

    lines = [
        f"There are {n_houses} houses in a row, numbered 1 to {n_houses}.",
        f"Each house has a unique value for each of the following attributes: {feature_list}.",
        f"The possible values are:",
    ]
    for feat in features:
        lines.append(f"  {feat}: {', '.join(feature_values[feat])}")

    lines.append("")
    lines.append("Using the following clues, determine the unique assignment:")
    lines.append("")
    for i, clue in enumerate(clues, 1):
        lines.append(f"  {i}. {_clue_to_text(clue)}")

    lines.append("")
    lines.append(
        'Output your answer as a JSON object mapping each house to its attributes, like:\n'
        '{"House 1": {"Name": "Alice", "Pet": "cat", ...}, "House 2": {...}, ...}'
    )
    return "\n".join(lines)


def _format_canonical_answer(
    solution: Dict[int, Dict[str, str]]
) -> Dict[str, Dict[str, str]]:
    return {f"House {h}": attrs for h, attrs in sorted(solution.items())}


# ---------------------------------------------------------------------------
# Puzzle generation
# ---------------------------------------------------------------------------

def _try_generate_puzzle(
    n_houses: int,
    features: List[str],
    rng: random.Random,
) -> Optional[Tuple[Dict[int, Dict[str, str]], List[Tuple], Dict[str, List[str]]]]:
    """Attempt to generate a minimal-clue puzzle.  Returns None on failure."""
    import z3  # noqa: F401 — ensure z3 importable early

    feature_values = {
        feat: rng.sample(FEATURE_POOLS[feat], n_houses) for feat in features
    }
    solution = _sample_solution(n_houses, features)
    # Fix feature_values to match solution
    for feat in features:
        feature_values[feat] = [solution[h][feat] for h in range(1, n_houses + 1)]

    all_clues = _enumerate_clues(solution, features)

    # Quick sanity: full clue set should be unique
    if not _is_unique(all_clues, solution, n_houses, features, feature_values, timeout_ms=10_000):
        return None  # degenerate puzzle (shouldn't happen)

    minimal_clues = _minimize_clues(all_clues, solution, n_houses, features, feature_values)

    # Sanity: minimal clues must still be unique
    if not _is_unique(minimal_clues, solution, n_houses, features, feature_values):
        return None

    return solution, minimal_clues, feature_values


# ---------------------------------------------------------------------------
# JSONL output
# ---------------------------------------------------------------------------

def _puzzle_id(n_houses: int, n_features: int, seed: int, index: int) -> str:
    raw = f"zebralogic_{n_houses}x{n_features}_{seed}_{index}"
    h = hashlib.sha1(raw.encode()).hexdigest()[:8]
    return f"zebralogic_{n_houses}x{n_features}_{h}"


def _puzzle_to_record(
    pid: str,
    n_houses: int,
    features: List[str],
    feature_values: Dict[str, List[str]],
    clues: List[Tuple],
    solution: Dict[int, Dict[str, str]],
    difficulty: int,
) -> UnifiedProblem:
    question = _format_question(n_houses, features, feature_values, clues)
    canonical = _format_canonical_answer(solution)
    return UnifiedProblem(
        problem_id=pid,
        domain="logic",
        difficulty=difficulty,
        source="zebralogic_generated",
        question=question,
        canonical_answer=canonical,
        verification_metadata={
            "grid_size": [n_houses, len(features)],
            "features": features,
            "cell_count": n_houses * len(features),
            "n_clues": len(clues),
        },
        raw_source_entry={
            "clues": [list(c) for c in clues],
            "feature_values": feature_values,
        },
    )


# ---------------------------------------------------------------------------
# Generation plan
# ---------------------------------------------------------------------------

# (n_houses, n_features, difficulty, count)
GENERATION_PLAN: List[Tuple[int, int, int, int]] = [
    # Difficulty 3: 3×3 and 3×4
    (3, 3, 3, 250),
    (3, 4, 3, 250),
    # Difficulty 4: 4×4 and 4×5
    (4, 4, 4, 250),
    (4, 5, 4, 250),
    # Difficulty 5: 5×5 and 6×6 (slow)
    (5, 5, 5, 200),
    (6, 6, 5, 100),
]


def ingest(
    plan: Optional[List[Tuple[int, int, int, int]]] = None,
    output_path: Optional[Path] = None,
    seed: int = 42,
    checkpoint_interval: int = 50,
) -> Dict[str, Any]:
    """Generate and write ZebraLogic puzzles.

    Parameters
    ----------
    plan:
        List of (n_houses, n_features, difficulty, count) tuples.
        Defaults to ``GENERATION_PLAN``.
    output_path:
        Destination JSONL file.  Defaults to
        ``data/processed/logic_zebralogic.jsonl``.
    seed:
        Base random seed for reproducibility.
    checkpoint_interval:
        Flush and print progress every N puzzles.
    """
    if plan is None:
        plan = GENERATION_PLAN

    repo_root = Path(__file__).resolve().parents[2]
    out = output_path or (repo_root / "data" / "processed" / "logic_zebralogic.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Load already-written IDs so we can resume
    seen_ids: set = set()
    if out.exists():
        with out.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    if pid := rec.get("problem_id"):
                        seen_ids.add(pid)
                except (json.JSONDecodeError, KeyError):
                    pass
        if seen_ids:
            print(
                f"[regenerate_zebralogic] resuming: {len(seen_ids)} puzzles already written",
                file=sys.stderr,
            )

    n_written = 0
    n_skipped = 0
    n_failed = 0
    per_difficulty: Counter = Counter()

    mode = "a" if seen_ids else "w"
    with out.open(mode, encoding="utf-8") as fh:
        for n_houses, n_features, difficulty, count in plan:
            features = FEATURE_ORDER[:n_features]
            rng = random.Random(seed ^ (n_houses * 1000 + n_features * 100 + difficulty))

            generated_for_block = 0
            attempt = 0
            block_id_base = n_written + n_skipped

            while generated_for_block < count:
                attempt += 1
                index = block_id_base + attempt
                pid = _puzzle_id(n_houses, n_features, seed, index)

                if pid in seen_ids:
                    n_skipped += 1
                    generated_for_block += 1
                    continue

                try:
                    result = _try_generate_puzzle(n_houses, features, rng)
                except Exception as exc:
                    print(
                        f"[regenerate_zebralogic] ERROR {n_houses}×{n_features}: {exc}",
                        file=sys.stderr,
                    )
                    n_failed += 1
                    if n_failed > 50:
                        print("[regenerate_zebralogic] too many failures, aborting", file=sys.stderr)
                        break
                    continue

                if result is None:
                    n_failed += 1
                    continue

                n_failed = 0  # reset per-run failure count on success
                solution, clues, feature_values = result

                record = _puzzle_to_record(
                    pid, n_houses, features, feature_values,
                    clues, solution, difficulty,
                )
                fh.write(record.to_jsonl() + "\n")
                seen_ids.add(pid)
                n_written += 1
                generated_for_block += 1
                per_difficulty[difficulty] += 1

                if n_written % checkpoint_interval == 0:
                    fh.flush()
                    print(
                        f"[regenerate_zebralogic] {n_written} puzzles written "
                        f"(diff={dict(sorted(per_difficulty.items()))})",
                        file=sys.stderr,
                    )

    summary = {
        "written": n_written,
        "skipped_resumed": n_skipped,
        "per_difficulty": dict(sorted(per_difficulty.items())),
        "output_path": str(out),
    }
    return summary


def _print_summary(s: Dict[str, Any]) -> None:
    print("=" * 60)
    print(f"Wrote:    {s['written']} puzzles → {s['output_path']}")
    print(f"Resumed:  {s['skipped_resumed']} puzzles already present")
    print(f"Per diff: {s['per_difficulty']}")
    print("=" * 60)


if __name__ == "__main__":
    summary = ingest()
    _print_summary(summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "output_path"}))
