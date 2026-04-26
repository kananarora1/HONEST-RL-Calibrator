"""Unified sampler that exposes processed JSONL datasets behind the same
interface as the procedural generators in ``server/generators/``.

Canonical interface:

    def generate(difficulty: int, seed: Optional[int] = None) -> tuple[str, str, str]:
        # returns (question, canonical_answer_as_string, problem_id)

Exposed as three bound methods:
    sampler.math_generate(difficulty, seed)
    sampler.code_generate(difficulty, seed)
    sampler.logic_generate(difficulty, seed)

Logic ``canonical_answer`` is a dict in the JSONL; it is serialized to a
JSON string here so the return type is always ``(str, str, str)``.  The
logic verifier re-parses the JSON string internally.

A ``verify()`` dispatcher is also exposed for the environment reward function:
    sampler.verify(problem_id: str, model_answer: str) -> bool

Module-level convenience functions ``generate_math``, ``generate_code`` and
``generate_logic`` are also exposed; they delegate to a process-wide
``UnifiedSampler`` singleton via ``get_sampler()``.
"""

from __future__ import annotations

import json
import logging
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from data.schema import UnifiedProblem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "processed"


# ---------------------------------------------------------------------------
# UnifiedSampler
# ---------------------------------------------------------------------------


class UnifiedSampler:
    """Load all processed problems into memory, expose per-domain generator
    methods with the exact same signature as ``server/generators/*_gen.py``."""

    def __init__(self, data_dir: Optional[Path | str] = None) -> None:
        self._data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR

        # (domain, difficulty) -> [UnifiedProblem, ...]
        self._buckets: Dict[Tuple[str, int], List[UnifiedProblem]] = defaultdict(list)
        # problem_id -> UnifiedProblem  (for verify() lookup)
        self._by_id: Dict[str, UnifiedProblem] = {}

        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._data_dir.exists():
            logger.warning("Data dir %s does not exist — sampler is empty.", self._data_dir)
            return

        total = 0
        for jsonl_path in sorted(self._data_dir.glob("*.jsonl")):
            file_count = 0
            with open(jsonl_path, encoding="utf-8") as fh:
                for lineno, raw in enumerate(fh, 1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        prob = UnifiedProblem.from_jsonl(raw)
                    except Exception as exc:
                        logger.debug(
                            "Skipping malformed line %d in %s: %s",
                            lineno, jsonl_path.name, exc,
                        )
                        continue
                    key = (prob.domain, prob.difficulty)
                    self._buckets[key].append(prob)
                    self._by_id[prob.problem_id] = prob
                    file_count += 1
            logger.info("  Loaded %4d records from %s", file_count, jsonl_path.name)
            total += file_count

        logger.info("UnifiedSampler ready: %d problems across %d buckets.", total, len(self._buckets))
        self._log_distribution()

    def _log_distribution(self) -> None:
        for (domain, diff), probs in sorted(self._buckets.items()):
            logger.info("  (%s, diff=%d) -> %d problems", domain, diff, len(probs))

    # ------------------------------------------------------------------
    # Internal sampler
    # ------------------------------------------------------------------

    def _sample(
        self,
        domain: str,
        difficulty: int,
        seed: Optional[int],
    ) -> UnifiedProblem:
        key = (domain, difficulty)
        pool = self._buckets.get(key)

        if not pool:
            # Fallback to nearest available difficulty for this domain
            domain_diffs = sorted(
                diff for (d, diff) in self._buckets if d == domain and self._buckets[(d, diff)]
            )
            if not domain_diffs:
                raise RuntimeError(
                    f"No problems loaded for domain '{domain}'. "
                    "Did you run the ingestion scripts?"
                )
            nearest = min(domain_diffs, key=lambda d: abs(d - difficulty))
            warnings.warn(
                f"No problems for ({domain}, difficulty={difficulty}); "
                f"falling back to difficulty={nearest}.",
                stacklevel=3,
            )
            pool = self._buckets[(domain, nearest)]

        rng = random.Random(seed) if seed is not None else random
        return rng.choice(pool)

    # ------------------------------------------------------------------
    # Generator methods — exact same signature as server/generators/*_gen.py
    # ------------------------------------------------------------------

    def math_generate(
        self,
        difficulty: int,
        seed: Optional[int] = None,
    ) -> Tuple[str, str, str]:
        """Return (question, canonical_answer, problem_id) for a math problem."""
        prob = self._sample("math", difficulty, seed)
        # math canonical_answer is always a string in the schema
        answer = str(prob.canonical_answer)
        return prob.question, answer, prob.problem_id

    def code_generate(
        self,
        difficulty: int,
        seed: Optional[int] = None,
    ) -> Tuple[str, str, str]:
        """Return (question, canonical_answer, problem_id) for a code problem."""
        prob = self._sample("code", difficulty, seed)
        answer = str(prob.canonical_answer)
        return prob.question, answer, prob.problem_id

    def logic_generate(
        self,
        difficulty: int,
        seed: Optional[int] = None,
    ) -> Tuple[str, str, str]:
        """Return (question, canonical_answer_json_str, problem_id) for a logic problem.

        The canonical_answer in the JSONL is a dict; we serialize it to a
        JSON string so the return type remains ``(str, str, str)``.  The
        logic verifier re-parses it internally.
        """
        prob = self._sample("logic", difficulty, seed)
        if isinstance(prob.canonical_answer, dict):
            answer = json.dumps(prob.canonical_answer)
        else:
            answer = str(prob.canonical_answer)
        return prob.question, answer, prob.problem_id

    # ------------------------------------------------------------------
    # verify() dispatcher
    # ------------------------------------------------------------------

    def verify(self, problem_id: str, model_answer: str) -> bool:
        """Dispatch to the correct domain verifier and return a bool.

        Parameters
        ----------
        problem_id:
            The ``problem_id`` field from the sampled ``UnifiedProblem``.
        model_answer:
            The raw string output from the model.

        Returns
        -------
        bool — True iff the model_answer is correct per the domain verifier.
        """
        prob = self._by_id.get(problem_id)
        if prob is None:
            logger.warning("verify() called with unknown problem_id=%r", problem_id)
            return False

        try:
            if prob.domain == "math":
                from data.verifiers.math_verifier import verify_math_answer
                return verify_math_answer(model_answer, str(prob.canonical_answer))

            elif prob.domain == "code":
                from data.verifiers.code_verifier import verify_code_answer
                return verify_code_answer(model_answer, prob.verification_metadata)

            elif prob.domain == "logic":
                from data.verifiers.logic_verifier import verify_logic_answer
                # Returns (bool, float); we drop the float to keep the binary contract
                passed, _acc = verify_logic_answer(
                    model_answer,
                    prob.canonical_answer,
                    prob.verification_metadata,
                )
                return passed

            else:
                logger.warning("Unknown domain '%s' for problem_id=%r", prob.domain, problem_id)
                return False

        except Exception as exc:
            logger.error(
                "verify() raised for problem_id=%r domain=%s: %s",
                problem_id, prob.domain, exc,
            )
            return False

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def bucket_counts(self) -> Dict[Tuple[str, int], int]:
        """Return a dict of (domain, difficulty) -> count."""
        return {k: len(v) for k, v in sorted(self._buckets.items())}

    def total_count(self) -> int:
        return len(self._by_id)


# ---------------------------------------------------------------------------
# Module-level singleton + convenience functions
# ---------------------------------------------------------------------------

_SINGLETON: Optional[UnifiedSampler] = None


def get_sampler() -> UnifiedSampler:
    """Return the process-wide ``UnifiedSampler``, loading it on first call."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = UnifiedSampler()
    return _SINGLETON


def generate_math(difficulty: int, seed: Optional[int] = None) -> Tuple[str, str, str]:
    """Module-level shim — returns (question, canonical_answer, problem_id).

    Falls back to the procedural math generator for any difficulty whose curated
    bucket is empty (e.g. when Hendrycks-MATH has not been ingested yet), so the
    correct difficulty level is always served.
    """
    sampler = get_sampler()
    key = ("math", difficulty)
    if sampler._buckets.get(key):
        return sampler.math_generate(difficulty, seed)

    from server.generators import math_gen as _procedural_math
    question, answer = _procedural_math.generate(difficulty, seed=seed)
    problem_id = f"procedural_math_d{difficulty}_{hash(question) & 0xFFFFFFFF:08x}"
    return question, answer, problem_id


def generate_code(difficulty: int, seed: Optional[int] = None) -> Tuple[str, str, str]:
    """Module-level shim — returns (question, canonical_answer, problem_id).

    Falls back to the procedural code generator for any difficulty whose curated
    bucket is empty (mirrors the logic domain's handling of d=1–2).  This avoids
    the 'falling back to difficulty=2' warning when APPS (d=3–5) has not been
    ingested yet, and provides a well-formed problem at the *correct* difficulty.
    """
    sampler = get_sampler()
    key = ("code", difficulty)
    if sampler._buckets.get(key):
        return sampler.code_generate(difficulty, seed)

    from server.generators import code_gen as _procedural_code
    question, answer = _procedural_code.generate(difficulty, seed=seed)
    problem_id = f"procedural_code_d{difficulty}_{hash(question) & 0xFFFFFFFF:08x}"
    return question, answer, problem_id


def generate_logic(difficulty: int, seed: Optional[int] = None) -> Tuple[str, str, str]:
    """Module-level shim — returns (question, canonical_answer_json_str, problem_id).

    Logic is dispatched here based on difficulty: difficulties 1-2 are served
    by the procedural generator in ``server.generators.logic_gen`` (string
    answers, ``problem_id`` prefixed with ``procedural_logic_``); difficulties
    3-5 are served by the curated ZebraLogic dataset (JSON-grid answers).
    """
    if difficulty <= 2:
        # Imported lazily so an unavailable procedural generator does not
        # prevent the rest of the sampler from loading.
        from server.generators import logic_gen as _procedural_logic
        question, answer = _procedural_logic.generate(difficulty, seed=seed)
        problem_id = f"procedural_logic_d{difficulty}_{hash(question) & 0xFFFFFFFF:08x}"
        return question, answer, problem_id
    return get_sampler().logic_generate(difficulty, seed)
