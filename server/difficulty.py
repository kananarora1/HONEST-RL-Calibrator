"""Adaptive difficulty management for the HONEST environment.

The rolling accuracy window looks at records in ``state.episode_history``
that have a ``"domain"`` key and a ``"correct"`` key.  Those records are
owned and written **exclusively by** ``HonestEnvironment.step()`` — this
module is a pure analyser and mutates only ``state.domain_difficulties``.

``update_difficulty`` is now a **pure side-effect-free function** w.r.t.
history: it reads history, optionally updates the difficulty scalar, and
returns ``(new_difficulty, changed)``.  The caller (environment) is
responsible for flagging the relevant history record with
``"difficulty_changed": True`` if it wishes to track transitions.
"""

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from models.models import HonestState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW: int = 20              # rolling accuracy window (episodes per domain)
HIGH_THRESHOLD: float = 0.70  # accuracy above this → increase difficulty
LOW_THRESHOLD:  float = 0.30  # accuracy below this → decrease difficulty
MIN_DIFFICULTY: int = 1
MAX_DIFFICULTY: int = 5
HYSTERESIS_EPISODES: int = 10  # min episodes between consecutive changes


# ---------------------------------------------------------------------------
# Adaptive sampling distribution: static floor + triangular overlay
# ---------------------------------------------------------------------------

# Always-on weight per difficulty 1..5 — protects against catastrophic
# forgetting of easy-problem competence as the curriculum advances.
STATIC_FLOOR: List[float] = [0.20, 0.15, 0.10, 0.05, 0.00]  # sums to 0.50

# Remaining weight (0.50) is distributed by a triangular kernel around
# the controller's current target_difficulty.
ADAPTIVE_BUDGET: float = 0.50


def triangular_overlay(target: int, total_weight: float = ADAPTIVE_BUDGET) -> List[float]:
    """Triangular distribution centered at ``target``, summing to ``total_weight``.

    Difficulties are 1-indexed; returns a 5-element list.
    Kernel: ``max(0, 3 - |target - d|)`` over d in [1..5], then renormalised
    to ``total_weight``.  At the edges (target=1 or target=5) the kernel is
    clipped, so less mass lands on phantom out-of-range difficulties — but
    the surviving mass is still renormalised so the overlay always sums to
    ``total_weight``.
    """
    raw = [max(0, 3 - abs(target - d)) for d in range(1, 6)]
    s = sum(raw)
    if s == 0:
        return [0.0] * 5
    return [r * total_weight / s for r in raw]


def compute_distribution(target_difficulty: int) -> List[float]:
    """Static floor + adaptive overlay.  Returns weights for difficulties 1..5.

    The result is renormalised to sum to exactly 1.0 to absorb floating-point
    drift, so callers can pass it directly to ``random.choices`` weights.
    """
    overlay = triangular_overlay(target_difficulty)
    distribution = [STATIC_FLOOR[i] + overlay[i] for i in range(5)]
    total = sum(distribution)
    return [d / total for d in distribution]


# ---------------------------------------------------------------------------
# DomainState + DifficultyController
# ---------------------------------------------------------------------------


@dataclass
class DomainState:
    """Per-domain state held by ``DifficultyController``."""
    target_difficulty: int = 1
    rolling_window: deque = field(default_factory=lambda: deque(maxlen=20))
    episodes_since_last_update: int = 0


class DifficultyController:
    """Adaptive difficulty controller with a static floor.

    Per-domain state: rolling 20-episode accuracy window, a target difficulty
    scalar, and a cooldown counter.  Hysteresis thresholds 75 / 25, cooldown
    of 10 outcomes per domain.

    Lifetime: one instance per ``HonestEnvironment`` (or one per training
    process for the local-rollout path).  The controller persists across
    episode boundaries — its state is *not* reset by ``env.reset()``.
    """

    UPDATE_THRESHOLD_UP: float = 0.75
    UPDATE_THRESHOLD_DOWN: float = 0.25
    COOLDOWN_EPISODES: int = 10
    WINDOW_SIZE: int = 20
    DIFFICULTY_MIN: int = MIN_DIFFICULTY
    DIFFICULTY_MAX: int = MAX_DIFFICULTY

    def __init__(self, domains: List[str], initial_target: int = 1) -> None:
        self.domains = list(domains)
        self.state: Dict[str, DomainState] = {
            d: DomainState(target_difficulty=initial_target) for d in self.domains
        }

    # --- sampling ----------------------------------------------------------

    def sample_difficulty(
        self,
        domain: str,
        rng: Optional[random.Random] = None,
    ) -> int:
        """Sample a difficulty 1..5 for ``domain`` using the current distribution."""
        target = self.state[domain].target_difficulty
        weights = compute_distribution(target)
        chooser = rng if rng is not None else random
        return chooser.choices([1, 2, 3, 4, 5], weights=weights, k=1)[0]

    # --- outcome tracking --------------------------------------------------

    def record_outcome(self, domain: str, correct: bool) -> Tuple[int, bool]:
        """Record an episode outcome.

        Returns ``(new_target_difficulty, did_update)``.  Should be called
        only with True/False — abstain / malformed (correct=None) episodes
        must NOT enter the rolling window.
        """
        s = self.state[domain]
        s.rolling_window.append(1 if correct else 0)
        s.episodes_since_last_update += 1

        did_update = False
        if (
            s.episodes_since_last_update >= self.COOLDOWN_EPISODES
            and len(s.rolling_window) >= self.WINDOW_SIZE
        ):
            accuracy = sum(s.rolling_window) / len(s.rolling_window)

            if (
                accuracy >= self.UPDATE_THRESHOLD_UP
                and s.target_difficulty < self.DIFFICULTY_MAX
            ):
                s.target_difficulty += 1
                did_update = True
            elif (
                accuracy <= self.UPDATE_THRESHOLD_DOWN
                and s.target_difficulty > self.DIFFICULTY_MIN
            ):
                s.target_difficulty -= 1
                did_update = True

            if did_update:
                # Reset the cooldown but keep the rolling window — the new
                # target's accuracy estimate phases in as fresh outcomes flow.
                s.episodes_since_last_update = 0

        return s.target_difficulty, did_update

    # --- introspection -----------------------------------------------------

    def get_distribution(self, domain: str) -> List[float]:
        return compute_distribution(self.state[domain].target_difficulty)

    def get_target(self, domain: str) -> int:
        return self.state[domain].target_difficulty

    def get_rolling_accuracy(self, domain: str) -> Optional[float]:
        s = self.state[domain]
        if len(s.rolling_window) == 0:
            return None
        return sum(s.rolling_window) / len(s.rolling_window)

    def snapshot(self) -> Dict[str, dict]:
        """Return a JSON-serialisable snapshot for logging / debugging."""
        return {
            d: {
                "target_difficulty": s.target_difficulty,
                "rolling_accuracy": self.get_rolling_accuracy(d),
                "episodes_since_update": s.episodes_since_last_update,
                "window_full": len(s.rolling_window) == self.WINDOW_SIZE,
                "window_size": len(s.rolling_window),
                "distribution": self.get_distribution(d),
            }
            for d, s in self.state.items()
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _domain_records(state: HonestState, domain: str) -> list[dict]:
    """Return the last WINDOW episode records for *domain* from history.

    Only records that contain both 'domain' and 'correct' keys are
    considered (i.e. the rich records written by the environment, not
    any stale auxiliary records).
    """
    records = [
        r for r in state.episode_history
        if r.get("domain") == domain and "correct" in r
    ]
    return records[-WINDOW:]


def _last_change_episode(state: HonestState, domain: str) -> int:
    """Return the global episode index of the most recent difficulty change
    for *domain*.

    Scans ``episode_history`` backwards for a record flagged with
    ``"difficulty_changed": True`` for the given domain.
    Returns 0 if no change has ever occurred (safe to change from episode 0).
    """
    for idx in range(len(state.episode_history) - 1, -1, -1):
        r = state.episode_history[idx]
        if r.get("domain") == domain and r.get("difficulty_changed"):
            return idx
    return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_rolling_accuracy(state: HonestState, domain: str) -> float:
    """Return the rolling accuracy (0.0–1.0) for *domain* over the last
    WINDOW answered episodes.

    Episodes where ``correct`` is ``None`` (abstain / malformed) are treated
    as incorrect.  Returns 0.5 (neutral) when there are no episodes yet.
    """
    records = _domain_records(state, domain)
    if not records:
        return 0.5  # neutral default — no change triggered
    correct_count = sum(1 for r in records if r.get("correct") is True)
    return correct_count / len(records)


def update_difficulty(
    state: HonestState,
    last_correctness: Optional[bool],
    domain: Optional[str] = None,
) -> Tuple[int, bool]:
    """Evaluate rolling accuracy and adjust ``state.domain_difficulties``
    in-place if a threshold is crossed.

    Parameters
    ----------
    state:
        The current environment state (mutated in-place for the difficulty
        scalar only — history is **not** touched here).
    last_correctness:
        Whether the most recent answer was correct.  ``None`` for
        abstain / malformed answers (counted as incorrect).
    domain:
        Override for the active domain.  Defaults to ``state.current_domain``.

    Returns
    -------
    (new_difficulty, changed) where ``changed`` is True iff the difficulty
    scalar was actually modified this call.  The caller can use this to
    stamp ``"difficulty_changed": True`` onto its own history record.
    """
    if domain is None:
        domain = state.current_domain

    current_difficulty = state.domain_difficulties.get(domain, 1)

    # We need to know how many history entries exist *before* the caller
    # appends the current step.  The caller must append its rich record
    # *before* calling this function so rolling accuracy includes the
    # current result.
    global_episode_idx = len(state.episode_history) - 1  # 0-indexed last item

    # --- compute rolling accuracy (includes the just-appended record) ---
    accuracy = get_rolling_accuracy(state, domain)

    # --- hysteresis guard ---
    last_change = _last_change_episode(state, domain)
    episodes_since_change = global_episode_idx - last_change
    if episodes_since_change < HYSTERESIS_EPISODES:
        return current_difficulty, False  # too soon to change

    # --- apply threshold rules ---
    new_difficulty = current_difficulty
    if accuracy > HIGH_THRESHOLD:
        new_difficulty = min(current_difficulty + 1, MAX_DIFFICULTY)
    elif accuracy < LOW_THRESHOLD:
        new_difficulty = max(current_difficulty - 1, MIN_DIFFICULTY)

    changed = new_difficulty != current_difficulty
    if changed:
        state.domain_difficulties[domain] = new_difficulty

    return new_difficulty, changed
