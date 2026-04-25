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

from typing import Optional, Tuple

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
