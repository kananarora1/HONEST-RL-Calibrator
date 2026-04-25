"""Tests for the adaptive ``DifficultyController`` in ``server/difficulty.py``.

Run from the project root:
    PYTHONPATH=. pytest data/tests/test_difficulty_controller.py -v
"""

from __future__ import annotations

import math
import random
from collections import Counter

import pytest

from server.difficulty import (
    ADAPTIVE_BUDGET,
    DifficultyController,
    STATIC_FLOOR,
    compute_distribution,
    triangular_overlay,
)


DOMAINS = ["math", "code", "logic"]


# ---------------------------------------------------------------------------
# 1. Distribution sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", [1, 2, 3, 4, 5])
def test_distribution_sums_to_one(target):
    dist = compute_distribution(target)
    assert len(dist) == 5
    assert math.isclose(sum(dist), 1.0, abs_tol=1e-9)


@pytest.mark.parametrize("target", [1, 2, 3, 4, 5])
def test_distribution_all_non_negative(target):
    dist = compute_distribution(target)
    assert all(w >= 0.0 for w in dist)


# ---------------------------------------------------------------------------
# 2. Floor preservation (catastrophic-forgetting protection)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", [1, 2, 3, 4, 5])
def test_floor_preserves_d1_minimum(target):
    """Difficulty-1 weight must always be >= the static floor for d1 (0.20)."""
    dist = compute_distribution(target)
    assert dist[0] >= STATIC_FLOOR[0] - 1e-9, (
        f"target={target}: d1 weight {dist[0]:.3f} dropped below floor"
    )


@pytest.mark.parametrize("target", [1, 2, 3, 4, 5])
def test_floor_preserves_easy_combined_minimum(target):
    """Combined d1+d2 weight must always be >= 0.35 (the easy floor)."""
    dist = compute_distribution(target)
    easy = dist[0] + dist[1]
    assert easy >= 0.35 - 1e-9, (
        f"target={target}: easy floor d1+d2 = {easy:.3f} fell below 0.35"
    )


def test_overlay_sums_to_budget():
    for t in [1, 2, 3, 4, 5]:
        overlay = triangular_overlay(t)
        assert math.isclose(sum(overlay), ADAPTIVE_BUDGET, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# 3. Cooldown enforcement
# ---------------------------------------------------------------------------


def test_cooldown_blocks_early_changes():
    """5 correct outcomes → not enough to update (cooldown=10 AND window not full)."""
    ctrl = DifficultyController(DOMAINS)
    initial = ctrl.get_target("math")
    for _ in range(5):
        ctrl.record_outcome("math", correct=True)
    assert ctrl.get_target("math") == initial


# ---------------------------------------------------------------------------
# 4. Hysteresis up
# ---------------------------------------------------------------------------


def test_hysteresis_up_promotes_after_window_fills():
    """20 correct outcomes — window full, cooldown elapsed, accuracy=1.0 ≥ 0.75."""
    ctrl = DifficultyController(DOMAINS)
    assert ctrl.get_target("math") == 1
    for _ in range(20):
        ctrl.record_outcome("math", correct=True)
    assert ctrl.get_target("math") == 2
    # Cooldown reset by the bump
    assert ctrl.state["math"].episodes_since_last_update == 0


# ---------------------------------------------------------------------------
# 5. Hysteresis down
# ---------------------------------------------------------------------------


def test_hysteresis_down_demotes_after_window_fills():
    ctrl = DifficultyController(DOMAINS, initial_target=3)
    for _ in range(20):
        ctrl.record_outcome("math", correct=False)
    assert ctrl.get_target("math") == 2


# ---------------------------------------------------------------------------
# 6. Hysteresis dead zone
# ---------------------------------------------------------------------------


def test_hysteresis_dead_zone_stays_put():
    """50% accuracy is in (0.25, 0.75) → no change."""
    ctrl = DifficultyController(DOMAINS, initial_target=3)
    outcomes = ([True, False] * 10)  # 20 outcomes, 50% accuracy
    for c in outcomes:
        ctrl.record_outcome("math", correct=c)
    assert ctrl.get_target("math") == 3


# ---------------------------------------------------------------------------
# 7. Bounds (floor / ceiling)
# ---------------------------------------------------------------------------


def test_target_does_not_drop_below_min():
    ctrl = DifficultyController(DOMAINS, initial_target=1)
    for _ in range(40):
        ctrl.record_outcome("math", correct=False)
    assert ctrl.get_target("math") == 1


def test_target_does_not_exceed_max():
    ctrl = DifficultyController(DOMAINS, initial_target=5)
    for _ in range(40):
        ctrl.record_outcome("math", correct=True)
    assert ctrl.get_target("math") == 5


# ---------------------------------------------------------------------------
# 8. Per-domain independence
# ---------------------------------------------------------------------------


def test_domains_track_independently():
    ctrl = DifficultyController(DOMAINS)
    for _ in range(20):
        ctrl.record_outcome("math", correct=True)
        ctrl.record_outcome("code", correct=False)
    assert ctrl.get_target("math") == 2
    assert ctrl.get_target("code") == 1  # already at floor — can't drop further
    # logic was untouched
    assert ctrl.get_target("logic") == 1
    assert ctrl.get_rolling_accuracy("logic") is None


# ---------------------------------------------------------------------------
# 9. Empirical sampling matches computed distribution
# ---------------------------------------------------------------------------


def test_sampling_matches_distribution():
    ctrl = DifficultyController(DOMAINS, initial_target=3)
    rng = random.Random(20260426)
    n = 10_000
    samples = [ctrl.sample_difficulty("math", rng=rng) for _ in range(n)]
    counts = Counter(samples)
    expected = compute_distribution(3)
    for d in [1, 2, 3, 4, 5]:
        observed = counts[d] / n
        # 2 std dev for a binomial proportion at n=10k is ~ 2 * sqrt(p*(1-p)/n)
        sigma = math.sqrt(expected[d - 1] * (1 - expected[d - 1]) / n)
        tol = max(2 * sigma, 0.005)
        assert abs(observed - expected[d - 1]) <= tol, (
            f"d={d}: empirical {observed:.4f} vs expected {expected[d-1]:.4f} "
            f"(tolerance {tol:.4f})"
        )


# ---------------------------------------------------------------------------
# 10. Abstain / malformed do NOT pollute the rolling window
# ---------------------------------------------------------------------------


def test_controller_only_records_real_outcomes():
    """Caller must not pass None into record_outcome; window length tracks
    only the True/False outcomes that are actually fed in."""
    ctrl = DifficultyController(DOMAINS)
    for _ in range(3):
        ctrl.record_outcome("math", correct=True)
    # Simulate that abstain/malformed episodes were skipped by the caller —
    # the window should reflect only the 3 real outcomes.
    s = ctrl.state["math"]
    assert len(s.rolling_window) == 3
    assert sum(s.rolling_window) == 3
    # Cooldown also only ticks on real outcomes
    assert s.episodes_since_last_update == 3


# ---------------------------------------------------------------------------
# Bonus: snapshot shape
# ---------------------------------------------------------------------------


def test_snapshot_contains_expected_keys():
    ctrl = DifficultyController(DOMAINS)
    snap = ctrl.snapshot()
    assert set(snap.keys()) == set(DOMAINS)
    for s in snap.values():
        assert {
            "target_difficulty",
            "rolling_accuracy",
            "episodes_since_update",
            "window_full",
            "window_size",
            "distribution",
        } <= s.keys()
        assert len(s["distribution"]) == 5
