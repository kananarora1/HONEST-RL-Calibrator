"""Tests for server/difficulty.py."""

import pytest

from models.models import HonestState
from server.difficulty import (
    HYSTERESIS_EPISODES,
    MAX_DIFFICULTY,
    MIN_DIFFICULTY,
    WINDOW,
    get_rolling_accuracy,
    update_difficulty,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_state(domain: str = "math", difficulty: int = 3) -> HonestState:
    return HonestState(
        current_domain=domain,
        domain_difficulties={domain: difficulty},
    )


def run_episodes(state: HonestState, outcomes: list, domain: str | None = None):
    """Feed a sequence of True/False/None outcomes into update_difficulty.

    Mirrors what HonestEnvironment.step() does: append the rich record first
    so update_difficulty (now a pure reader) can compute rolling accuracy from
    the current step included.
    """
    active_domain = domain or state.current_domain
    diff = state.domain_difficulties.get(active_domain, 1)
    for outcome in outcomes:
        record: dict = {
            "domain": active_domain,
            "correct": outcome,
            "difficulty": diff,
            "difficulty_changed": False,
        }
        state.episode_history.append(record)
        new_diff, changed = update_difficulty(state, outcome, domain=active_domain)
        if changed:
            record["difficulty_changed"] = True
        diff = new_diff


# ---------------------------------------------------------------------------
# get_rolling_accuracy
# ---------------------------------------------------------------------------


class TestGetRollingAccuracy:
    def test_returns_neutral_for_empty_history(self):
        state = make_state()
        assert get_rolling_accuracy(state, "math") == pytest.approx(0.5)

    def test_all_correct(self):
        state = make_state()
        run_episodes(state, [True] * 5)
        assert get_rolling_accuracy(state, "math") == pytest.approx(1.0)

    def test_all_wrong(self):
        state = make_state()
        run_episodes(state, [False] * 5)
        assert get_rolling_accuracy(state, "math") == pytest.approx(0.0)

    def test_mixed(self):
        state = make_state()
        run_episodes(state, [True, True, False, False, True])  # 3/5 = 0.6
        assert get_rolling_accuracy(state, "math") == pytest.approx(0.6)

    def test_window_is_capped_at_20(self):
        state = make_state()
        # 25 episodes: first 5 wrong, last 20 all correct
        run_episodes(state, [False] * 5 + [True] * 20)
        # Window only sees last 20 — all correct
        assert get_rolling_accuracy(state, "math") == pytest.approx(1.0)

    def test_none_counted_as_wrong(self):
        state = make_state()
        run_episodes(state, [None, None, True])  # 1/3
        assert get_rolling_accuracy(state, "math") == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Difficulty increases
# ---------------------------------------------------------------------------


class TestDifficultyIncrease:
    def test_increases_after_sustained_high_accuracy(self):
        state = make_state(difficulty=2)
        # Fill window: 15 correct out of 15 → accuracy = 1.0 > 0.70
        # hysteresis: need gap of 10 from last change (init = 0), so episode 10+
        run_episodes(state, [True] * 15)
        assert state.domain_difficulties["math"] == 3

    def test_does_not_increase_before_hysteresis_clears(self):
        state = make_state(difficulty=2)
        # 9 correct → accuracy > 0.70 but hysteresis blocks (< 10 episodes since ep 0)
        run_episodes(state, [True] * 9)
        assert state.domain_difficulties["math"] == 2

    def test_caps_at_max_difficulty(self):
        state = make_state(difficulty=MAX_DIFFICULTY)
        run_episodes(state, [True] * 15)
        assert state.domain_difficulties["math"] == MAX_DIFFICULTY


# ---------------------------------------------------------------------------
# Difficulty decreases
# ---------------------------------------------------------------------------


class TestDifficultyDecrease:
    def test_decreases_after_sustained_low_accuracy(self):
        state = make_state(difficulty=3)
        # 15 wrong → accuracy = 0.0 < 0.30
        run_episodes(state, [False] * 15)
        assert state.domain_difficulties["math"] == 2

    def test_does_not_decrease_before_hysteresis_clears(self):
        state = make_state(difficulty=3)
        run_episodes(state, [False] * 9)
        assert state.domain_difficulties["math"] == 3

    def test_floors_at_min_difficulty(self):
        state = make_state(difficulty=MIN_DIFFICULTY)
        run_episodes(state, [False] * 15)
        assert state.domain_difficulties["math"] == MIN_DIFFICULTY


# ---------------------------------------------------------------------------
# Hysteresis — no rapid oscillation
# ---------------------------------------------------------------------------


class TestHysteresis:
    def test_no_second_change_before_hysteresis_window(self):
        state = make_state(difficulty=2)
        # First increase fires at episode ~10
        run_episodes(state, [True] * 11)
        diff_after_first = state.domain_difficulties["math"]
        assert diff_after_first == 3

        # Immediately continue with more correct answers — next change
        # should NOT fire until HYSTERESIS_EPISODES more episodes pass
        run_episodes(state, [True] * (HYSTERESIS_EPISODES - 1))
        assert state.domain_difficulties["math"] == 3  # still 3

    def test_second_change_fires_after_hysteresis_window(self):
        state = make_state(difficulty=2)
        # First change
        run_episodes(state, [True] * 11)
        assert state.domain_difficulties["math"] == 3

        # Give hysteresis window full room + enough data: HYSTERESIS_EPISODES + some
        # We need accuracy to remain > 0.70 and 10+ episodes since last change
        run_episodes(state, [True] * (HYSTERESIS_EPISODES + 1))
        assert state.domain_difficulties["math"] == 4

    def test_oscillation_blocked(self):
        """Rapid correct→wrong cannot ping-pong the difficulty."""
        state = make_state(difficulty=3)
        # 11 correct → difficulty raised to 4
        run_episodes(state, [True] * 11)
        assert state.domain_difficulties["math"] == 4

        # Immediately 9 wrong → NOT enough to decrease (hysteresis + window)
        run_episodes(state, [False] * 9)
        assert state.domain_difficulties["math"] == 4  # no decrease yet


# ---------------------------------------------------------------------------
# Per-domain independence
# ---------------------------------------------------------------------------


class TestPerDomainIndependence:
    def test_math_accuracy_does_not_affect_code(self):
        state = HonestState(
            current_domain="math",
            domain_difficulties={"math": 2, "code": 3},
        )
        # Drive math accuracy very high
        run_episodes(state, [True] * 15, domain="math")
        # math goes up
        assert state.domain_difficulties["math"] > 2
        # code is untouched
        assert state.domain_difficulties["code"] == 3

    def test_separate_hysteresis_per_domain(self):
        state = HonestState(
            current_domain="math",
            domain_difficulties={"math": 2, "code": 2},
        )
        # Only drive math
        run_episodes(state, [True] * 11, domain="math")
        assert state.domain_difficulties["math"] == 3
        # code still untouched at 2
        assert state.domain_difficulties["code"] == 2

    def test_domains_track_independently(self):
        state = HonestState(
            current_domain="math",
            domain_difficulties={"math": 3, "logic": 3},
        )
        # Drive math down
        run_episodes(state, [False] * 15, domain="math")
        # Drive logic up
        run_episodes(state, [True] * 15, domain="logic")
        assert state.domain_difficulties["math"] < 3
        assert state.domain_difficulties["logic"] > 3


# ---------------------------------------------------------------------------
# Bounds enforcement
# ---------------------------------------------------------------------------


class TestBoundsEnforcement:
    @pytest.mark.parametrize("start", [1, 2, 3, 4, 5])
    def test_difficulty_always_in_range(self, start):
        state = make_state(difficulty=start)
        # Extreme runs in both directions
        run_episodes(state, [True] * 30 + [False] * 30)
        d = state.domain_difficulties["math"]
        assert MIN_DIFFICULTY <= d <= MAX_DIFFICULTY
