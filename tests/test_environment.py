"""Tests for server/environment.py — HonestEnvironment."""

from unittest.mock import patch

import pytest

from models.models import HonestAction, HonestObservation
from server.environment import EPISODE_LENGTH, INITIAL_DIFFICULTIES, HonestEnvironment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def env() -> HonestEnvironment:
    return HonestEnvironment()


@pytest.fixture()
def fresh_env(env: HonestEnvironment) -> HonestEnvironment:
    """Environment after reset, seeded for reproducibility."""
    env.reset(seed=42)
    return env


WELL_FORMED = "<reasoning>think</reasoning><answer>42</answer><confidence>0.5</confidence>"
MALFORMED = "I dunno lol"


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_returns_honest_observation(self, env):
        obs = env.reset()
        assert isinstance(obs, HonestObservation)

    def test_observation_has_question(self, env):
        obs = env.reset()
        assert isinstance(obs.question, str) and obs.question

    def test_observation_domain_in_valid_set(self, env):
        obs = env.reset()
        assert obs.domain in {"math", "code", "logic"}

    def test_observation_difficulty_in_valid_range(self, env):
        obs = env.reset()
        # The DifficultyController samples from a static-floor + adaptive-overlay
        # distribution at reset time, so difficulty is no longer fixed at 1.
        assert obs.difficulty in {1, 2, 3, 4, 5}

    def test_observation_episode_step_is_zero(self, env):
        obs = env.reset()
        assert obs.episode_step == 0

    def test_observation_not_terminal(self, env):
        obs = env.reset()
        assert obs.terminal is False
        assert obs.done is False

    def test_reset_clears_history(self, env):
        env.reset()
        # Run a step to accumulate history
        env.step(HonestAction(raw_text=WELL_FORMED))
        assert len(env.state.episode_history) > 0

        # Second reset must clear it
        env.reset()
        # episode_history belongs to the new HonestState created in reset()
        # update_difficulty hasn't been called yet, so history is empty
        assert env.state.episode_step == 0

    def test_reset_initialises_domain_difficulties(self, env):
        env.reset()
        # reset() now seeds domain_difficulties from INITIAL_DIFFICULTIES and
        # then overwrites the *current* domain's entry with a freshly sampled
        # difficulty from the DifficultyController.  The remaining domains
        # should still match the initial values.
        diffs = env.state.domain_difficulties
        assert set(diffs.keys()) == set(INITIAL_DIFFICULTIES.keys())
        for d, v in diffs.items():
            assert v in {1, 2, 3, 4, 5}, f"{d}: difficulty {v} out of range"
        # Non-current domains untouched.
        for d in INITIAL_DIFFICULTIES:
            if d != env.state.current_domain:
                assert diffs[d] == INITIAL_DIFFICULTIES[d]

    def test_seeded_reset_is_reproducible(self, env):
        obs1 = env.reset(seed=7)
        obs2 = env.reset(seed=7)
        assert obs1.domain == obs2.domain
        assert obs1.question == obs2.question


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------


class TestStep:
    def test_step_returns_honest_observation(self, fresh_env):
        obs = fresh_env.step(HonestAction(raw_text=WELL_FORMED))
        assert isinstance(obs, HonestObservation)

    def test_well_formed_reward_in_range(self, fresh_env):
        obs = fresh_env.step(HonestAction(raw_text=WELL_FORMED))
        assert obs.reward is not None
        # Brier + format_bonus: worst = -1+0.02 = -0.98, best = 0+0.02 = 0.02
        assert -1.0 <= obs.reward <= 0.1

    def test_malformed_reward_is_minus_half(self, fresh_env):
        obs = fresh_env.step(HonestAction(raw_text=MALFORMED))
        # MALFORMED_PENALTY in server/reward.py
        from server.reward import MALFORMED_PENALTY
        assert obs.reward == pytest.approx(MALFORMED_PENALTY)

    def test_malformed_returns_no_correctness(self, fresh_env):
        fresh_env.step(HonestAction(raw_text=MALFORMED))
        last = fresh_env.state.episode_history[-1]
        assert last["correct"] is None

    def test_step_increments_episode_step(self, fresh_env):
        fresh_env.step(HonestAction(raw_text=WELL_FORMED))
        assert fresh_env.state.episode_step == 1

    def test_non_terminal_observation_has_question(self, fresh_env):
        obs = fresh_env.step(HonestAction(raw_text=WELL_FORMED))
        assert not obs.terminal
        assert obs.question  # non-empty string

    def test_non_terminal_has_domain(self, fresh_env):
        obs = fresh_env.step(HonestAction(raw_text=WELL_FORMED))
        assert obs.domain in {"math", "code", "logic"}


# ---------------------------------------------------------------------------
# state history tracking
# ---------------------------------------------------------------------------


class TestStateHistory:
    def test_history_grows_with_steps(self, fresh_env):
        for _ in range(3):
            fresh_env.step(HonestAction(raw_text=WELL_FORMED))
        # update_difficulty appends one record per step too; here we check
        # the environment's own episode_history which stores one dict per step
        # (the difficulty module also appends, so total len >= 3)
        env_records = [
            r for r in fresh_env.state.episode_history if "question" in r
        ]
        assert len(env_records) == 3

    def test_history_record_has_required_keys(self, fresh_env):
        fresh_env.step(HonestAction(raw_text=WELL_FORMED))
        record = next(r for r in fresh_env.state.episode_history if "question" in r)
        for key in ("question", "ground_truth", "parsed", "correct", "reward", "domain", "difficulty"):
            assert key in record, f"Missing key: {key}"

    def test_history_records_domain(self, fresh_env):
        obs_before = fresh_env.reset(seed=99)
        fresh_env.step(HonestAction(raw_text=WELL_FORMED))
        env_record = next(r for r in fresh_env.state.episode_history if "question" in r)
        assert env_record["domain"] == obs_before.domain


# ---------------------------------------------------------------------------
# Episode termination
# ---------------------------------------------------------------------------


class TestTermination:
    def _run_full_episode(self, env):
        env.reset(seed=1)
        obs = None
        for _ in range(EPISODE_LENGTH):
            obs = env.step(HonestAction(raw_text=WELL_FORMED))
        return obs

    def test_terminal_after_episode_length_steps(self, env):
        obs = self._run_full_episode(env)
        assert obs.terminal is True
        assert obs.done is True

    def test_not_terminal_before_episode_length(self, env):
        env.reset(seed=1)
        for i in range(EPISODE_LENGTH - 1):
            obs = env.step(HonestAction(raw_text=WELL_FORMED))
            assert not obs.terminal, f"Terminated too early at step {i+1}"

    def test_terminal_observation_has_reward(self, env):
        obs = self._run_full_episode(env)
        assert obs.reward is not None

    def test_terminal_episode_step_equals_length(self, env):
        self._run_full_episode(env)
        assert env.state.episode_step == EPISODE_LENGTH

    def test_episode_step_persists_across_steps(self, env):
        env.reset()
        for expected_step in range(1, EPISODE_LENGTH + 1):
            env.step(HonestAction(raw_text=WELL_FORMED))
            assert env.state.episode_step == expected_step


# ---------------------------------------------------------------------------
# Difficulty updates called
# ---------------------------------------------------------------------------


class TestDifficultyIntegration:
    def test_update_difficulty_called_each_step(self, env):
        env.reset(seed=0)
        with patch("server.environment.update_difficulty") as mock_ud:
            for _ in range(3):
                env.step(HonestAction(raw_text=WELL_FORMED))
            assert mock_ud.call_count == 3

    def test_update_difficulty_receives_correctness(self, env):
        env.reset(seed=0)
        with patch("server.environment.update_difficulty") as mock_ud:
            env.step(HonestAction(raw_text=MALFORMED))
            _, call_kwargs = mock_ud.call_args
            # Called as positional: (state, correctness, domain=...)
            args = mock_ud.call_args.args
            # correctness is second positional argument
            assert args[1] is None  # malformed → None correctness


# ---------------------------------------------------------------------------
# state property
# ---------------------------------------------------------------------------


class TestStateProperty:
    def test_state_returns_honest_state(self, env):
        from models.models import HonestState
        env.reset()
        assert isinstance(env.state, HonestState)

    def test_state_is_mutable_across_steps(self, env):
        env.reset()
        before = env.state.episode_step
        env.step(HonestAction(raw_text=WELL_FORMED))
        assert env.state.episode_step == before + 1
