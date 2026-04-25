"""HonestEnvironment — main environment class for the HONEST calibration benchmark.

Generators are now backed by the **unified sampler** in ``data/sampler/`` so the
environment serves curated problems (Hendrycks MATH, MBPP+APPS, ZebraLogic) at
the difficulty chosen by the adaptive ``DifficultyController``. Procedural-only
fallback for logic d=1,2 is handled inside the unified sampler itself.

The environment threads the sampler's stable ``problem_id`` through to
``server.reward.compute_reward``, which dispatches to the unified ``verify()``
when the id is from the curated dataset and falls back to the local verifier
for procedural problems.
"""

import logging
import random
import uuid
from typing import Any, Optional

from data.sampler.environment_adapter import (
    code_generate,
    logic_generate,
    math_generate,
)
from models.models import HonestAction, HonestObservation, HonestState
from openenv.core.env_server.interfaces import Environment
from server.difficulty import DifficultyController, update_difficulty
from server.reward import compute_reward, parse_action

logger = logging.getLogger(__name__)

DOMAINS = ["math", "code", "logic"]
EPISODE_LENGTH = 5
INITIAL_DIFFICULTIES = {"math": 1, "code": 1, "logic": 1}


class HonestEnvironment(Environment):
    """HONEST: Honesty-Optimised and Normalized Environment for Self-Triage.

    Each episode presents the agent with a sequence of questions drawn from
    three domains (math, code, logic) at adaptively-chosen difficulty levels.
    The agent must respond with an <answer>/<confidence> pair or <abstain/>.
    Rewards are computed using the Brier-score calibration scheme.
    """

    # All mutable state lives inside self._state — no class-level shared state.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state: HonestState = HonestState(episode_id="")
        # Unified-sampler-backed generators; signature matches the procedural ones
        # but returns (question, canonical_answer, problem_id).
        self._generators = {
            "math":  math_generate,
            "code":  code_generate,
            "logic": logic_generate,
        }
        # Adaptive difficulty controller — persists across reset() calls so the
        # curriculum adapts over the full lifetime of the environment instance.
        self.difficulty_controller = DifficultyController(domains=list(DOMAINS))
        self._current_question: Optional[str] = None
        self._current_answer: Optional[str] = None
        self._current_problem_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _generate_problem(
        self,
        domain: str,
        difficulty: int,
        seed: Optional[int] = None,
    ) -> tuple[str, str, str]:
        """Call the unified sampler for *domain* at *difficulty*.

        Returns ``(question, canonical_answer, problem_id)``. On unexpected
        failure the call is retried at decreasing difficulty so the env never
        serves an empty observation.
        """
        for diff_try in (difficulty, max(1, difficulty - 1), 1):
            try:
                question, answer, pid = self._generators[domain](diff_try, seed=seed)
                if diff_try != difficulty:
                    logger.warning(
                        "Generator(%s, d=%d) failed; fell back to d=%d.",
                        domain, difficulty, diff_try,
                    )
                return question, answer, pid
            except Exception as exc:
                logger.warning(
                    "Generator(%s, d=%d) raised: %s — retrying at lower difficulty.",
                    domain, diff_try, exc,
                )
        # As a last resort produce a trivial math problem so the env stays alive.
        question, answer, pid = math_generate(1, seed=seed)
        return question, answer, pid

    def _refresh_controller_snapshot(self) -> None:
        """Mirror the current controller snapshot onto self._state for observers."""
        try:
            self._state.difficulty_controller_state = self.difficulty_controller.snapshot()
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("snapshot() failed: %s", exc)

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> HonestObservation:
        """Start a new episode and return the first observation."""
        ep_id = episode_id or str(uuid.uuid4())

        self._state = HonestState(
            episode_id=ep_id,
            domain_difficulties=dict(INITIAL_DIFFICULTIES),
            episode_step=0,
            episode_history=[],
        )

        rng = random.Random(seed) if seed is not None else random
        domain = rng.choice(DOMAINS)
        self._state.current_domain = domain

        # Pull the difficulty for this episode from the adaptive controller.
        # Snapshot the chosen value back into state so the rest of the env
        # (reward, history, observation) reads a consistent scalar.
        difficulty = self.difficulty_controller.sample_difficulty(domain, rng=rng)
        self._state.domain_difficulties[domain] = difficulty
        question, answer, problem_id = self._generate_problem(domain, difficulty, seed=seed)
        self._current_question = question
        self._current_answer = answer
        self._current_problem_id = problem_id
        self._state.current_problem_id = problem_id
        self._refresh_controller_snapshot()

        logger.info(
            "reset: episode_id=%s domain=%s difficulty=%d problem_id=%s",
            ep_id,
            domain,
            difficulty,
            problem_id,
        )

        return HonestObservation(
            question=question,
            domain=domain,
            difficulty=difficulty,
            episode_step=0,
            done=False,
            reward=None,
            problem_id=problem_id,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self,
        action: HonestAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> HonestObservation:
        """Process one agent action and advance the environment."""
        domain = self._state.current_domain
        difficulty = self._state.domain_difficulties[domain]
        problem_id = self._current_problem_id

        parsed = parse_action(action.raw_text)

        # Handle <request_hint/>
        if parsed.get("type") == "hint":
            reward_value = -0.05  # The cost of information
            correctness = None
            self._state.hints_revealed += 1

            # Append a generic hint (in a v2, generators could provide domain-specific hints)
            self._current_question += (
                f"\n\n[System Hint {self._state.hints_revealed}: Review your assumptions. "
                f"Break the problem down into smaller discrete steps.]"
            )

            self._state.episode_step += 1
            terminal = self._state.episode_step >= EPISODE_LENGTH

            self._state.episode_history.append({
                "action": "requested_hint",
                "reward": reward_value,
                "domain": domain,
                "difficulty": difficulty,
                "problem_id": problem_id,
            })

            # Stay on the current problem.
            return HonestObservation(
                question=self._current_question,
                domain=domain,
                difficulty=difficulty,
                episode_step=self._state.episode_step,
                previous_correctness=None,
                terminal=terminal,
                done=terminal,
                reward=reward_value,
                problem_id=problem_id,
            )

        # Standard MDP path: <answer> or <abstain> or malformed.
        reward_value, correctness = compute_reward(
            parsed,
            self._current_answer,
            difficulty,
            problem_id=problem_id,
            domain=domain,
        )

        # Append the single authoritative history record for this step.
        # update_difficulty() reads it (for rolling accuracy) and must NOT
        # append anything itself.
        step_record: dict = {
            "question": self._current_question,
            "ground_truth": self._current_answer,
            "problem_id": problem_id,
            "parsed": parsed,
            "correct": correctness,
            "reward": reward_value,
            "domain": domain,
            "difficulty": difficulty,
            "hints_used": self._state.hints_revealed,
            "difficulty_changed": False,
        }
        self._state.episode_history.append(step_record)

        self._state.episode_step += 1

        # Legacy per-episode scalar update — retained as a no-op compatibility
        # shim for tests that mock it. The real adaptive logic lives in the
        # DifficultyController below.
        difficulty_update = update_difficulty(self._state, correctness, domain=domain)
        if isinstance(difficulty_update, tuple) and len(difficulty_update) >= 2:
            diff_changed = bool(difficulty_update[1])
        else:
            diff_changed = False
        if diff_changed:
            step_record["difficulty_changed"] = True

        # Adaptive controller — only record real Answer outcomes (skip
        # abstain / hint / malformed so they do not pollute the window).
        if correctness is not None:
            new_target, controller_changed = self.difficulty_controller.record_outcome(
                domain, bool(correctness)
            )
            if controller_changed:
                logger.info(
                    "Difficulty controller: %s target now %d (rolling acc %.2f)",
                    domain,
                    new_target,
                    self.difficulty_controller.get_rolling_accuracy(domain) or 0.0,
                )

        terminal = self._state.episode_step >= EPISODE_LENGTH

        logger.info(
            "step %d: domain=%s difficulty=%d parsed_type=%s reward=%.4f correct=%s terminal=%s",
            self._state.episode_step,
            domain,
            difficulty,
            parsed.get("type"),
            reward_value,
            correctness,
            terminal,
        )

        if terminal:
            self._refresh_controller_snapshot()
            return HonestObservation(
                question="",
                domain=domain,
                difficulty=difficulty,
                episode_step=self._state.episode_step,
                previous_correctness=correctness,
                terminal=True,
                done=True,
                reward=reward_value,
                problem_id=problem_id,
            )

        # Pick next problem — domain uniformly random; difficulty from controller.
        next_domain = random.choice(DOMAINS)
        self._state.current_domain = next_domain
        next_difficulty = self.difficulty_controller.sample_difficulty(next_domain)
        self._state.domain_difficulties[next_domain] = next_difficulty
        next_question, next_answer, next_problem_id = self._generate_problem(
            next_domain, next_difficulty
        )

        self._current_question = next_question
        self._current_answer = next_answer
        self._current_problem_id = next_problem_id
        self._state.current_problem_id = next_problem_id
        self._state.hints_revealed = 0  # Reset hints for the new problem
        self._refresh_controller_snapshot()

        return HonestObservation(
            question=next_question,
            domain=next_domain,
            difficulty=next_difficulty,
            episode_step=self._state.episode_step,
            previous_correctness=correctness,
            terminal=False,
            done=False,
            reward=reward_value,
            problem_id=next_problem_id,
        )

    # state property
    @property
    def state(self) -> HonestState:
        return self._state
