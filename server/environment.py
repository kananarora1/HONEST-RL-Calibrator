"""HonestEnvironment — main environment class for the HONEST calibration benchmark."""

import logging
import random
import uuid
from typing import Any, Optional

from models.models import HonestAction, HonestObservation, HonestState
from openenv.core.env_server.interfaces import Environment
from server.difficulty import DifficultyController, update_difficulty
from server.generators import code_gen, logic_gen, math_gen
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
        self._generators = {
            "math": math_gen.generate,
            "code": code_gen.generate,
            "logic": logic_gen.generate,
        }
        # Adaptive difficulty controller — persists across reset() calls.
        self.difficulty_controller = DifficultyController(domains=list(DOMAINS))
        self._current_question: Optional[str] = None
        self._current_answer: Optional[str] = None

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
        question, answer = self._generators[domain](difficulty, seed=seed)
        self._current_question = question
        self._current_answer = answer

        logger.info(
            "reset: episode_id=%s domain=%s difficulty=%d",
            ep_id,
            domain,
            difficulty,
        )

        return HonestObservation(
            question=question,
            domain=domain,
            difficulty=difficulty,
            episode_step=0,
            done=False,
            reward=None,
        )

    # In server/environment.py -> step()

    def step(
        self,
        action: HonestAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> HonestObservation:
        """Process one agent action and advance the environment."""
        domain = self._state.current_domain
        difficulty = self._state.domain_difficulties[domain]

        parsed = parse_action(action.raw_text)
        
        # Handle <request_hint/>
        if parsed.get("type") == "hint":
            reward_value = -0.05  # The cost of information
            correctness = None
            self._state.hints_revealed += 1
            
            # Append a generic hint (in a v2, generators could provide domain-specific hints)
            self._current_question += f"\n\n[System Hint {self._state.hints_revealed}: Review your assumptions. Break the problem down into smaller discrete steps.]"
            
            self._state.episode_step += 1
            terminal = self._state.episode_step >= EPISODE_LENGTH
            
            # Record action in history
            self._state.episode_history.append({
                "action": "requested_hint",
                "reward": reward_value,
                "domain": domain,
                "difficulty": difficulty,
            })
            
            # Return observation BUT stay on the current problem
            return HonestObservation(
                question=self._current_question,
                domain=domain,
                difficulty=difficulty,
                episode_step=self._state.episode_step,
                previous_correctness=None,
                terminal=terminal,
                done=terminal,
                reward=reward_value,
            )

        # STANDARD MDP LOGIC: Handle <answer> or <abstain>
        reward_value, correctness = compute_reward(
            parsed,
            self._current_answer,
            difficulty,
            domain=domain,
        )

        # Append the single authoritative history record for this step.
        # update_difficulty() will read it (for rolling accuracy) and must
        # NOT append anything itself.
        step_record: dict = {
            "question": self._current_question,
            "ground_truth": self._current_answer,
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

        # Legacy per-episode scalar update (still consumed by some tests).
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
            return HonestObservation(
                question="",
                domain=domain,
                difficulty=difficulty,
                episode_step=self._state.episode_step,
                previous_correctness=correctness,
                terminal=True,
                done=True,
                reward=reward_value,
            )

        # Pick next problem — domain uniformly random; difficulty from controller.
        next_domain = random.choice(DOMAINS)
        self._state.current_domain = next_domain
        next_difficulty = self.difficulty_controller.sample_difficulty(next_domain)
        self._state.domain_difficulties[next_domain] = next_difficulty
        next_question, next_answer = self._generators[next_domain](next_difficulty)
        
        self._current_question = next_question
        self._current_answer = next_answer
        self._state.hints_revealed = 0 # Reset hints for the new problem

        return HonestObservation(
            question=next_question,
            domain=next_domain,
            difficulty=next_difficulty,
            episode_step=self._state.episode_step,
            previous_correctness=correctness,
            terminal=False,
            done=False,
            reward=reward_value,
        )

    # state property
    @property
    def state(self) -> HonestState:
        return self._state
