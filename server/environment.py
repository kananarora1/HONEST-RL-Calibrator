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
from openenv.core.env_server.types import EnvironmentMetadata
from server.difficulty import DifficultyController, update_difficulty
from server.hindsight import (
    HindsightCoordinator,
    compute_hindsight_reward,
    parse_hindsight,
)
from server.reward import compute_reward, parse_action

logger = logging.getLogger(__name__)

DOMAINS = ["math", "code", "logic"]
EPISODE_LENGTH = 5
INITIAL_DIFFICULTIES = {"math": 1, "code": 1, "logic": 1}

# Hindsight slot prompt prefix injected into the *next* observation when
# ``HindsightCoordinator.maybe_request`` returned True. Kept short so it
# doesn't dominate the question budget on small models.
HINDSIGHT_PROMPT_PREFIX = (
    "[Hindsight slot] You answered the previous problem with confidence "
    "{c_prev:.2f}. The correct answer was: {gt}. Given this outcome, what "
    "confidence *should* you have expressed? Respond ONLY with "
    "<hindsight>0.XX</hindsight>.\n\n"
)


class HonestEnvironment(Environment):
    """HONEST: Honesty-Optimised and Normalized Environment for Self-Triage.

    Each episode presents the agent with a sequence of questions drawn from
    three domains (math, code, logic) at adaptively-chosen difficulty levels.
    The agent must respond with an <answer>/<confidence> pair or <abstain/>.
    Rewards are computed using the Brier-score calibration scheme.
    """

    # All mutable state lives inside self._state — no class-level shared state.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        hindsight_probability: float = 0.0,
        hindsight_weight: float = 0.3,
        smc=None,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        hindsight_probability:
            Probability of injecting a hindsight slot after each AnswerAction.
            ``0.0`` (default) preserves the legacy behaviour. See
            ``server.hindsight`` and ``SELF_LEARNING.md §2``.
        hindsight_weight:
            ``k`` in the hindsight reward ``R_h = -k(r-y)^2``.
        smc:
            Optional ``SelfMutatingCurriculum`` (Pillar 3). When provided,
            ``_generate_problem`` is routed through it so difficulty values
            above ``DIFFICULTY_MAX`` automatically produce mutated problems.
            ``None`` (default) preserves the legacy unified-sampler path.
        """
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

        # Pillar 1: hindsight coordinator (probability=0 disables the slot
        # entirely so legacy clients are unaffected).
        self.hindsight = HindsightCoordinator(probability=hindsight_probability)
        self.hindsight_weight = float(hindsight_weight)

        # Pillar 3: self-mutating curriculum. None = legacy path.
        self.smc = smc

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

        If a ``SelfMutatingCurriculum`` is attached (Pillar 3), difficulty
        values above the controller's hard MAX are routed through the
        mutator chain. Otherwise we use the legacy retry-on-fallback path.

        Returns ``(question, canonical_answer, problem_id)``. On unexpected
        failure the call is retried at decreasing difficulty so the env never
        serves an empty observation.
        """
        if self.smc is not None and self.smc.is_above_base(domain, difficulty):
            try:
                rng = random.Random(seed) if seed is not None else None
                return self.smc.sample(domain, difficulty, rng=rng)
            except Exception as exc:
                logger.warning("SMC.sample(%s, d=%d) raised: %s — falling back to base.",
                               domain, difficulty, exc)

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
    # OpenEnv runtime metadata
    # ------------------------------------------------------------------

    def get_metadata(self) -> EnvironmentMetadata:
        """Public environment description served at ``GET /metadata``.

        Overriding the default implementation populates the OpenEnv runtime
        contract (``name`` / ``description`` are validated by
        ``openenv validate``) and surfaces a structured one-liner that
        Spaces UIs and the OpenAPI schema can consume directly.
        """
        return EnvironmentMetadata(
            name="HONEST-Env",
            description=(
                "Honesty-Optimised and Normalized Environment for Self-Triage — "
                "an OpenEnv calibration benchmark across math, code, and logic "
                "where agents must report a confidence (or abstain) under a "
                "Brier-score reward."
            ),
            version="0.1.0",
            author="HONEST-Env Contributors",
            documentation_url="https://github.com/Rushhaabhhh/HONEST-RL-Calibrator",
        )

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

        # Pillar 1: if a hindsight slot is in flight from the previous step,
        # consume it FIRST — it is a meta-step that does not advance the
        # episode_step counter and reuses the existing reveal context.
        if self.hindsight.pending():
            return self._step_hindsight(action, domain, difficulty, problem_id)

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
        # Capture the GT *before* we move on to the next problem so we can
        # surface it as `revealed_answer` in the returned observation
        # (required for the hindsight calibration path — see server/hindsight.py).
        revealed_answer = self._current_answer
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
                # Reveal GT only when the agent actually answered (correctness
                # is not None). Abstain / malformed do not reveal — preserves
                # the rule "no reveal if the agent didn't commit".
                revealed_answer=revealed_answer if correctness is not None else None,
                terminal=True,
                done=True,
                reward=reward_value,
                problem_id=problem_id,
            )

        # Pillar 1: decide whether to inject a hindsight slot AS the next
        # observation. If yes, we *do not* advance to a new problem — the
        # slot reuses the just-completed (q, gt) context.
        c_prev = parsed["confidence"] if parsed.get("type") == "answer" else None
        rng_for_hindsight = random.Random()
        request_slot = self.hindsight.maybe_request(correctness, c_prev, rng=rng_for_hindsight)
        if request_slot:
            slot_q = HINDSIGHT_PROMPT_PREFIX.format(
                c_prev=(c_prev if c_prev is not None else 0.5),
                gt=revealed_answer if revealed_answer is not None else "(undisclosed)",
            )
            return HonestObservation(
                question=slot_q,
                domain=domain,
                difficulty=difficulty,
                episode_step=self._state.episode_step,
                previous_correctness=correctness,
                revealed_answer=revealed_answer,
                terminal=False,
                done=False,
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
            # Reveal GT of the *just-completed* problem (not the new one) when
            # the agent actually answered. Powers the optional hindsight signal.
            revealed_answer=revealed_answer if correctness is not None else None,
            terminal=False,
            done=False,
            reward=reward_value,
            problem_id=next_problem_id,
        )

    # ------------------------------------------------------------------
    # Hindsight slot handler (Pillar 1)
    # ------------------------------------------------------------------

    def _step_hindsight(
        self,
        action: HonestAction,
        domain: str,
        difficulty: int,
        problem_id: Optional[str],
    ) -> HonestObservation:
        """Handle a step where the previous reveal asked for hindsight.

        Grades the retrospective confidence and then continues to a brand
        new problem. Crucially we *do not* increment ``episode_step`` here:
        the hindsight slot is a meta-step on top of the just-completed
        problem, not a fresh problem. Episode termination is therefore
        determined by ``episode_step`` only, never by the slot.
        """
        active, y, c_prev = self.hindsight.consume()
        # active should always be True here — guarded by the caller.

        parsed = parse_hindsight(action.raw_text)

        if parsed["type"] != "hindsight" or y is None:
            # Soft penalty: the model emitted something other than a
            # well-formed hindsight despite the slot being open.
            reward_value = -0.1
            retrospective = None
        else:
            retrospective = parsed["retrospective"]
            reward_value = compute_hindsight_reward(
                retrospective,
                bool(y),
                weight=self.hindsight_weight,
            )

        self._state.episode_history.append({
            "kind": "hindsight",
            "previous_correctness": y,
            "retrospective": retrospective,
            "previous_confidence": c_prev,
            "reward": reward_value,
            "domain": domain,
            "difficulty": difficulty,
            "problem_id": problem_id,
        })

        # Advance to a new problem (not just terminal of current).
        next_domain = random.choice(DOMAINS)
        self._state.current_domain = next_domain
        next_difficulty = self.difficulty_controller.sample_difficulty(next_domain)
        self._state.domain_difficulties[next_domain] = next_difficulty
        nq, na, npid = self._generate_problem(next_domain, next_difficulty)

        self._current_question = nq
        self._current_answer = na
        self._current_problem_id = npid
        self._state.current_problem_id = npid
        self._state.hints_revealed = 0
        self._refresh_controller_snapshot()

        terminal = self._state.episode_step >= EPISODE_LENGTH
        return HonestObservation(
            question=nq if not terminal else "",
            domain=next_domain,
            difficulty=next_difficulty,
            episode_step=self._state.episode_step,
            previous_correctness=bool(y) if y is not None else None,
            revealed_answer=None,  # the prior reveal is consumed
            terminal=terminal,
            done=terminal,
            reward=reward_value,
            problem_id=npid,
        )

    # state property
    @property
    def state(self) -> HonestState:
        return self._state
