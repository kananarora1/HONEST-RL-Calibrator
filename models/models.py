from typing import Any, Dict, List, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class HonestAction(Action):
    raw_text: str


class HonestObservation(Observation):
    question: str
    domain: str
    difficulty: int
    episode_step: int
    previous_correctness: Optional[bool] = None
    revealed_answer: Optional[str] = None
    # Stable id of the served problem (curated dataset id, or "procedural_<domain>_dN_..." for
    # generated problems). Used downstream by the unified-sampler verify() dispatcher.
    problem_id: Optional[str] = None
    # terminal mirrors Observation.done; kept separate for semantic clarity
    terminal: bool = False


class HonestState(State):
    current_domain: str = ""
    domain_difficulties: Dict[str, int] = Field(default_factory=dict)
    episode_step: int = 0
    episode_history: List[Any] = Field(default_factory=list)
    hints_revealed: int = 0
    # Latest snapshot of the adaptive DifficultyController (per-domain target, rolling
    # accuracy, sampling distribution). Refreshed at the end of every step() so external
    # consumers (eval, dashboards) can read curriculum state without poking internals.
    difficulty_controller_state: Dict[str, Any] = Field(default_factory=dict)
    # Stable id of the currently served problem; mirrors HonestObservation.problem_id.
    current_problem_id: Optional[str] = None
