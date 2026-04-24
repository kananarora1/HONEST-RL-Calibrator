"""Action parsing, reward computation, and multi-reward functions for GRPO."""

import re
from typing import List, Optional, Tuple

from server.verifier import verify_answer

# ---------------------------------------------------------------------------
# Format patterns
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(
    r"<answer>(.*?)</answer>\s*<confidence>(.*?)</confidence>",
    re.DOTALL | re.IGNORECASE,
)
_ABSTAIN_RE = re.compile(r"<abstain\s*/>", re.IGNORECASE)


# ---------------------------------------------------------------------------
# parse_action
# ---------------------------------------------------------------------------


def parse_action(raw_text: str) -> dict:
    """Parse raw LLM output into a structured action dict.

    Possible returns:
    - {"type": "answer", "answer": str, "confidence": float}
    - {"type": "abstain"}
    - {"type": "malformed"}
    """
    text = raw_text.strip()

    # 1. Check for abstain first (no ambiguity)
    if _ABSTAIN_RE.search(text):
        return {"type": "abstain"}

    # 2. Try to extract answer + confidence
    m = _ANSWER_RE.search(text)
    if m:
        answer_str = m.group(1).strip()
        conf_str   = m.group(2).strip()

        # Fix: reject empty answer strings — a blank answer is not well-formed
        if not answer_str:
            return {"type": "malformed"}

        try:
            confidence = float(conf_str)
        except ValueError:
            return {"type": "malformed"}

        # Guard against inf/nan which can cause OverflowError downstream
        import math
        if not math.isfinite(confidence):
            return {"type": "malformed"}

        # Clamp confidence to [0, 1]
        confidence = max(0.0, min(1.0, confidence))

        return {
            "type":       "answer",
            "answer":     answer_str,
            "confidence": confidence,
        }

    # 3. Nothing matched → malformed
    return {"type": "malformed"}


# ---------------------------------------------------------------------------
# compute_reward  (single-scalar, used by environment and legacy code)
# ---------------------------------------------------------------------------

FORMAT_BONUS = 0.05   # raised from 0.02 — must be meaningful vs -0.5 malformed


def compute_reward(
    parsed: dict,
    ground_truth: str,
    difficulty: int,
) -> Tuple[float, Optional[bool]]:
    """Compute (reward, correctness_or_None) from a parsed action.

    Reward scheme
    -------------
    malformed                     : (-0.5,  None)
    abstain, difficulty >= 7      : ( 0.0,  None)
    abstain, difficulty <  7      : (-0.3,  None)
    answer                        : (brier_score + format_bonus, correct)

    Brier score component:
        brier = -((confidence - target) ** 2)
    where target = 1.0 if correct else 0.0.

    Format bonus = 0.05 for any well-formed answer tag.
    """
    action_type = parsed.get("type")

    if action_type == "malformed":
        return (-0.5, None)

    if action_type == "abstain":
        if difficulty >= 7:
            return (0.0, None)
        return (-0.3, None)

    if action_type == "answer":
        try:
            correct = verify_answer(parsed["answer"], ground_truth)
        except Exception:
            # Defensive: any verifier exception → treat as wrong
            correct = False
        target = 1.0 if correct else 0.0
        brier  = -((parsed["confidence"] - target) ** 2)
        reward = brier + FORMAT_BONUS
        return (reward, correct)

    # Fallback — treat unknown types as malformed
    return (-0.5, None)


# ---------------------------------------------------------------------------
# Multi-reward functions (TRL GRPOTrainer reward_funcs list)
#
# Each function receives:
#   completions   : List[str]  — raw model outputs for this step
#   prompts       : List[str]  — input prompts (passed by TRL)
#   ground_truth  : List[str]  — from the dataset column
#   difficulty    : List[int]  — from the dataset column
#   **kwargs      — any other dataset columns TRL passes through
#
# Magnitude budget (so Brier always dominates):
#   reward_brier      [-1.00, +0.05]  primary calibration signal
#   reward_format     [ 0.00, +0.05]  early-training compliance bonus
#   reward_accuracy   [ 0.00, +0.10]  correctness encouragement
#   reward_anti_hedge [-0.07,  0.00]  prevents always-0.5 collapse
# ---------------------------------------------------------------------------


def reward_brier(
    completions: List[str],
    prompts: List[str],
    ground_truth: List[str],
    difficulty: List[int],
    **kwargs,
) -> List[float]:
    """Primary reward: Brier-score calibration + format bonus.

    This is the dominant signal — its magnitude (up to ±1) dwarfs the others.
    Ground truth and difficulty come through as dataset columns, not a fragile
    string-keyed dict, so there is no tokeniser-drift lookup failure.
    """
    rewards = []
    for comp, gt, diff in zip(completions, ground_truth, difficulty):
        parsed = parse_action(comp)
        r, _   = compute_reward(parsed, str(gt), int(diff))
        rewards.append(float(r))
    return rewards


def reward_format(
    completions: List[str],
    **kwargs,
) -> List[float]:
    """Format compliance reward: +0.05 for well-formed output, 0.0 otherwise.

    Kept small so it cannot dominate a correct Brier response (~+0.05 max),
    but large enough to give the model a non-zero gradient signal during the
    first ~50 steps when most outputs are malformed.
    """
    rewards = []
    for comp in completions:
        parsed = parse_action(comp)
        if parsed["type"] in ("answer", "abstain"):
            rewards.append(0.05)
        else:
            rewards.append(0.0)
    return rewards


def reward_accuracy(
    completions: List[str],
    prompts: List[str],
    ground_truth: List[str],
    **kwargs,
) -> List[float]:
    """Correctness bonus: +0.10 if the answer is correct, 0.0 otherwise.

    Orthogonal to Brier: a model can get high Brier reward by saying
    confidence=0.01 when wrong; this adds a small push toward actually
    being right regardless of confidence.  Capped at 0.10 so it doesn't
    override calibration (a wrong answer with great calibration is still
    more valuable than a right answer with terrible calibration).
    """
    rewards = []
    for comp, gt in zip(completions, ground_truth):
        parsed = parse_action(comp)
        if parsed["type"] == "answer":
            try:
                correct = verify_answer(parsed["answer"], str(gt))
            except Exception:
                correct = False
            rewards.append(0.10 if correct else 0.0)
        else:
            rewards.append(0.0)
    return rewards


def reward_anti_hedge(
    completions: List[str],
    **kwargs,
) -> List[float]:
    """Anti-hedging penalty: -0.07 when confidence is in [0.45, 0.55].

    Without this, GRPO will quickly collapse to always outputting confidence≈0.5
    (Brier = −0.25, a locally stable but uninformative fixed point).
    This penalty breaks the symmetry and forces the model to commit.

    The dead zone [0.45, 0.55] is intentionally narrow — we don't want to
    penalise genuine uncertainty at 0.4 or 0.6.
    """
    rewards = []
    for comp in completions:
        parsed = parse_action(comp)
        if parsed["type"] == "answer" and 0.45 <= parsed["confidence"] <= 0.55:
            rewards.append(-0.07)
        else:
            rewards.append(0.0)
    return rewards
