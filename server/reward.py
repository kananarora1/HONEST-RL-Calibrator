"""Action parsing, reward computation, and multi-reward functions for GRPO."""

import re
from typing import Any, Dict, List, Optional, Tuple

from server.verifier import verify_answer


_ANSWER_RE = re.compile(
    r"<reasoning>(.*?)</reasoning>\s*<answer>(.*?)</answer>\s*<confidence>(.*?)</confidence>",
    re.DOTALL | re.IGNORECASE,
)

MALFORMED_PENALTY = -1.0
ABSTAIN_PENALTY   = 0.0
HINT_PENALTY      = -0.25
FORMAT_BONUS      = 0.15


def parse_action(raw_text: str) -> dict:
    if re.search(r'<request_hint\s*/?>', raw_text):
        return {"type": "hint"}
        
    if re.search(r'<abstain\s*/?>', raw_text):
        return {"type": "abstain"}
        
    m = _ANSWER_RE.search(raw_text)
    if m:
        reasoning_str = m.group(1).strip()
        answer_str    = m.group(2).strip()
        conf_str      = m.group(3).strip()
        
        # Require both reasoning and answer to prevent "Reasoning Bypass" cheat
        if not answer_str or not reasoning_str:
            return {"type": "malformed"}
            
        try:
            confidence = float(conf_str)
            confidence = max(0.0, min(1.0, confidence)) # Clamp
            return {
                "type": "answer",
                "answer": answer_str,
                "confidence": confidence
            }
        except ValueError:
            return {"type": "malformed"}
            
    return {"type": "malformed"}


def _verify(
    model_answer: str,
    ground_truth: str,
    problem_id: Optional[str],
    domain: Optional[str],
    verification_metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Route to the domain-aware verifier."""
    if problem_id and problem_id.startswith("procedural_"):
        return verify_answer(model_answer, ground_truth, domain=None, verification_metadata=verification_metadata)

    if problem_id:
        try:
            from data.sampler.unified_sampler import get_sampler
            return get_sampler().verify(problem_id, model_answer)
        except Exception:
            pass
    return verify_answer(model_answer, ground_truth, domain=domain, verification_metadata=verification_metadata)


def compute_reward(
    parsed: dict,
    ground_truth: str,
    difficulty: int,
    problem_id: Optional[str] = None,
    domain: Optional[str] = None,
    verification_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Optional[bool]]:
    """
    Compute (reward, correctness_or_None) from a parsed action.
    """
    action_type = parsed.get("type")

    if action_type == "malformed":
        return (MALFORMED_PENALTY, None)

    if action_type == "hint":
        return (HINT_PENALTY, None)

    if action_type == "abstain":
        if difficulty >= 7:
            return (0.0, None)
        return (ABSTAIN_PENALTY, None)

    if action_type == "answer":
        try:
            correct = _verify(
                parsed["answer"],
                ground_truth,
                problem_id,
                domain,
                verification_metadata,
            )
        except Exception:
            correct = False
            
        target = 1.0 if correct else 0.0
        
        brier = -1.5 * ((parsed["confidence"] - target) ** 2)
        reward = brier + FORMAT_BONUS
        
        return (reward, correct)

    return (MALFORMED_PENALTY, None)


"""
Multi-reward functions for TRL GRPOTrainer.

Smoothed Magnitude budget (Total bounds ~ [-1.50, +1.00]):
  reward_brier      [-1.50, +0.15]  Primary calibration signal (dampened)
  reward_format     [ 0.00, +0.15]  Early-training compliance bonus
  reward_accuracy   [-0.15, +0.85]  Correctness bonus / Incorrect penalty
"""

def reward_brier(
    completions: List[str],
    prompts: List[str],
    ground_truth: List[str],
    difficulty: List[int],
    **kwargs,
) -> List[float]:
    rewards = []
    pid_list = kwargs.get("problem_id", [None] * len(completions))
    domains = kwargs.get("domain", [None] * len(completions))
    verification_metadatas = kwargs.get("verification_metadata", [{}] * len(completions))

    for idx, (comp, gt, diff) in enumerate(zip(completions, ground_truth, difficulty)):
        domain = domains[idx] if isinstance(domains, list) and idx < len(domains) else None
        pid = pid_list[idx] if isinstance(pid_list, list) and idx < len(pid_list) else None
        v_meta = verification_metadatas[idx] if isinstance(verification_metadatas, list) and idx < len(verification_metadatas) else None
        
        parsed = parse_action(comp)
        r, _ = compute_reward(
            parsed,
            str(gt),
            int(diff),
            problem_id=pid,
            domain=domain,
            verification_metadata=v_meta,
        )
        rewards.append(float(r))
    return rewards


def reward_format(
    completions: List[str],
    **kwargs,
) -> List[float]:
    """Format compliance reward: +0.15 for well-formed output, 0.0 otherwise."""
    rewards = []
    for comp in completions:
        parsed = parse_action(comp)
        if parsed["type"] in ("answer", "abstain", "hint"):
            rewards.append(0.15)
        else:
            rewards.append(0.0)
    return rewards


def reward_accuracy(
    completions: List[str],
    prompts: List[str],
    ground_truth: List[str],
    **kwargs,
) -> List[float]:
    """Correctness bonus: +0.85 if correct, -0.15 if incorrect."""
    rewards = []
    pid_list = kwargs.get("problem_id", [None] * len(completions))
    domains = kwargs.get("domain", [None] * len(completions))
    verification_metadatas = kwargs.get("verification_metadata", [{}] * len(completions))
    
    for idx, (comp, gt) in enumerate(zip(completions, ground_truth)):
        domain = domains[idx] if isinstance(domains, list) and idx < len(domains) else None
        pid = pid_list[idx] if isinstance(pid_list, list) and idx < len(pid_list) else None
        v_meta = verification_metadatas[idx] if isinstance(verification_metadatas, list) and idx < len(verification_metadatas) else None

        parsed = parse_action(comp)
        if parsed["type"] == "answer":
            try:
                correct = _verify(parsed["answer"], str(gt), pid, domain, v_meta)
            except Exception:
                correct = False
            rewards.append(0.85 if correct else -0.15)
        else:
            rewards.append(0.0)
    return rewards

# NOTE: reward_anti_hedge has been DELETED to prevent the 0.7 confidence exploit.