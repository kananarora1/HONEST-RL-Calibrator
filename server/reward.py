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


# ---------------------------------------------------------------------------
# Lenient EVAL-ONLY parser
#
# Training rewards MUST continue to use the strict `parse_action` above — it
# is the anti-cheat gate that prevents a "reasoning bypass" exploit. This
# lenient variant is intended ONLY for evaluation (e.g. OOD MCQ datasets
# where the model may emit prose instead of the exact XML contract). It never
# loosens anything for the training pipeline.
#
# Behaviour:
#   1. Strict parser first. If it succeeds, return its result with
#      parsed_mode="strict" (untouched semantics).
#   2. Otherwise try a best-effort recovery:
#        - <answer>...</answer> tag
#        - "Answer: X" / "final answer is X" / "option (X) is correct"
#        - last "(A)"-style MCQ marker
#      and for confidence:
#        - <confidence>...</confidence>
#        - "80% confident" / "confidence: 0.8" / "I am 0.8 confident"
#   3. If no answer is recoverable → still "malformed" (fail closed).
#   4. If an answer is recovered but no confidence → neutral prior 0.5 and
#      parsed_mode="lenient_default_conf", so downstream calibration metrics
#      stay honest (the reward path below never grants the +0.15 FORMAT_BONUS
#      to lenient-parsed answers).
# ---------------------------------------------------------------------------

_LENIENT_ANSWER_TAG_RE = re.compile(
    r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE
)
_LENIENT_CONF_TAG_RE = re.compile(
    r"<confidence>\s*([0-9]*\.?[0-9]+)\s*</confidence>", re.IGNORECASE
)
_LENIENT_ABSTAIN_RE = re.compile(r"<abstain\s*/?>", re.IGNORECASE)

# Ordered list of answer-recovery patterns. Each must have exactly one capture
# group containing the raw answer token.
_LENIENT_ANSWER_PATTERNS = [
    re.compile(r"(?im)^\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|\-)\s*\(?([A-Za-z0-9][^\n\.\)]{0,40}?)\)?\s*(?:\.|$|\n)"),
    re.compile(r"(?im)\b(?:final\s+)?answer\s+is\s*\(?([A-E])\)?\b"),
    re.compile(r"(?im)\bcorrect\s+(?:answer|option|choice)\s+is\s*\(?([A-E])\)?\b"),
    re.compile(r"(?im)\boption\s+\(?([A-E])\)?\s+is\s+correct\b"),
]
# MCQ letter-in-parens fallback (e.g. "...the correct choice is (A).").
_LENIENT_PARENS_LETTER_RE = re.compile(r"\(\s*([A-E])\s*\)")

# Confidence recovery, in priority order.
_LENIENT_CONF_PCT_RE = re.compile(r"([0-9]{1,3}(?:\.[0-9]+)?)\s*%\s*confiden", re.IGNORECASE)
_LENIENT_CONF_KV_RE = re.compile(
    r"(?i)confiden(?:ce|t)\s*(?:is|:|=|of|level)?\s*([0-9]*\.?[0-9]+)\s*(%?)"
)


def _extract_answer_lenient(text: str) -> Optional[str]:
    m = _LENIENT_ANSWER_TAG_RE.search(text)
    if m and m.group(1).strip():
        return m.group(1).strip()

    for pat in _LENIENT_ANSWER_PATTERNS:
        matches = list(pat.finditer(text))
        if matches:
            cand = matches[-1].group(1).strip().rstrip(".,;:)")
            if cand:
                return cand

    parens = list(_LENIENT_PARENS_LETTER_RE.finditer(text))
    if parens:
        return parens[-1].group(1).upper()

    return None


def _extract_confidence_lenient(text: str) -> Tuple[float, bool]:
    """Return (confidence, found_explicit). If not found, returns (0.5, False)."""
    m = _LENIENT_CONF_TAG_RE.search(text)
    if m:
        try:
            return float(m.group(1)), True
        except ValueError:
            pass

    m = _LENIENT_CONF_PCT_RE.search(text)
    if m:
        try:
            return float(m.group(1)) / 100.0, True
        except ValueError:
            pass

    m = _LENIENT_CONF_KV_RE.search(text)
    if m:
        try:
            v = float(m.group(1))
            if m.group(2) == "%" or v > 1.0:
                v = v / 100.0
            return v, True
        except ValueError:
            pass

    return 0.5, False


def parse_action_lenient(raw_text: str) -> dict:
    """Eval-only parser. Tries strict ``parse_action`` first; on malformed,
    attempts a best-effort recovery without weakening the training contract.

    Returned dicts always contain a ``parsed_mode`` key for bookkeeping:
        "strict"                - strict regex matched (identical to parse_action)
        "lenient"               - recovered answer AND explicit confidence
        "lenient_default_conf"  - recovered answer, no explicit confidence (0.5)
    Malformed results do not carry parsed_mode.
    """
    if raw_text is None:
        return {"type": "malformed"}

    strict = parse_action(raw_text)
    if strict["type"] != "malformed":
        out = dict(strict)
        out["parsed_mode"] = "strict"
        return out

    if _LENIENT_ABSTAIN_RE.search(raw_text):
        return {"type": "abstain", "parsed_mode": "strict"}

    ans = _extract_answer_lenient(raw_text)
    if not ans:
        return {"type": "malformed"}

    conf, explicit = _extract_confidence_lenient(raw_text)
    conf = max(0.0, min(1.0, float(conf)))

    return {
        "type": "answer",
        "answer": ans,
        "confidence": conf,
        "parsed_mode": "lenient" if explicit else "lenient_default_conf",
    }