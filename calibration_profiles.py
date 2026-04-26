"""Shared calibration profiles for training and evaluation.

These presets make cross-model comparisons fair by standardizing:
1) prompt style (reasoning mode),
2) data mixture (domain + difficulty weights),
3) model-aware defaults for GRPO knobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


SUPPORTED_PRESETS = ("qwen7b", "llama3b", "phi4mini")
REASONING_MODES = ("required",)


@dataclass(frozen=True)
class CalibrationPreset:
    """Per-model calibration preset.

    Splits cleanly into three blocks:
      1) Data composition  (domain + difficulty mixture, dataset size).
      2) GRPO defaults     (model-aware sampling / optimization knobs).
      3) Reward & KL plan  (auxiliary weights, KL-beta schedule, controller
         seed).

    All values are research-justified for short calibration RL runs
    (≤ 400 GRPO steps, single LoRA stage). They are NOT general SFT
    defaults — they are tuned for stable Brier-score gradients on
    Qwen-7B / Llama-3B / Phi-4-mini under the committed reward scheme
    in ``server/reward.py`` (Brier scale -1.5, FORMAT_BONUS 0.15,
    accuracy bonus +0.85 / -0.15).

    NOTE on anti-hedge: the project deliberately *removed* the
    anti-hedge auxiliary in commit ``3690671`` to plug a 0.7-confidence
    exploit (the model could sit just outside the [0.4, 0.6] band and
    avoid the penalty while still hedging). We intentionally do NOT
    add it back here — calibration is shaped purely by the Brier
    gradient + accuracy bonus, with KL keeping the policy honest.
    """

    name: str
    model_hint: str

    # --- Data composition ---------------------------------------------------
    domain_weights: Dict[str, float]
    difficulty_weights: Dict[int, float]
    default_prompt_dataset_size: int

    # --- GRPO defaults ------------------------------------------------------
    default_num_generations: int
    default_max_completion_length: int
    default_temperature: float
    default_learning_rate: float
    default_beta: float
    default_lora_r: int
    default_max_steps: int

    # --- Reward composition -------------------------------------------------
    # Primary signal is `make_brier_with_curriculum_feedback` (weight 1.0,
    # hard-coded — combines Brier reward + curriculum feedback in one tap).
    # Auxiliaries below are independent reward functions weighted per-preset.
    reward_format_weight: float    # multiplier on +0.15 format bonus
    reward_accuracy_weight: float  # multiplier on the +0.85/-0.15 correctness reward

    # --- KL schedule (start tight, relax after stabilization) --------------
    # beta_start kicks in at step 0 (prevents early policy explosion);
    # beta_end takes over after kl_relax_frac × max_steps (allows
    # calibration consolidation in the second half of training).
    beta_end: float
    kl_relax_frac: float

    # --- Adaptive difficulty controller -----------------------------------
    # Initial per-domain target_difficulty for DifficultyController. Stronger
    # base models benefit from starting at 2 (the bulk of curriculum signal
    # lives at diff 2-3); weaker models need to start at 1 to avoid an
    # all-zero rolling accuracy that would force the controller to plateau
    # at MIN_DIFFICULTY without ever exploring the easy band's calibration.
    default_initial_target: int


MODEL_PRESETS: Dict[str, CalibrationPreset] = {
    # ─────────────────────────────────────────────────────────────────────
    # Qwen2.5-7B-Instruct  →  A100 40 GB
    # Strong baseline reasoning (~60 % acc on diff-2/3) → Brier gradient is
    # rich from early steps. We can afford fewer rollouts (G=8) and a
    # slightly higher beta (0.05) because the policy doesn't drift much.
    # Format weight kept at 1.0 because Qwen format-compliance locks in
    # within ~30 steps; over-weighting format would crowd out the Brier
    # gradient. Accuracy weight 1.0 keeps the +0.85/-0.15 correctness
    # incentive symmetric with the Brier signal magnitude (-1.5..+0.15).
    # ─────────────────────────────────────────────────────────────────────
    "qwen7b": CalibrationPreset(
        name="qwen7b",
        model_hint="Qwen/Qwen2.5-7B-Instruct",
        domain_weights={"math": 0.50, "code": 0.35, "logic": 0.15},
        difficulty_weights={1: 0.20, 2: 0.35, 3: 0.30, 4: 0.10, 5: 0.05},
        default_prompt_dataset_size=3500,
        default_num_generations=8,
        default_max_completion_length=512,
        default_temperature=0.80,
        default_learning_rate=1.5e-6,
        default_beta=0.05,
        default_lora_r=32,
        default_max_steps=300,
        reward_format_weight=1.0,
        reward_accuracy_weight=1.0,
        beta_end=0.02,
        kl_relax_frac=0.50,
        default_initial_target=2,
    ),
    # ─────────────────────────────────────────────────────────────────────
    # Llama-3.2-3B-Instruct  →  L4 24 GB
    # Weaker reasoning (~45 % on diff-2) and noisier rollouts. We bump G to
    # 10 and use temp=0.9 for exploration, but keep lr=1e-6 conservative
    # because 3B Llama tends to format-collapse under aggressive lr.
    # Format weight 1.5 because Llama is the most prone to format drift
    # (especially after the KL relaxes); a stronger format anchor
    # protects the +0.15 bonus from being overshadowed during fast lr
    # decay. Accuracy weight 0.8 because the Brier signal alone is
    # already aggressive on the smaller model.
    # ─────────────────────────────────────────────────────────────────────
    "llama3b": CalibrationPreset(
        name="llama3b",
        model_hint="meta-llama/Llama-3.2-3B-Instruct",
        domain_weights={"math": 0.45, "code": 0.35, "logic": 0.20},
        difficulty_weights={1: 0.30, 2: 0.35, 3: 0.20, 4: 0.10, 5: 0.05},
        default_prompt_dataset_size=3500,
        default_num_generations=10,
        default_max_completion_length=512,
        default_temperature=0.90,
        default_learning_rate=1.0e-6,
        default_beta=0.04,
        default_lora_r=16,
        default_max_steps=350,
        reward_format_weight=1.5,
        reward_accuracy_weight=0.8,
        beta_end=0.015,
        kl_relax_frac=0.55,
        default_initial_target=1,
    ),
    # ─────────────────────────────────────────────────────────────────────
    # Phi-4-mini-instruct  →  L4 24 GB (sequential after Llama)
    # Best format compliance of the trio; reaches reliable XML by step ~25.
    # Dataset intentionally smaller (2500) since 250 × 8 = 2000 prompts are
    # consumed. Format weight 1.0 (Phi already does it well); accuracy
    # weight 1.0 to mirror Qwen's symmetric incentive shape.
    # ─────────────────────────────────────────────────────────────────────
    "phi4mini": CalibrationPreset(
        name="phi4mini",
        model_hint="microsoft/Phi-4-mini-instruct",
        domain_weights={"math": 0.45, "code": 0.35, "logic": 0.20},
        difficulty_weights={1: 0.25, 2: 0.35, 3: 0.25, 4: 0.10, 5: 0.05},
        default_prompt_dataset_size=2500,
        default_num_generations=8,
        default_max_completion_length=384,
        default_temperature=0.75,
        default_learning_rate=1.5e-6,
        default_beta=0.04,
        default_lora_r=16,
        default_max_steps=250,
        reward_format_weight=1.0,
        reward_accuracy_weight=1.0,
        beta_end=0.015,
        kl_relax_frac=0.50,
        default_initial_target=2,
    ),
}


def infer_preset_name(model_id: str) -> str:
    """Infer preset from model id; defaults to qwen7b for unknown ids."""
    m = (model_id or "").lower()
    if "qwen" in m and "7b" in m:
        return "qwen7b"
    if "llama" in m and "3b" in m:
        return "llama3b"
    if "phi-4-mini" in m or ("phi" in m and "mini" in m):
        return "phi4mini"
    # Legacy fallback: many users still run phi-3.5-mini
    if "phi-3.5" in m or "phi3.5" in m:
        return "phi4mini"
    return "qwen7b"


def get_preset(model_id: str, preset_override: str = "auto") -> CalibrationPreset:
    preset_name = infer_preset_name(model_id) if preset_override == "auto" else preset_override
    if preset_name not in MODEL_PRESETS:
        valid = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(f"Unknown preset '{preset_name}'. Valid presets: {valid}")
    return MODEL_PRESETS[preset_name]


def _normalize_weights(weight_map: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(max(v, 0.0) for v in weight_map.values()))
    if total <= 0.0:
        n = len(weight_map)
        return {k: 1.0 / n for k in weight_map}
    return {k: max(v, 0.0) / total for k, v in weight_map.items()}


def parse_weight_csv(
    csv_text: Optional[str],
    keys: List[str],
) -> Optional[Dict[str, float]]:
    """Parse comma-separated weight list aligned to ``keys``."""
    if not csv_text:
        return None
    parts = [p.strip() for p in csv_text.split(",") if p.strip()]
    if len(parts) != len(keys):
        raise ValueError(f"Expected {len(keys)} weights for {keys}, got {len(parts)}")
    raw = {k: float(v) for k, v in zip(keys, parts)}
    return _normalize_weights(raw)


def parse_difficulty_csv(csv_text: Optional[str]) -> Optional[Dict[int, float]]:
    """Parse 5 comma-separated difficulty weights for levels 1..5."""
    parsed = parse_weight_csv(csv_text, ["1", "2", "3", "4", "5"])
    if parsed is None:
        return None
    return {int(k): v for k, v in parsed.items()}


def prompt_templates(reasoning_mode: str) -> tuple[str, str]:
    """Return (system_prompt, user_template) for selected reasoning mode."""
    mode = (reasoning_mode or "required").lower()
    if mode not in REASONING_MODES:
        valid = ", ".join(REASONING_MODES)
        raise ValueError(f"Invalid reasoning_mode '{reasoning_mode}'. Valid: {valid}")

    system_prompt = """You are a precise and well-calibrated AI assistant.

Respond in EXACTLY this format:
<reasoning>
Briefly solve the problem.
</reasoning>
<answer>YOUR_ANSWER_HERE</answer>
<confidence>0.X</confidence>

Rules:
- Confidence must be between 0.0 and 1.0
- If very unsure, output <abstain/>
- Keep reasoning concise, then provide final answer and confidence."""
    user_template = (
        "{question}\n\n"
        "Think briefly in <reasoning>, then provide <answer> and <confidence>."
    )
    return system_prompt, user_template
