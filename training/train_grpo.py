UNSLOTH_AVAILABLE = False
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
    UNSLOTH_AVAILABLE = True
    print("Unsloth available — using optimised path.")
except Exception:
    print("Unsloth not available, using HF fallback.")

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Generators are backed by the unified sampler so training draws from the
# curated Hendrycks-MATH / MBPP+APPS / ZebraLogic corpus at the difficulty
# chosen by the adaptive DifficultyController. Logic d=1,2 falls back to the
# procedural generator inside the unified sampler itself.
from data.sampler.environment_adapter import (
    code_generate,
    get_sampler,
    logic_generate,
    math_generate,
)
from server.difficulty import DifficultyController
from server.reward import (
    compute_reward,
    parse_action,
    reward_accuracy,
    reward_format,
)
from calibration_profiles import (
    REASONING_MODES,
    SUPPORTED_PRESETS,
    get_preset,
    parse_weight_csv,
    prompt_templates,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("grpo")

# Defaults: preset auto-inferred from --model-id will override these where
# unset. Hard-coded value below is used only for argparse display / when no
# preset is selected.
MODEL_ID         = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LEN      = 2048
DOMAINS          = ["math", "code", "logic"]

GENERATORS: Dict[str, Callable[..., tuple]] = {
    "math":  math_generate,
    "code":  code_generate,
    "logic": logic_generate,
}


def _build_prompt_text(
    tokenizer,
    question: str,
    system_prompt: str,
    user_template: str,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_template.format(question=question)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_prompt_dataset(
    n: int,
    tokenizer,
    controller: DifficultyController,
    system_prompt: str,
    user_template: str,
    domain_weights: Dict[str, float],
) -> Dataset:
    """Build a *lazy* GRPO prompt dataset.

    The dataset stores ``n`` placeholder rows; the actual prompt, ground truth,
    domain, difficulty and ``problem_id`` are resolved on first access via
    ``set_transform``. Resolved rows are then **cached by ``_idx``** so that
    repeated lookups of the same index always return the *same* prompt.

    This invariant is critical for GRPO: the trainer's RepeatRandomSampler
    requests every prompt index ``num_generations`` times in a row to obtain
    a group of completions that share a single prompt, then computes
    group-relative advantages over those completions. If we returned a fresh
    prompt on every ``__getitem__`` call, every "group" would actually contain
    ``num_generations`` *different* prompts and the GRPO advantage signal
    would collapse.

    The curriculum still adapts because there are ``n`` distinct ``_idx``
    values: each *fresh* index triggers a new
    ``controller.sample_difficulty()`` call that sees the latest curriculum
    state. As the reward wrapper feeds correctness back into the controller,
    later (still-unresolved) indices increasingly draw from the updated
    target distribution.

    Requires ``dataloader_num_workers=0`` (set in ``main()``) so the
    controller closure and the resolution cache live in the trainer process.
    """
    log.info("Building lazy prompt dataset (%d placeholder rows).", n)
    placeholders = [
        {
            "prompt":       "",
            "ground_truth": "",
            "difficulty":   1,
            "domain":       "math",
            "problem_id":   "",
            "_idx":         i,
        }
        for i in range(n)
    ]
    ds = Dataset.from_list(placeholders)

    sampling_rng = random.Random(1337)
    resolved_cache: Dict[int, dict] = {}

    # Domain pool & weights — sampled per-row so the data mix matches the
    # preset's domain composition (Qwen 50/35/15, Llama 45/35/20, etc.).
    # Difficulty is *always* drawn from the controller (which encodes both
    # the static floor and the triangular overlay over its target band),
    # not from a global difficulty distribution: per-domain adaptation is
    # the whole point of the RLVE curriculum.
    domain_pool = [d for d in DOMAINS if domain_weights.get(d, 0.0) > 0.0]
    if not domain_pool:
        domain_pool = list(DOMAINS)
    domain_pool_weights = [domain_weights.get(d, 0.0) for d in domain_pool]
    if sum(domain_pool_weights) <= 0.0:
        domain_pool_weights = [1.0] * len(domain_pool)

    def _resolve_one() -> dict:
        """Sample one (domain, difficulty) and call the unified sampler."""
        for _attempt in range(20):
            domain = sampling_rng.choices(
                domain_pool, weights=domain_pool_weights, k=1,
            )[0]
            difficulty = controller.sample_difficulty(domain, rng=sampling_rng)
            try:
                question, ground_truth, problem_id = GENERATORS[domain](
                    difficulty,
                    seed=sampling_rng.randint(0, 2**31 - 1),
                )
                return {
                    "prompt":       _build_prompt_text(tokenizer, question, system_prompt, user_template),
                    "ground_truth": str(ground_truth),
                    "difficulty":   int(difficulty),
                    "domain":       domain,
                    "problem_id":   str(problem_id),
                }
            except Exception as exc:
                log.debug("generator(%s, d=%d) raised %s — retrying", domain, difficulty, exc)
                continue
        log.warning("Generator retries exhausted; emitting a fallback math problem.")
        question, gt, pid = math_generate(1, seed=sampling_rng.randint(0, 2**31 - 1))
        return {
            "prompt":       _build_prompt_text(tokenizer, question, system_prompt, user_template),
            "ground_truth": str(gt),
            "difficulty":   1,
            "domain":       "math",
            "problem_id":   str(pid),
        }

    def _transform(batch: Dict[str, list]) -> Dict[str, list]:
        idx_list = list(batch["_idx"])
        out_prompt:   List[str] = []
        out_gt:       List[str] = []
        out_diff:     List[int] = []
        out_domain:   List[str] = []
        out_pid:      List[str] = []
        for idx in idx_list:
            row = resolved_cache.get(int(idx))
            if row is None:
                row = _resolve_one()
                resolved_cache[int(idx)] = row
            out_prompt.append(row["prompt"])
            out_gt.append(row["ground_truth"])
            out_diff.append(row["difficulty"])
            out_domain.append(row["domain"])
            out_pid.append(row["problem_id"])
        return {
            "prompt":       out_prompt,
            "ground_truth": out_gt,
            "difficulty":   out_diff,
            "domain":       out_domain,
            "problem_id":   out_pid,
            "_idx":         idx_list,
        }

    ds.set_transform(_transform)
    ds._honest_resolved_cache = resolved_cache  # exposed for tests/inspection
    return ds


# ---------------------------------------------------------------------------
# Reward distribution logging + curriculum feedback
# ---------------------------------------------------------------------------

_reward_history: deque = deque(maxlen=500)


def _log_reward_dist(rewards: List[float], step: int) -> None:
    _reward_history.extend(rewards)
    if step % 10 == 0 and len(_reward_history) > 0:
        arr = np.array(_reward_history)
        log.info(
            f"Step {step:04d} | mean={arr.mean():.4f}  std={arr.std():.4f}  "
            f"min={arr.min():.4f}  max={arr.max():.4f}  n={len(arr)}"
        )


def make_brier_reward(
    step_ref: list,
    controller: Optional[DifficultyController] = None,
):
    """Return a TRL-compatible reward function that:
      1) computes the calibrated Brier reward (the model's primary signal), AND
      2) (if a controller is provided) records ONE outcome per unique prompt
         into ``controller`` so the curriculum actually moves during training.

    Outcome aggregation: TRL calls a reward function once per *batch*, where
    a batch contains ``num_generations`` completions for *one* prompt
    (per device, per gradient-accumulation slot). We collapse the per-completion
    correctness into a single binary outcome per (domain, problem_id) pair via
    majority vote, then call ``controller.record_outcome`` exactly once per
    distinct prompt. This avoids over-weighting a single prompt by treating
    its 16 correlated rollouts as 16 independent observations.

    When ``controller=None`` (the ``--no-controller`` path), the function still
    computes Brier rewards and logs distributions, but skips outcome
    recording entirely — so the curriculum stays at its initial target.
    """

    def _wrapped(
        completions: List[str],
        prompts: List[str],
        ground_truth: List[str],
        difficulty: List[int],
        **kwargs: Any,
    ) -> List[float]:
        n = len(completions)
        domains = kwargs.get("domain") or [None] * n
        pid_list = kwargs.get("problem_id") or [None] * n
        v_metas = kwargs.get("verification_metadata") or [{}] * n

        rewards: List[float] = []
        group_correct: Dict[tuple, list] = defaultdict(list)

        for idx, (comp, gt, diff) in enumerate(zip(completions, ground_truth, difficulty)):
            domain = domains[idx] if idx < len(domains) else None
            pid    = pid_list[idx] if idx < len(pid_list) else None
            v_meta = v_metas[idx] if idx < len(v_metas) else {}

            parsed = parse_action(comp)
            r, correct = compute_reward(
                parsed,
                str(gt),
                int(diff),
                problem_id=pid,
                domain=domain,
                verification_metadata=v_meta,
            )
            rewards.append(float(r))

            # Only real Answer outcomes (True/False) feed the controller.
            # Abstain / hint / malformed → correct is None → skip.
            if controller is not None and correct is not None and domain in DOMAINS:
                group_correct[(domain, pid)].append(bool(correct))

        # Record one outcome per unique (domain, problem_id) — majority vote.
        if controller is not None:
            for (dom, _pid), corrects in group_correct.items():
                if not corrects:
                    continue
                majority = sum(corrects) > (len(corrects) / 2.0)
                try:
                    controller.record_outcome(dom, majority)
                except Exception as exc:
                    log.debug("record_outcome(%s, %s) raised: %s", dom, majority, exc)

        # Log running reward distribution.
        step_ref[0] += 1
        _log_reward_dist(rewards, step_ref[0])

        return rewards

    _wrapped.__name__ = (
        "reward_brier_with_curriculum_feedback"
        if controller is not None else "reward_brier"
    )
    return _wrapped


def make_weighted(fn, weight: float, name: Optional[str] = None):
    """Multiply each per-rollout reward by a constant weight.

    Used to scale the auxiliary ``reward_format`` and ``reward_accuracy``
    signals per-preset without modifying the underlying reward functions
    (which are also imported by the eval scripts and tests).
    """
    if abs(weight - 1.0) < 1e-9:
        # Identity case: preserve original __name__ for cleaner logs.
        return fn

    def _scaled(*args, **kwargs):
        return [r * weight for r in fn(*args, **kwargs)]
    _scaled.__name__ = (name or fn.__name__) + f"_x{weight:g}"
    return _scaled


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _is_bfloat16_supported():
    if UNSLOTH_AVAILABLE:
        return is_bfloat16_supported()
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def load_model_unsloth(hf_token, model_id: str, lora_r: int = 16, lora_alpha: int = 32):
    log.info(f"Loading {model_id} via Unsloth (4-bit) | LoRA r={lora_r} α={lora_alpha}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
        token=hf_token,
    )
    # Only force the Qwen-2.5 chat template for Qwen models. For Llama-3.2 and
    # Phi-4-mini, the model's own chat template (already on the tokenizer)
    # is the right choice — overwriting it breaks generation.
    if "qwen" in model_id.lower():
        tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    tokenizer.padding_side = "left"

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model, tokenizer


def load_model_standard(hf_token, model_id: str, lora_r: int = 16, lora_alpha: int = 32):
    log.info(f"Loading {model_id} via HF transformers (4-bit bnb) | LoRA r={lora_r} α={lora_alpha}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if _is_bfloat16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, token=hf_token,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class KLEarlyStopCallback(TrainerCallback):
    """Hard-stop training if KL diverges too long.

    Distinct from ``AdaptiveBetaCallback``: this is a circuit breaker
    that fires when the KL has stayed above ``kl_threshold`` for
    ``patience`` consecutive steps. ``AdaptiveBetaCallback`` is a
    proactive *schedule* on the KL coefficient itself.
    """

    def __init__(self, kl_threshold: float = 0.5, patience: int = 20):
        self.kl_threshold = kl_threshold
        self.patience = patience
        self._counter = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        kl = logs.get("kl") or logs.get("objective/kl")
        if kl is not None:
            if kl > self.kl_threshold:
                self._counter += 1
                log.warning(
                    f"KL={kl:.4f} > {self.kl_threshold} "
                    f"({self._counter}/{self.patience} consecutive steps)"
                )
                if self._counter >= self.patience:
                    log.error("KL divergence too high for too long — stopping training.")
                    control.should_training_stop = True
            else:
                self._counter = 0


class AdaptiveBetaCallback(TrainerCallback):
    """Two-phase KL beta schedule.

    Research insight (RLHF / RLCR literature): a *high* KL coefficient
    early prevents the policy from drifting before the reward signal
    stabilizes; a *lower* beta later allows the model to consolidate
    calibration improvements without being constantly pulled back to the
    base distribution. We linearly interpolate between ``beta_start`` and
    ``beta_end`` in a single transition window centered on
    ``relax_frac × max_steps``.

    Implemented by mutating ``trainer.args.beta`` in place — TRL's
    GRPOTrainer reads ``self.args.beta`` per step.
    """

    def __init__(
        self,
        trainer_ref: list,
        beta_start: float,
        beta_end: float,
        max_steps: int,
        relax_frac: float = 0.5,
        transition_window: int = 30,
    ):
        self.trainer_ref = trainer_ref
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.max_steps = max(1, int(max_steps))
        self.relax_step = int(self.max_steps * relax_frac)
        self.window = max(1, int(transition_window))

    def _scheduled_beta(self, step: int) -> float:
        if step < self.relax_step - self.window // 2:
            return self.beta_start
        if step > self.relax_step + self.window // 2:
            return self.beta_end
        t = (step - (self.relax_step - self.window // 2)) / float(self.window)
        return self.beta_start + t * (self.beta_end - self.beta_start)

    def on_step_begin(self, args, state, control, **kwargs):
        trainer = self.trainer_ref[0]
        if trainer is None:
            return
        new_beta = self._scheduled_beta(state.global_step)
        if abs(trainer.args.beta - new_beta) > 1e-6:
            trainer.args.beta = float(new_beta)
            if state.global_step % 10 == 0:
                log.info(f"[AdaptiveBeta] step={state.global_step} beta={new_beta:.4f}")


class RewardHealthCallback(TrainerCallback):
    """Detects dead-batch regimes (reward_std ≈ 0).

    GRPO's advantage = (reward - mean) / std. When std → 0 across a
    group, advantages collapse and the policy receives no learning
    signal. This is the silent failure mode behind a flat reward curve.

    Behavior:
      • Warns once per ``warn_patience`` consecutive low-std steps.
      • Stops training if it persists for ``fatal_patience`` steps —
        at that point the run is wasting compute and the model has
        almost certainly mode-collapsed onto a single canned output
        (typically the safe "<abstain/>" or all-0.5 confidence).
    """

    def __init__(
        self,
        warn_threshold: float = 1e-4,
        warn_patience: int = 5,
        fatal_patience: int = 30,
    ):
        self.warn_threshold = warn_threshold
        self.warn_patience = warn_patience
        self.fatal_patience = fatal_patience
        self._dead = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        r_std = logs.get("reward_std") or logs.get("rewards/mean_std")
        if r_std is None:
            return
        if r_std < self.warn_threshold:
            self._dead += 1
            if self._dead >= self.fatal_patience:
                log.error(
                    f"reward_std={r_std:.6f} < {self.warn_threshold} for "
                    f"{self._dead} steps — collapse detected, stopping."
                )
                control.should_training_stop = True
            elif self._dead == self.warn_patience:
                log.warning(
                    f"reward_std={r_std:.6f} stuck near zero "
                    f"({self._dead}/{self.fatal_patience} steps)"
                )
        else:
            self._dead = 0


class DifficultyControllerLogCallback(TrainerCallback):
    """Emit DifficultyController snapshot to TRL logs (and thus to wandb).

    Logs 5 keys per domain × 3 domains = 15 keys per step:
    target, rolling_acc, dist_d1, dist_d3, dist_d5.
    """

    def __init__(self, controller: DifficultyController):
        self.controller = controller

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        snapshot = self.controller.snapshot()
        for domain, s in snapshot.items():
            logs[f"difficulty/{domain}/target"] = s["target_difficulty"]
            logs[f"difficulty/{domain}/rolling_acc"] = (
                s["rolling_accuracy"] if s["rolling_accuracy"] is not None else 0.0
            )
            dist = s["distribution"]
            logs[f"difficulty/{domain}/dist_d1"] = dist[0]
            logs[f"difficulty/{domain}/dist_d3"] = dist[2]
            logs[f"difficulty/{domain}/dist_d5"] = dist[4]


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def _warm_up_unified_sampler() -> None:
    """Force-load the unified sampler once at startup so the first dataset
    access doesn't pay the 380MB JSONL load cost mid-training. Fail fast with
    a useful error if the curated data is missing."""
    log.info("Warming up unified sampler (loading curated datasets into memory)...")
    sampler = get_sampler()
    total = sampler.total_count()
    if total == 0:
        raise SystemExit(
            "Unified sampler is empty — no curated problems found in data/processed/. "
            "Run the ingestion scripts:\n"
            "  PYTHONPATH=. python data/ingestion/ingest_hendrycks_math.py\n"
            "  PYTHONPATH=. python data/ingestion/ingest_mbpp.py\n"
            "  PYTHONPATH=. python -m data.ingestion.ingest_apps\n"
            "  PYTHONPATH=. python data/ingestion/regenerate_zebralogic.py\n"
        )
    counts = sampler.bucket_counts()
    log.info("Unified sampler ready: %d problems across %d (domain, difficulty) buckets.",
             total, len(counts))
    for (dom, diff), c in sorted(counts.items()):
        log.info("  (%-5s, d=%d) -> %d problems", dom, diff, c)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    # ─ Run / model ───────────────────────────────────────────────────────
    p.add_argument("--dry-run",   action="store_true")
    p.add_argument("--no-wandb",  action="store_true")
    p.add_argument("--model-id",  type=str, default=MODEL_ID)
    p.add_argument(
        "--model-preset",
        choices=["auto", *SUPPORTED_PRESETS],
        default="auto",
        help="Calibration preset. 'auto' infers from --model-id.",
    )
    p.add_argument(
        "--reasoning-mode",
        choices=list(REASONING_MODES),
        default="optional",
        help="Prompt style: none (fast), optional (recommended), required (strict CoT).",
    )
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory. Defaults to ./honest-{model-slug}-grpo")

    # ─ Data ──────────────────────────────────────────────────────────────
    p.add_argument("--prompt-dataset-size", type=int, default=None,
                   help="Prompt pool size. If unset, preset default is used.")
    p.add_argument("--domain-weights", type=str, default=None,
                   help="Override domain weights as csv [math,code,logic], e.g. 0.5,0.35,0.15")

    # ─ GRPO knobs (preset-aware: None = use preset default) ──────────────
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--num-generations", type=int, default=None)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=None)
    p.add_argument("--max-completion-length", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None,
                   help="Defaults to 2× lora-r if not set.")

    # ─ Trainer (uniform across presets) ──────────────────────────────────
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)

    # ─ Hardware / Colab ──────────────────────────────────────────────────
    p.add_argument(
        "--colab-profile",
        choices=["none", "t4", "l4", "a100"],
        default="none",
        help=(
            "GPU safety caps (clipping, never raising risky values).\n"
            "  t4   – T4 16 GB  : cap G≤4,  max_len≤512,  r≤16, ga≥16\n"
            "  l4   – L4 24 GB  : cap G≤10, max_len≤640,  r≤16, ga≥8\n"
            "  a100 – A100 40 GB: cap G≤16, max_len≤1024, r≤32, ga≥4\n"
            "  none – no overrides"
        ),
    )

    # ─ Adaptive curriculum ───────────────────────────────────────────────
    p.add_argument(
        "--no-controller",
        action="store_true",
        help="Disable adaptive feedback. Curriculum stays at initial_target.",
    )
    p.add_argument(
        "--controller-initial-target",
        type=int,
        default=None,
        help="Per-domain initial target_difficulty. Defaults to preset value.",
    )
    return p


def _apply_preset_defaults(args, preset) -> None:
    """Fill any unset CLI arg from the preset. Mutates ``args`` in place."""
    if args.prompt_dataset_size is None:
        args.prompt_dataset_size = preset.default_prompt_dataset_size
    if args.max_steps is None:
        args.max_steps = preset.default_max_steps
    if args.num_generations is None:
        args.num_generations = preset.default_num_generations
    if args.gradient_accumulation_steps is None:
        # Smaller models need more accumulation to keep effective batch ≥ 64
        # while we hold per_device_train_batch_size at 1.
        args.gradient_accumulation_steps = 8 if preset.name == "qwen7b" else 16
    if args.max_completion_length is None:
        args.max_completion_length = preset.default_max_completion_length
    if args.temperature is None:
        args.temperature = preset.default_temperature
    if args.learning_rate is None:
        args.learning_rate = preset.default_learning_rate
    if args.beta is None:
        args.beta = preset.default_beta
    if args.lora_r is None:
        args.lora_r = preset.default_lora_r


def _apply_colab_profile_caps(args) -> None:
    """Clip hyperparameters that exceed safe caps for the chosen GPU.

    Note: a profile only *clips down* (e.g. ``num_generations``,
    ``max_completion_length``, ``lora_r``) and *raises up*
    ``gradient_accumulation_steps``. It never widens user choices —
    so explicit ``--num-generations 16`` on T4 is silently safe-clipped to 4
    rather than crashing OOM mid-training.
    """
    profile_caps = {
        "t4":   {"G": 4,  "max_len": 512,  "r": 16, "min_ga": 16},
        "l4":   {"G": 10, "max_len": 640,  "r": 16, "min_ga": 8},
        "a100": {"G": 16, "max_len": 1024, "r": 32, "min_ga": 4},
        "none": None,
    }
    caps = profile_caps[args.colab_profile]
    if caps is None:
        return
    if args.num_generations > caps["G"]:
        log.warning("num_generations=%d exceeds %s cap (%d). Clipping.",
                    args.num_generations, args.colab_profile, caps["G"])
        args.num_generations = caps["G"]
    if args.max_completion_length > caps["max_len"]:
        log.warning("max_completion_length=%d exceeds %s cap (%d). Clipping.",
                    args.max_completion_length, args.colab_profile, caps["max_len"])
        args.max_completion_length = caps["max_len"]
    if args.lora_r > caps["r"]:
        log.warning("lora_r=%d exceeds %s cap (%d). Clipping.",
                    args.lora_r, args.colab_profile, caps["r"])
        args.lora_r = caps["r"]
    if args.gradient_accumulation_steps < caps["min_ga"]:
        log.warning(
            "gradient_accumulation_steps=%d below %s minimum (%d). Raising.",
            args.gradient_accumulation_steps, args.colab_profile, caps["min_ga"],
        )
        args.gradient_accumulation_steps = caps["min_ga"]


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    preset = get_preset(args.model_id, args.model_preset)
    domain_weights = (
        parse_weight_csv(args.domain_weights, ["math", "code", "logic"])
        or preset.domain_weights
    )
    system_prompt, user_template = prompt_templates(args.reasoning_mode)

    _apply_preset_defaults(args, preset)
    _apply_colab_profile_caps(args)

    if args.lora_alpha is None:
        args.lora_alpha = args.lora_r * 2
    if args.output_dir is None:
        slug = args.model_id.split("/")[-1].lower().replace(".", "-")
        args.output_dir = f"./honest-{slug}-grpo"

    hf_token  = os.environ.get("HF_TOKEN")
    env_url   = os.environ.get("HONEST_ENV_URL", "").strip()
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    report_to = "none" if (args.no_wandb or not wandb_key) else "wandb"

    if env_url:
        log.warning(
            "HONEST_ENV_URL=%s is set, but the trainer no longer routes rollouts "
            "through the env server (TRL's GRPOConfig does not accept "
            "environment_factory in this version). All rewards are computed "
            "locally; the env server is unused for training. Ignore this URL or "
            "unset HONEST_ENV_URL to silence this warning.",
            env_url,
        )

    # Difficulty controller is built early so dry-run can introspect it.
    initial_target = (
        args.controller_initial_target
        if args.controller_initial_target is not None
        else preset.default_initial_target
    )
    difficulty_controller = DifficultyController(
        domains=DOMAINS, initial_target=initial_target,
    )
    feedback_controller = None if args.no_controller else difficulty_controller

    # ─── Dry run smoke test ─────────────────────────────────────────────
    if args.dry_run:
        dry_completions = [
            "<reasoning>r</reasoning><answer>42</answer><confidence>0.9</confidence>",
            "<reasoning>r</reasoning><answer>41</answer><confidence>0.5</confidence>",
            "<abstain/>",
            "malformed output",
        ]
        dry_gt      = ["42", "42", "42", "42"]
        dry_diff    = [1, 1, 1, 1]
        dry_domains = ["math", "math", "math", "math"]
        dry_pids    = ["procedural_math_d1_dryrun"] * 4

        step_ref = [0]
        wrapped = make_brier_reward(step_ref, controller=feedback_controller)
        rewards = wrapped(
            dry_completions, [], dry_gt, dry_diff,
            domain=dry_domains, problem_id=dry_pids,
        )
        weighted_format = make_weighted(reward_format, preset.reward_format_weight, "format")
        weighted_accuracy = make_weighted(
            reward_accuracy, preset.reward_accuracy_weight, "accuracy",
        )
        print("Dry run: reward smoke test")
        print(f"  preset:                      {preset.name} (model_hint={preset.model_hint})")
        print(f"  reasoning_mode:              {args.reasoning_mode}")
        print(f"  brier_with_curriculum:       {rewards}")
        print(f"  reward_format        (×{preset.reward_format_weight}): "
              f"{weighted_format(dry_completions)}")
        print(f"  reward_accuracy      (×{preset.reward_accuracy_weight}): "
              f"{weighted_accuracy(dry_completions, [], dry_gt, domain=dry_domains, problem_id=dry_pids)}")
        print(f"  controller snapshot (math):  {difficulty_controller.snapshot()['math']}")
        print(f"  KL schedule:                 beta {args.beta:.4f} → {preset.beta_end:.4f} "
              f"@ step {int((args.max_steps or 0) * preset.kl_relax_frac)}/{args.max_steps}")
        print(f"  hyperparams:                 G={args.num_generations} "
              f"max_len={args.max_completion_length} ga={args.gradient_accumulation_steps} "
              f"lr={args.learning_rate:g} temp={args.temperature} lora_r={args.lora_r}")
        print(f"  curriculum mode:             "
              f"{'ADAPTIVE' if feedback_controller else 'STATIC (init_target=' + str(initial_target) + ')'}")
        return

    if not torch.cuda.is_available():
        raise SystemExit("No GPU detected.")

    from trl import GRPOConfig, GRPOTrainer

    log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"Torch: {torch.__version__}")

    # Load curated data once, up front, so failure mode is loud and immediate.
    _warm_up_unified_sampler()

    if UNSLOTH_AVAILABLE:
        model, tokenizer = load_model_unsloth(
            hf_token, args.model_id,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        )
    else:
        model, tokenizer = load_model_standard(
            hf_token, args.model_id,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        )

    # The same controller instance is shared between (a) the lazy dataset's
    # set_transform sampler, (b) the curriculum-feedback reward wrapper, and
    # (c) the wandb log callback — so dashboard, prompt distribution, and
    # controller state all reflect the same instance.
    train_dataset = build_prompt_dataset(
        args.prompt_dataset_size,
        tokenizer,
        controller=difficulty_controller,
        system_prompt=system_prompt,
        user_template=user_template,
        domain_weights=domain_weights,
    )
    bf16 = _is_bfloat16_supported()

    step_ref = [0]
    brier_with_feedback = make_brier_reward(step_ref, controller=feedback_controller)
    weighted_format = make_weighted(
        reward_format, preset.reward_format_weight, "format",
    )
    weighted_accuracy = make_weighted(
        reward_accuracy, preset.reward_accuracy_weight, "accuracy",
    )

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        learning_rate=args.learning_rate,

        beta=args.beta,
        max_grad_norm=args.max_grad_norm,

        scale_rewards=True,
        num_iterations=1,
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        fp16=not bf16,
        bf16=bf16,
        optim="adamw_8bit",
        report_to=report_to,
        seed=args.seed,
        # IMPORTANT: must be 0 so the lazy dataset's controller closure runs
        # in-process. With workers > 0, the controller is pickled per-worker
        # and updates from the reward wrapper would not be visible.
        dataloader_num_workers=0,

        **({"max_steps": args.max_steps} if args.max_steps else {}),
    )

    _trainer_ref: list = [None]  # filled after construction so callback can mutate beta
    callbacks: list = [
        KLEarlyStopCallback(kl_threshold=0.5, patience=20),
        AdaptiveBetaCallback(
            _trainer_ref,
            beta_start=args.beta,
            beta_end=preset.beta_end,
            max_steps=args.max_steps or 1,
            relax_frac=preset.kl_relax_frac,
        ),
        RewardHealthCallback(warn_threshold=1e-4, warn_patience=5, fatal_patience=30),
    ]
    if feedback_controller is not None:
        callbacks.append(DifficultyControllerLogCallback(difficulty_controller))

    trainer = GRPOTrainer(
        model=model,
        # brier_with_feedback is the primary calibration reward AND the
        # curriculum feedback channel. reward_format and reward_accuracy
        # are auxiliary signals (XML compliance + correctness bonus),
        # weighted per-preset.
        reward_funcs=[brier_with_feedback, weighted_format, weighted_accuracy],
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    _trainer_ref[0] = trainer

    log.info("=" * 60)
    log.info(f"Model:   {args.model_id}")
    log.info(f"Backend: {'Unsloth' if UNSLOTH_AVAILABLE else 'HF transformers'}")
    log.info(f"GPU:     {torch.cuda.get_device_name(0)} | bf16 supported: {bf16}")
    log.info(f"Reward:  brier_with_feedback + format×{preset.reward_format_weight} + "
             f"accuracy×{preset.reward_accuracy_weight}  (local, curated data)")
    log.info(
        "Preset:  %s (inferred=%s) | reasoning_mode=%s",
        preset.name,
        get_preset(args.model_id, "auto").name,
        args.reasoning_mode,
    )
    log.info("DataMix: domain_weights=%s | initial_target=%d", domain_weights, initial_target)
    if feedback_controller is not None:
        log.info(
            "Curriculum: ADAPTIVE | logged via DifficultyControllerLogCallback "
            "(per-domain target/rolling_acc/dist_d1,d3,d5 → wandb)",
        )
    else:
        log.info(
            "Curriculum: STATIC (--no-controller) | sampling fixed at "
            "init target=%d for all domains", initial_target,
        )
    log.info(
        "KL:      beta_start=%.4f → beta_end=%.4f @ step %d (relax_frac=%.2f)",
        args.beta, preset.beta_end,
        int((args.max_steps or 0) * preset.kl_relax_frac),
        preset.kl_relax_frac,
    )
    log.info(
        "GRPO:    gens=%d | bs=%d | ga=%d | max_len=%d | beta=%.4f | "
        "grad_norm=%.2f | lr=%.2e | temp=%.2f | lora_r=%d | lora_α=%d",
        args.num_generations,
        args.per_device_train_batch_size,
        args.gradient_accumulation_steps,
        args.max_completion_length,
        args.beta,
        args.max_grad_norm,
        args.learning_rate,
        args.temperature,
        args.lora_r,
        args.lora_alpha,
    )
    log.info("=" * 60)

    t0 = time.time()
    trainer.train()
    log.info(f"Training complete in {(time.time()-t0)/60:.1f} min.")

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path / "final_adapters"))
    tokenizer.save_pretrained(str(out_path / "final_adapters"))
    log.info(f"Saved to {out_path / 'final_adapters'}")

    # Persist final controller state alongside the LoRA adapter so eval
    # scripts can reproduce / chart the curriculum the model was trained on.
    snap_path = out_path / "final_adapters" / "controller_state.json"
    snap_path.write_text(json.dumps(
        difficulty_controller.snapshot(), indent=2, default=str,
    ))
    log.info(f"Final controller state -> {snap_path}")


if __name__ == "__main__":
    main()
