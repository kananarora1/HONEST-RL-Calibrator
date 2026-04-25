UNSLOTH_AVAILABLE = False
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
    UNSLOTH_AVAILABLE = True
    print("Unsloth available — using optimised path.")
except Exception:
    print("Unsloth not available, using HF fallback.")

import argparse
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("grpo")

MODEL_ID         = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR       = "./honest-qwen-3b-grpo"
MAX_SEQ_LEN      = 2048
N_PROMPT_DATASET = 30000   # placeholder rows; resolved lazily via set_transform.
DOMAINS          = ["math", "code", "logic"]

GENERATORS: Dict[str, Callable[..., tuple]] = {
    "math":  math_generate,
    "code":  code_generate,
    "logic": logic_generate,
}

SYSTEM_PROMPT = """You are a precise and well-calibrated AI assistant.

When answering questions, you MUST respond in EXACTLY this format:
<reasoning>
Briefly think step-by-step to solve the problem.
</reasoning>
<answer>YOUR_ANSWER_HERE</answer>
<confidence>0.X</confidence>

Where:
- YOUR_ANSWER_HERE is your best answer to the question
- 0.X is a number between 0.0 and 1.0 representing how confident you are

Rules:
- Confidence 1.0 = completely certain
- Confidence 0.5 = 50/50 guess
- Confidence 0.0 = completely uncertain
- If you are very unsure, use <abstain/> instead
- Never include explanations outside the XML tags
- For numeric answers, give the number only (no units unless asked)
- For string answers, give the exact value only"""

USER_TEMPLATE = "{question}\n\nThink step-by-step in the <reasoning> block, then provide your final answer and confidence."


def _build_prompt_text(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_TEMPLATE.format(question=question)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_prompt_dataset(
    n: int,
    tokenizer,
    controller: DifficultyController,
) -> Dataset:
    """Build a *lazy* GRPO prompt dataset.

    The dataset stores ``n`` placeholder rows; the actual prompt, ground truth,
    domain, difficulty and ``problem_id`` are resolved on each ``__getitem__``
    by sampling difficulty from ``controller`` and calling the unified sampler.
    This makes the curriculum truly adaptive: as the reward wrapper feeds
    correctness into the controller, subsequent dataset accesses see the
    updated target distribution.

    Requires ``dataloader_num_workers=0`` (set in ``main()``) so the controller
    closure runs in the trainer process and updates take effect immediately.
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

    def _resolve_one() -> Optional[dict]:
        """Sample one (domain, difficulty) and call the unified sampler."""
        for _attempt in range(20):
            domain = sampling_rng.choice(DOMAINS)
            difficulty = controller.sample_difficulty(domain, rng=sampling_rng)
            try:
                question, ground_truth, problem_id = GENERATORS[domain](
                    difficulty,
                    seed=sampling_rng.randint(0, 2**31 - 1),
                )
                return {
                    "question":     question,
                    "ground_truth": str(ground_truth),
                    "difficulty":   int(difficulty),
                    "domain":       domain,
                    "problem_id":   str(problem_id),
                }
            except Exception as exc:
                log.debug("generator(%s, d=%d) raised %s — retrying", domain, difficulty, exc)
                continue
        # Last resort: a trivial procedural-style row so the dataloader never starves.
        log.warning("Generator retries exhausted; emitting a fallback math problem.")
        question, gt, pid = math_generate(1, seed=sampling_rng.randint(0, 2**31 - 1))
        return {
            "question":     question,
            "ground_truth": str(gt),
            "difficulty":   1,
            "domain":       "math",
            "problem_id":   str(pid),
        }

    def _transform(batch: Dict[str, list]) -> Dict[str, list]:
        n_in = len(batch["_idx"])
        out_prompt:   List[str] = []
        out_gt:       List[str] = []
        out_diff:     List[int] = []
        out_domain:   List[str] = []
        out_pid:      List[str] = []
        for _ in range(n_in):
            row = _resolve_one()
            if row is None:
                continue
            out_prompt.append(_build_prompt_text(tokenizer, row["question"]))
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
            "_idx":         list(batch["_idx"]),
        }

    ds.set_transform(_transform)
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


def make_brier_with_curriculum_feedback(
    controller: DifficultyController,
    step_ref: list,
):
    """Return a TRL-compatible reward function that:
      1) computes the calibrated Brier reward (the model's primary signal), AND
      2) records a single outcome per unique prompt into ``controller`` so the
         curriculum actually moves during training.

    Outcome aggregation: TRL calls a reward function once per *batch*, where
    a batch contains ``num_generations`` completions for *one* prompt
    (per device, per gradient-accumulation slot). We collapse the per-completion
    correctness into a single binary outcome per (domain, problem_id) pair via
    majority vote, then call ``controller.record_outcome`` exactly once per
    distinct prompt. This avoids over-weighting a single prompt by treating
    its 16 correlated rollouts as 16 independent observations.
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
            if correct is not None and domain in DOMAINS:
                group_correct[(domain, pid)].append(bool(correct))

        # Record one outcome per unique (domain, problem_id) — majority vote.
        for (dom, _pid), corrects in group_correct.items():
            if not corrects:
                continue
            majority = sum(corrects) > (len(corrects) / 2.0)
            try:
                controller.record_outcome(dom, majority)
            except Exception as exc:
                log.debug("record_outcome(%s, %s) raised: %s", dom, majority, exc)

        # Log running reward distribution (matches the previous behaviour).
        step_ref[0] += 1
        _log_reward_dist(rewards, step_ref[0])

        return rewards

    _wrapped.__name__ = "reward_brier_with_curriculum_feedback"
    return _wrapped


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _is_bfloat16_supported():
    if UNSLOTH_AVAILABLE:
        return is_bfloat16_supported()
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def load_model_unsloth(hf_token, model_id: str):
    log.info(f"Loading {model_id} via Unsloth (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
        token=hf_token,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    tokenizer.padding_side = "left"

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model, tokenizer


def load_model_standard(hf_token, model_id: str):
    log.info(f"Loading {model_id} via HF transformers (4-bit bnb)...")
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
        r=16,
        lora_alpha=16,
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
# main
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
            "  PYTHONPATH=. python data/ingestion/ingest_apps.py\n"
            "  PYTHONPATH=. python data/ingestion/regenerate_zebralogic.py\n"
        )
    counts = sampler.bucket_counts()
    log.info("Unified sampler ready: %d problems across %d (domain, difficulty) buckets.",
             total, len(counts))
    for (dom, diff), c in sorted(counts.items()):
        log.info("  (%-5s, d=%d) -> %d problems", dom, diff, c)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-wandb",  action="store_true")
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--prompt-dataset-size", type=int, default=N_PROMPT_DATASET)
    parser.add_argument("--num-generations", type=int, default=16)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1.5e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--colab-profile",
        choices=["none", "t4", "l4", "a100"],
        default="none",
        help="Apply Colab-friendly overrides for smaller GPUs.",
    )
    args = parser.parse_args()

    if args.colab_profile == "t4":
        args.num_generations = 4
        args.gradient_accumulation_steps = max(args.gradient_accumulation_steps, 16)
        args.max_completion_length = min(args.max_completion_length, 768)
    elif args.colab_profile == "l4":
        args.num_generations = min(args.num_generations, 8)
        args.max_completion_length = min(args.max_completion_length, 1024)
    elif args.colab_profile == "a100":
        args.num_generations = min(args.num_generations, 16)

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

        # Smoke the curriculum feedback wrapper end-to-end.
        ctrl = DifficultyController(domains=DOMAINS)
        step_ref = [0]
        wrapped = make_brier_with_curriculum_feedback(ctrl, step_ref)

        rewards = wrapped(
            dry_completions, [], dry_gt, dry_diff,
            domain=dry_domains, problem_id=dry_pids,
        )
        print("Dry run: reward smoke test")
        print("  brier_with_curriculum_feedback:", rewards)
        print("  reward_format:", reward_format(dry_completions))
        print("  reward_accuracy:", reward_accuracy(
            dry_completions, [], dry_gt, domain=dry_domains, problem_id=dry_pids,
        ))
        print("  controller snapshot after one batch:", ctrl.snapshot())
        return

    if not torch.cuda.is_available() and not args.dry_run:
        raise SystemExit("No GPU detected.")

    from trl import GRPOConfig, GRPOTrainer

    log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"Torch: {torch.__version__}")

    # Load curated data once, up front, so failure mode is loud and immediate.
    _warm_up_unified_sampler()

    if UNSLOTH_AVAILABLE:
        model, tokenizer = load_model_unsloth(hf_token, args.model_id)
    else:
        model, tokenizer = load_model_standard(hf_token, args.model_id)

    # One controller shared between (a) the lazy dataset's set_transform sampler,
    # (b) the curriculum-feedback reward wrapper, and (c) the wandb logging
    # callback — so the dashboard, the prompt distribution, and the controller
    # state all reflect the same instance.
    difficulty_controller = DifficultyController(domains=DOMAINS)

    train_dataset = build_prompt_dataset(
        args.prompt_dataset_size, tokenizer, controller=difficulty_controller,
    )
    bf16 = _is_bfloat16_supported()

    step_ref = [0]
    brier_with_feedback = make_brier_with_curriculum_feedback(
        difficulty_controller, step_ref,
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

    trainer = GRPOTrainer(
        model=model,
        # brier_with_feedback is the primary calibration reward AND the
        # curriculum feedback channel. reward_format and reward_accuracy
        # are auxiliary signals (format compliance + correctness bonus).
        reward_funcs=[brier_with_feedback, reward_format, reward_accuracy],
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[
            KLEarlyStopCallback(kl_threshold=0.5, patience=20),
            DifficultyControllerLogCallback(difficulty_controller),
        ],
    )

    log.info("=" * 60)
    log.info(f"Model:   {args.model_id}")
    log.info(f"Backend: {'Unsloth' if UNSLOTH_AVAILABLE else 'HF transformers'}")
    log.info(f"GPU:     {torch.cuda.get_device_name(0)} | bf16 supported: {bf16}")
    log.info(f"Reward:  brier_with_feedback + format + accuracy (local, curated data)")
    log.info(
        "GRPO: gens=%d | bs=%d | ga=%d | max_len=%d | beta=%.4f | grad_norm=%.2f | lr=%.2e | temp=%.2f",
        args.num_generations,
        args.per_device_train_batch_size,
        args.gradient_accumulation_steps,
        args.max_completion_length,
        args.beta,
        args.max_grad_norm,
        args.learning_rate,
        args.temperature,
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


if __name__ == "__main__":
    main()
