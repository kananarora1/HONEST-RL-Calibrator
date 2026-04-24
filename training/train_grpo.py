"""
training/train_grpo.py — GRPO Reinforcement Learning against HONEST-RL-Calibrator

Uses Group Relative Policy Optimization (GRPO) via TRL + Unsloth to teach
Qwen2.5-3B-Instruct that overconfident wrong answers are heavily penalized.

Run on a GPU machine (e.g. Google Colab A100/T4):
    export HONEST_ENV_URL="https://your-space.hf.space"
    export WANDB_API_KEY="your-key"          # optional, can set report_to="none"
    python training/train_grpo.py

Dry-run (no GPU, no server, sanity-check only):
    python training/train_grpo.py --dry-run --max-steps 5
"""

# ── Unsloth MUST be first — only import if actually available ───────────────
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
    UNSLOTH_AVAILABLE = True
except (ImportError, NotImplementedError):
    # Unsloth not available (e.g. no NVIDIA GPU / AMD machine / CPU-only).
    # Dry-run mode still works without it using a dummy tokenizer.
    UNSLOTH_AVAILABLE = False

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from server.generators import code_gen, logic_gen, math_gen
from server.reward import parse_action, compute_reward
from server.verifier import verify_answer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("grpo")

# ---------------------------------------------------------------------------
# Constants / Config
# ---------------------------------------------------------------------------
MODEL_ID         = "Qwen/Qwen2.5-3B-Instruct"
SFT_ADAPTERS_DIR = str(PROJECT_ROOT / "training" / "format_sft_adapters")
OUTPUT_DIR       = "./honest-qwen-3b-grpo"
MAX_SEQ_LEN      = 2048
N_PROMPT_DATASET = 1000

GENERATORS = {
    "math":  math_gen.generate,
    "code":  code_gen.generate,
    "logic": logic_gen.generate,
}

SYSTEM_PROMPT = """\
You are a precise and well-calibrated AI assistant.

When answering questions, you MUST respond in EXACTLY this format:
<answer>YOUR_ANSWER_HERE</answer><confidence>0.X</confidence>

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
- For string answers, give the exact value only

Example responses:
<answer>42</answer><confidence>0.9</confidence>
<answer>Paris</answer><confidence>0.8</confidence>
<abstain/>"""

USER_TEMPLATE = "{question}\n\nRespond only with the XML format specified."

# ---------------------------------------------------------------------------
# Reward function  (called by GRPOTrainer for each completion)
# ---------------------------------------------------------------------------

def _ground_truth_store():
    """Thread-safe store that holds ground truth for the current batch."""
    return {}

_GT_STORE: dict = {}   # keyed by question text → ground_truth, difficulty


def _local_reward_fn(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
    """
    GRPO reward function.

    For each (prompt, completion) pair:
      1. Extract the question from the prompt.
      2. Look up the stored ground truth for that question.
      3. Parse the completion with parse_action().
      4. Compute reward using compute_reward() (Brier score).

    This function is called synchronously by TRL's GRPOTrainer.
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        entry = _GT_STORE.get(prompt)
        if entry is None:
            # Fallback: malformed reward if no ground truth available
            rewards.append(-0.5)
            continue

        ground_truth = entry["ground_truth"]
        difficulty   = entry["difficulty"]
        parsed       = parse_action(completion)
        reward, _    = compute_reward(parsed, ground_truth, difficulty)
        rewards.append(float(reward))

    return rewards


# ---------------------------------------------------------------------------
# Async reward function (used when HONEST_ENV_URL is set — live environment)
# ---------------------------------------------------------------------------

async def _env_reward_async(
    prompt: str,
    completion: str,
    env_url: str,
) -> float:
    """
    Send (prompt, completion) to a live HONEST-RL-Calibrator HF Space server
    and return its reward. Falls back to local Brier score if server fails.
    """
    try:
        import aiohttp
        action_payload = {"raw_text": completion}
        async with aiohttp.ClientSession() as session:
            # Reset gets a new question; we step immediately with the completion
            async with session.post(f"{env_url}/reset") as resp:
                reset_data = await resp.json()
            step_payload = {"session_id": reset_data.get("session_id", ""), **action_payload}
            async with session.post(f"{env_url}/step", json=step_payload) as resp:
                step_data = await resp.json()
            return float(step_data.get("reward", -0.5))
    except Exception as e:
        log.warning(f"Server reward failed ({e}), falling back to local Brier score.")
        entry = _GT_STORE.get(prompt)
        if entry is None:
            return -0.5
        parsed = parse_action(completion)
        reward, _ = compute_reward(parsed, entry["ground_truth"], entry["difficulty"])
        return float(reward)


def make_env_reward_fn(env_url: str):
    """Wrap async env reward into a sync function for GRPOTrainer."""
    def _fn(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
        loop = asyncio.new_event_loop()
        try:
            tasks = [_env_reward_async(p, c, env_url)
                     for p, c in zip(prompts, completions)]
            return loop.run_until_complete(asyncio.gather(*tasks))
        finally:
            loop.close()
    return _fn


# ---------------------------------------------------------------------------
# Prompt dataset builder
# ---------------------------------------------------------------------------

def build_prompt_dataset(n: int, tokenizer) -> list:
    """
    Generate n prompts using the problem generators.
    Ground truths are stored in _GT_STORE keyed by prompt string.
    Returns a HuggingFace-compatible list of dicts with "prompt" key.
    """
    log.info(f"Building prompt dataset ({n} prompts)...")
    rng    = random.Random(1337)
    domain_list = list(GENERATORS.keys())
    records = []
    attempts = 0

    while len(records) < n and attempts < n * 5:
        attempts += 1
        domain     = rng.choice(domain_list)
        difficulty = rng.randint(1, 5)
        seed       = 500_000 + attempts

        try:
            question, ground_truth = GENERATORS[domain](difficulty, seed=seed)
        except (RuntimeError, Exception):
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_TEMPLATE.format(question=question)},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        _GT_STORE[prompt_text] = {
            "ground_truth": ground_truth,
            "difficulty":   difficulty,
            "domain":       domain,
        }
        records.append({"prompt": prompt_text})

    log.info(f"  → {len(records)} prompts ready ({attempts} attempts).")
    return records


# ---------------------------------------------------------------------------
# Reward distribution logger (hook)
# ---------------------------------------------------------------------------

_reward_history: deque = deque(maxlen=500)
_step_count = 0


def _log_reward_dist(rewards: List[float], step: int):
    _reward_history.extend(rewards)
    if step % 10 == 0 and len(_reward_history) > 0:
        arr = np.array(_reward_history)
        log.info(
            f"Step {step:04d} | "
            f"mean={arr.mean():.4f}  std={arr.std():.4f}  "
            f"min={arr.min():.4f}  max={arr.max():.4f}  "
            f"n={len(arr)}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GRPO RL training for HONEST-RL-Calibrator")
    parser.add_argument("--dry-run", action="store_true",
                        help="Sanity check: load model, build dataset, skip actual training")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max training steps (e.g. 5 for smoke test)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging even if API key is set")
    args = parser.parse_args()

    # Enforce Unsloth only for real training, not dry-run
    if not args.dry_run and not UNSLOTH_AVAILABLE:
        raise SystemExit(
            "Unsloth not found. Install with:\n"
            "  pip install unsloth trl peft datasets bitsandbytes\n"
            "Run this script on a machine with an NVIDIA GPU.\n"
            "For a local dry-run, use: python training/train_grpo.py --dry-run"
        )

    env_url = os.environ.get("HONEST_ENV_URL", "")
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    report_to = "none" if (args.no_wandb or not wandb_key) else "wandb"

    # ── Dry-run: use lightweight HF tokenizer, skip model load ──────────────
    if args.dry_run:
        log.info("DRY-RUN mode — using AutoTokenizer (no GPU needed).")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, trust_remote_code=True
        )
        tokenizer.padding_side = "left"
        raw_records = build_prompt_dataset(min(N_PROMPT_DATASET, 20), tokenizer)
        log.info(f"  Dataset sample (truncated):\n{raw_records[0]['prompt'][:400]}...")
        log.info(f"  Ground truth store size: {len(_GT_STORE)}")
        # Test reward function on a dummy completion
        sample_prompt   = raw_records[0]["prompt"]
        sample_entry    = _GT_STORE[sample_prompt]
        dummy_answer    = sample_entry["ground_truth"]
        dummy_completion = f"<answer>{dummy_answer}</answer><confidence>0.8</confidence>"
        test_rewards = _local_reward_fn([dummy_completion], [sample_prompt])
        log.info(f"  Reward smoke test → {test_rewards[0]:.4f} (should be ~+0.02 for correct answer @ 0.8 conf)")
        log.info("Dry run complete ✓")
        return

    # ── 1. Load model ────────────────────────────────────────────────────────
    log.info(f"Loading {MODEL_ID} via Unsloth (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    tokenizer.padding_side = "left"   # required for GRPO

    # ── 2. Optionally load SFT adapters (format compliance baseline) ─────────
    sft_path = Path(SFT_ADAPTERS_DIR)
    if sft_path.exists() and any(sft_path.iterdir()):
        from peft import PeftModel
        log.info(f"Loading SFT adapters from {SFT_ADAPTERS_DIR}...")
        model = PeftModel.from_pretrained(model, SFT_ADAPTERS_DIR)
        model = model.merge_and_unload()  # merge before adding new LoRA
        log.info("  SFT adapters merged.")
    else:
        log.info("No SFT adapters found — starting from base model.")

    # ── 3. Attach fresh LoRA for GRPO ────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ── 4. Build prompt dataset ───────────────────────────────────────────────
    raw_records = build_prompt_dataset(N_PROMPT_DATASET, tokenizer)

    if args.dry_run:
        log.info("DRY-RUN: model loaded, dataset built — skipping training.")
        log.info(f"  Sample prompt (truncated):\n{raw_records[0]['prompt'][:300]}...")
        log.info("Dry run complete ✓")
        return

    train_dataset = Dataset.from_list(raw_records)

    # ── 5. Choose reward function ─────────────────────────────────────────────
    if env_url:
        log.info(f"Using live HONEST server reward: {env_url}")
        reward_fn = make_env_reward_fn(env_url)
    else:
        log.info("No HONEST_ENV_URL set — using local Brier score reward.")
        reward_fn = _local_reward_fn

    # ── 6. GRPO config ────────────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_generations=8,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        num_train_epochs=3,
        max_prompt_length=512,
        max_completion_length=512,
        save_steps=50,
        logging_steps=1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        report_to=report_to,
        seed=42,
        **({"max_steps": args.max_steps} if args.max_steps else {}),
    )

    # ── 7. Reward wrapper with logging ────────────────────────────────────────
    _step_ref = [0]

    def logged_reward_fn(completions, prompts, **kwargs):
        rewards = reward_fn(completions, prompts, **kwargs)
        _step_ref[0] += 1
        _log_reward_dist(rewards, _step_ref[0])
        return rewards

    # ── 8. Initialize and run GRPOTrainer ─────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        config=grpo_config,
        reward_funcs=[logged_reward_fn],
        train_dataset=train_dataset,
    )

    log.info("=" * 60)
    log.info("Starting GRPO training…")
    log.info(f"  Model:       {MODEL_ID}")
    log.info(f"  Reward:      {'live env @ ' + env_url if env_url else 'local Brier'}")
    log.info(f"  W&B:         {report_to}")
    log.info(f"  Output:      {OUTPUT_DIR}")
    log.info("=" * 60)

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    log.info(f"Training complete in {elapsed/60:.1f} min.")

    # ── 9. Save adapters ──────────────────────────────────────────────────────
    out_path = Path(OUTPUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path / "final_adapters"))
    tokenizer.save_pretrained(str(out_path / "final_adapters"))
    log.info(f"Adapters saved to {out_path / 'final_adapters'}")

    if wandb_key and not args.no_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
