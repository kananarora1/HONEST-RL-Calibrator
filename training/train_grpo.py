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
from collections import deque
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from server.generators import code_gen, logic_gen, math_gen
from client.client import HonestEnv
from server.reward import (
    reward_brier,
    reward_format,
    reward_accuracy,
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
N_PROMPT_DATASET = 3000 

GENERATORS = {
    "math":  math_gen.generate,
    "code":  code_gen.generate,
    "logic": logic_gen.generate,
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


def build_prompt_dataset(n: int, tokenizer) -> list:
    log.info(f"Building prompt dataset ({n} prompts)...")
    rng = random.Random(1337)
    domain_list = list(GENERATORS.keys())
    records = []
    attempts = 0
    
    diff_weights = [0.40, 0.30, 0.15, 0.10, 0.05] 
    diff_choices = [1, 2, 3, 4, 5]

    while len(records) < n and attempts < n * 5:
        attempts += 1
        domain = rng.choice(domain_list)
        difficulty = rng.choices(diff_choices, weights=diff_weights, k=1)[0]
        seed = 500_000 + attempts
        
        try:
            question, ground_truth = GENERATORS[domain](difficulty, seed=seed)
        except Exception:
            continue
            
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_TEMPLATE.format(question=question)},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        records.append({
            "prompt":       prompt_text,
            "ground_truth": str(ground_truth),
            "difficulty":   difficulty,
            "domain":       domain,
        })
        
    log.info(f"  -> {len(records)} prompts ready ({attempts} attempts).")
    return records


# Reward distribution logging
_reward_history: deque = deque(maxlen=500)

def _log_reward_dist(rewards, step):
    _reward_history.extend(rewards)
    if step % 10 == 0 and len(_reward_history) > 0:
        arr = np.array(_reward_history)
        log.info(
            f"Step {step:04d} | mean={arr.mean():.4f}  std={arr.std():.4f}  "
            f"min={arr.min():.4f}  max={arr.max():.4f}  n={len(arr)}"
        )

def wrap_with_logging(fn, step_ref):
    def _logged(completions, prompts, **kwargs):
        rewards = fn(completions, prompts, **kwargs)
        step_ref[0] += 1
        _log_reward_dist(rewards, step_ref[0])
        return rewards
    return _logged

# Model loading
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

# KL early-stopping callback 
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
    parser.add_argument("--max-completion-length", type=int, default=2048)
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

    if args.dry_run:
        dry_completions = [
            "<answer>42</answer><confidence>0.9</confidence>",
            "<answer>41</answer><confidence>0.5</confidence>",
            "<abstain/>",
            "malformed output",
        ]
        dry_gt = ["42", "42", "42", "42"]
        dry_diff = [1, 1, 1, 1]
        dry_domains = ["math", "math", "math", "math"]
        print("Dry run: reward smoke test")
        print("reward_brier:", reward_brier(dry_completions, [], dry_gt, dry_diff, domain=dry_domains))
        print("reward_format:", reward_format(dry_completions))
        return

    if not torch.cuda.is_available() and not args.dry_run:
        raise SystemExit("No GPU detected.")

    from trl import GRPOConfig, GRPOTrainer

    log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"Torch: {torch.__version__}")

    if UNSLOTH_AVAILABLE:
        model, tokenizer = load_model_unsloth(hf_token, args.model_id)
    else:
        model, tokenizer = load_model_standard(hf_token, args.model_id)

    raw_records   = build_prompt_dataset(args.prompt_dataset_size, tokenizer)
    train_dataset = Dataset.from_list(raw_records)
    bf16 = _is_bfloat16_supported()

    _step_ref = [0]
    logged_brier = wrap_with_logging(reward_brier, _step_ref)

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
        
        environment_factory=lambda: HonestEnv(base_url=env_url).sync() if env_url else None,
        **({  "max_steps": args.max_steps} if args.max_steps else {}),
    )

    trainer = GRPOTrainer(
        model=model,
        # The environment_factory handles the core Brier score reward natively from the server.
        # `reward_format` acts as a local auxiliary penalty to strictly enforce XML structure.
        reward_funcs=[reward_format, reward_accuracy] if env_url else [logged_brier, reward_format, reward_accuracy],
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[KLEarlyStopCallback(kl_threshold=0.5, patience=20)],
    )

    log.info("=" * 60)
    log.info(f"Model:   {args.model_id}")
    log.info(f"Backend: {'Unsloth' if UNSLOTH_AVAILABLE else 'HF transformers'}")
    log.info(f"GPU:     {torch.cuda.get_device_name(0)} | bf16 supported: {bf16}")
    log.info(f"Reward:  {'live @ ' + env_url if env_url else 'local multi-reward'}")
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