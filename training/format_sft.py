"""
training/format_sft.py — Lightweight Format-Only SFT using Unsloth

Goal: Teach Qwen2.5-3B to reliably output <answer>...</answer><confidence>0.X</confidence>
      format regardless of question complexity. NOT about improving accuracy.

Usage (on a machine with NVIDIA GPU):
    pip install unsloth trl peft datasets bitsandbytes
    python training/format_sft.py

Outputs:
    training/format_sft_adapters/    ← LoRA adapter weights
    training/format_compliance_report.txt  ← before/after format rates
"""

# Unsloth MUST be imported first before trl/transformers/peft
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
except ImportError:
    raise SystemExit(
        "ERROR: Unsloth not found. Install with:\n"
        "  pip install unsloth trl peft datasets bitsandbytes\n"
        "This script requires an NVIDIA GPU (e.g., Google Colab T4)."
    )

import os
import sys
import json
import random
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

from server.generators import math_gen, code_gen, logic_gen
from calibration_profiles import prompt_templates

# Format-SFT teaches the strict-XML contract that the rest of the pipeline
# enforces. We use the "required" reasoning mode here even though training
# may run in "optional" — the assistant target text already includes the
# canonical format, so seeing the strict-mode user template at SFT time
# only reinforces it without changing what the model has to emit.
SYSTEM_PROMPT, USER_TEMPLATE = prompt_templates("required")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID       = "Qwen/Qwen2.5-3B-Instruct"
MAX_SEQ_LEN    = 1024
N_EXAMPLES     = 500
LORA_R         = 8           # Low rank — we only want format compliance
LR             = 2e-5        # Small LR, gentle nudge
N_EPOCHS       = 2
BATCH_SIZE     = 2
GRAD_ACCUM     = 4           # effective batch = 8
ADAPTERS_DIR   = str(PROJECT_ROOT / "training" / "format_sft_adapters")
REPORT_PATH    = str(PROJECT_ROOT / "training" / "format_compliance_report.txt")

GENERATORS = [
    ("math",  math_gen.generate),
    ("code",  code_gen.generate),
    ("logic", logic_gen.generate),
]

# Varied confidence temperatures so model doesn't memorize a single value
CONFIDENCE_BUCKETS = {
    1: [0.9, 0.9, 0.9, 0.7],        # easy: mostly high conf
    2: [0.9, 0.7, 0.7, 0.5],        # medium-easy
    3: [0.7, 0.5, 0.5, 0.3],        # medium-hard
    4: [0.5, 0.3, 0.3, 0.1],        # hard
    5: [0.3, 0.1, 0.1, 0.1],        # very hard
}


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def build_dataset(n: int) -> Dataset:
    """Generate n format-compliance training examples."""
    records = []
    rng = random.Random(42)

    print(f"Generating {n} training examples...")
    attempts = 0
    while len(records) < n and attempts < n * 5:
        attempts += 1
        domain_name, gen_fn = rng.choice(GENERATORS)
        difficulty = rng.randint(1, 5)
        seed = 800_000 + attempts

        try:
            question, ground_truth = gen_fn(difficulty, seed=seed)
        except (RuntimeError, Exception):
            continue

        conf_choices = CONFIDENCE_BUCKETS[difficulty]
        conf = rng.choice(conf_choices)

        # Mix correct answers (use ground truth) with intentionally wrong ones
        # Wrong answers with appropriate low confidence teach the model
        # it's okay to be uncertain
        use_correct = rng.random() < 0.65   # 65% correct, 35% wrong
        if use_correct:
            answer = ground_truth
        else:
            # Generate a plausible-looking wrong answer
            if domain_name == "math":
                try:
                    numeric = int(ground_truth)
                    offset = rng.choice([-2, -1, 1, 2, 5, -5, 10, -10])
                    answer = str(numeric + offset)
                except ValueError:
                    answer = "unknown"
            elif domain_name == "code":
                try:
                    numeric = int(ground_truth)
                    answer = str(numeric + rng.choice([1, -1, 2, -2]))
                except ValueError:
                    answer = "None"
            else:
                answer = rng.choice(["A", "B", "C", "D", "First", "Second"])
            # Wrong answer → cap confidence low so model learns calibration
            conf = min(conf, 0.5)

        # Occasionally teach abstain on the hardest problems
        if difficulty == 5 and rng.random() < 0.12:
            assistant_text = "<abstain/>"
        else:
            assistant_text = f"<answer>{answer}</answer><confidence>{conf}</confidence>"

        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_TEMPLATE.format(question=question)},
                {"role": "assistant", "content": assistant_text},
            ]
        })

    print(f"  → Generated {len(records)} examples ({attempts} attempts)")
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Format compliance evaluation (quick, no model needed)
# ---------------------------------------------------------------------------

FORMAT_RE = re.compile(
    r"<answer>.*?</answer><confidence>[01](\.\d+)?</confidence>|<abstain/>",
    re.DOTALL,
)

def quick_format_eval(model, tokenizer, n_probe: int = 30) -> float:
    """
    Run a quick forward pass on n_probe random questions and return
    the fraction whose raw output matches the required format.
    """
    import torch
    FastLanguageModel.for_inference(model)

    rng = random.Random(99)
    ok = 0
    for i in range(n_probe):
        domain_name, gen_fn = rng.choice(GENERATORS)
        difficulty = rng.randint(1, 3)   # probe easy/medium only for speed
        try:
            question, _ = gen_fn(difficulty, seed=700_000 + i)
        except Exception:
            continue

        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": USER_TEMPLATE.format(question=question)},
        ]
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            out = model.generate(
                input_ids=ids,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][ids.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if FORMAT_RE.search(text):
            ok += 1

    rate = ok / n_probe
    FastLanguageModel.for_training(model)
    return rate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(ADAPTERS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    # ── 1. Load model ────────────────────────────────────────────────────────
    print(f"\nLoading {MODEL_ID} with Unsloth (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )

    # ── 2. Attach small LoRA adapters ────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_R,  # alpha = r → no scaling gain
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ── 3. Before-SFT compliance rate ────────────────────────────────────────
    print("\nMeasuring BEFORE-SFT format compliance (30 probes)...")
    before_rate = quick_format_eval(model, tokenizer, n_probe=30)
    print(f"  Before SFT format rate: {before_rate:.1%}")

    # ── 4. Build dataset ─────────────────────────────────────────────────────
    raw_dataset = build_dataset(N_EXAMPLES)

    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    def apply_template(examples):
        return {"text": [
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            for msgs in examples["messages"]
        ]}

    dataset = raw_dataset.map(apply_template, batched=True)

    # ── 5. Train ─────────────────────────────────────────────────────────────
    total_steps = (N_EXAMPLES // (BATCH_SIZE * GRAD_ACCUM)) * N_EPOCHS
    print(f"\nStarting format SFT: {N_EPOCHS} epochs, ~{total_steps} steps...")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            num_train_epochs=N_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_ratio=0.05,
            learning_rate=LR,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir=ADAPTERS_DIR,
            report_to="none",
            save_strategy="no",
        ),
    )

    trainer.train()

    # ── 6. After-SFT compliance rate ─────────────────────────────────────────
    print("\nMeasuring AFTER-SFT format compliance (30 probes)...")
    after_rate = quick_format_eval(model, tokenizer, n_probe=30)
    print(f"  After SFT format rate: {after_rate:.1%}")

    # ── 7. Save adapters ─────────────────────────────────────────────────────
    print(f"\nSaving LoRA adapters to {ADAPTERS_DIR} ...")
    model.save_pretrained(ADAPTERS_DIR)
    tokenizer.save_pretrained(ADAPTERS_DIR)

    # ── 8. Write compliance report ───────────────────────────────────────────
    report = (
        f"Format SFT Compliance Report\n"
        f"{'='*40}\n"
        f"Model:          {MODEL_ID}\n"
        f"Training steps: {total_steps}\n"
        f"LoRA rank:      {LORA_R}\n"
        f"N examples:     {N_EXAMPLES}\n"
        f"\n"
        f"Before SFT format rate: {before_rate:.1%}\n"
        f"After  SFT format rate: {after_rate:.1%}\n"
        f"Improvement:            {(after_rate - before_rate):.1%}\n"
        f"\n"
        f"{'PASS' if after_rate >= 0.95 else 'NEEDS MORE TRAINING'}: "
        f"Target is ≥95% format compliance.\n"
    )
    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(f"\n{report}")
    print(f"Report saved to {REPORT_PATH}")
    print("Done! You can now load the adapters for RL training.")


if __name__ == "__main__":
    main()
