"""Baseline evaluation for calibration-focused HONEST training.

Evaluates 3 domains × 5 difficulties = 15 conditions, configurable samples each.
Results are saved to eval/baseline_results.json.

Usage:
    python eval/baseline_eval.py [--samples N] [--output PATH] [--device cuda|cpu]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ---- path setup so we can import server/models from project root ----------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from server.generators import code_gen, logic_gen, math_gen  # noqa: E402
from server.reward import compute_reward, parse_action  # noqa: E402
from eval.metrics import compute_brier, compute_ece, compute_ace, compute_mce  # noqa: E402
from calibration_profiles import (  # noqa: E402
    REASONING_MODES,
    SUPPORTED_PRESETS,
    get_preset,
    prompt_templates,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DOMAINS = ["math", "code", "logic"]
DIFFICULTIES = [1, 2, 3, 4, 5]
DEFAULT_SAMPLES = 20

GENERATORS = {
    "math": math_gen.generate,
    "code": code_gen.generate,
    "logic": logic_gen.generate,
}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str, device: str):
    """Load model and tokenizer. Returns (model, tokenizer)."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: Install with: pip install transformers accelerate torch")
        sys.exit(1)

    print(f"Loading tokenizer: {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print(f"Loading model: {model_id} (device={device}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.\n")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate_response(
    model,
    tokenizer,
    question: str,
    system_prompt: str,
    user_template: str,
    max_new_tokens: int = 128,
) -> str:
    """Run one inference pass using the chat template."""
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_template.format(question=question)},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy — deterministic baseline
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only newly generated tokens
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Evaluate one condition
# ---------------------------------------------------------------------------

def evaluate_condition(
    model,
    tokenizer,
    domain: str,
    difficulty: int,
    n_samples: int,
    system_prompt: str,
    user_template: str,
    max_new_tokens: int,
    verbose: bool = False,
    response_fn=None,
) -> dict:
    """Run n_samples problems for one (domain, difficulty) condition."""
    generator = GENERATORS[domain]
    _generate = response_fn if response_fn is not None else generate_response

    records = []
    for i in range(n_samples):
        seed = (difficulty * 1000) + i
        question, ground_truth = generator(difficulty, seed=seed)

        raw = _generate(
            model,
            tokenizer,
            question,
            system_prompt=system_prompt,
            user_template=user_template,
            max_new_tokens=max_new_tokens,
        )
        parsed = parse_action(raw)

        format_valid = parsed["type"] in ("answer", "abstain")
        correct: Optional[bool] = None
        confidence: Optional[float] = None

        reward, correct = compute_reward(parsed, ground_truth, difficulty, domain=domain)

        if parsed["type"] == "answer":
            confidence = parsed["confidence"]
        elif parsed["type"] == "abstain":
            confidence = 0.0  # treat abstain as zero-confidence for calibration

        records.append({
            "question": question[:120],
            "ground_truth": ground_truth,
            "raw_response": raw[:200],
            "parsed_type": parsed["type"],
            "correct": correct,
            "confidence": confidence,
            "reward": reward,
            "format_valid": format_valid,
        })

        if verbose:
            status = "✓" if correct else ("~" if correct is None else "✗")
            print(
                f"  [{domain}/{difficulty}] sample {i+1:02d}/{n_samples}: "
                f"{status} conf={confidence or 'n/a'} reward={reward:.3f}"
            )

    # --- aggregate ---
    answered = [r for r in records if r["correct"] is not None]
    correct_answers = [r for r in answered if r["correct"]]

    confidences_all = [r["confidence"] for r in answered if r["confidence"] is not None]
    correctness_all = [1 if r["correct"] else 0 for r in answered if r["confidence"] is not None]

    accuracy = len(correct_answers) / n_samples if n_samples > 0 else 0.0
    format_rate = sum(1 for r in records if r["format_valid"]) / n_samples
    mean_conf = float(sum(confidences_all) / len(confidences_all)) if confidences_all else 0.0
    mean_reward = float(sum(r["reward"] for r in records) / len(records))

    return {
        "n_samples": n_samples,
        "accuracy": round(accuracy, 4),
        "format_rate": round(format_rate, 4),
        "mean_confidence": round(mean_conf, 4),
        "mean_reward": round(mean_reward, 4),
        "brier": round(compute_brier(confidences_all, correctness_all), 4),
        "ece": round(compute_ece(confidences_all, correctness_all), 4),
        "ace": round(compute_ace(confidences_all, correctness_all), 4),
        "mce": round(compute_mce(confidences_all, correctness_all), 4),
        "n_correct": len(correct_answers),
        "n_answered": len(answered),
        "n_malformed": sum(1 for r in records if r["parsed_type"] == "malformed"),
        "n_abstain": sum(1 for r in records if r["parsed_type"] == "abstain"),
        "samples": records,
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(conditions: dict):
    header = f"{'Condition':<14} {'Acc':>6} {'FmtOK':>6} {'MeanConf':>9} {'Reward':>8} {'ECE':>7} {'Brier':>7}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    all_acc, all_conf, all_reward = [], [], []
    all_confidences, all_correctness = [], []

    for key in sorted(conditions):
        c = conditions[key]
        flag = ""
        if c["accuracy"] == 0.0:
            flag = " ⚠ ZERO"
        elif c["format_rate"] < 0.70:
            flag = " ⚠ FMT"
        print(
            f"{key:<14} {c['accuracy']:>6.1%} {c['format_rate']:>6.1%} "
            f"{c['mean_confidence']:>9.3f} {c['mean_reward']:>8.3f} "
            f"{c['ece']:>7.4f} {c['brier']:>7.4f}{flag}"
        )
        all_acc.append(c["accuracy"])
        all_conf.append(c["mean_confidence"])
        all_reward.append(c["mean_reward"])
        # Collect answered samples for global calibration
        for s in c.get("samples", []):
            if s["confidence"] is not None:
                all_confidences.append(s["confidence"])
                all_correctness.append(1 if s["correct"] else 0)

    print(sep)
    print(
        f"{'OVERALL':<14} {sum(all_acc)/len(all_acc):>6.1%} {'':>6} "
        f"{sum(all_conf)/len(all_conf):>9.3f} "
        f"{sum(all_reward)/len(all_reward):>8.3f} "
        f"{compute_ece(all_confidences, all_correctness):>7.4f} "
        f"{compute_brier(all_confidences, all_correctness):>7.4f}"
    )
    print(sep)

    # Decision guidance
    zero_conds = [k for k, v in conditions.items() if v["accuracy"] == 0.0]
    fmt_fail = [k for k, v in conditions.items() if v["format_rate"] < 0.70]
    print()
    if zero_conds:
        print(f"⚠  ZERO accuracy:  {', '.join(zero_conds)}")
        print("   → Consider reducing difficulty ceiling or prompting differently.")
    if fmt_fail:
        print(f"⚠  Low format rate: {', '.join(fmt_fail)}")
        print("   → Consider format-SFT before RL (Step 3.2).")
    if not zero_conds and not fmt_fail:
        print("✓  All conditions pass (accuracy > 0, format rate ≥ 70%).")
        print("   → Ready for RL training (Step 3.4).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Baseline eval for HONEST-RL-Calibrator")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help="Problems per condition (default: 20)")
    parser.add_argument("--output", type=str, default="eval/baseline_results.json",
                        help="Output JSON path")
    parser.add_argument("--device", type=str, default="auto",
                        help="Torch device: auto, cuda, cpu")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-sample status")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip model loading; generate dummy responses for testing")
    parser.add_argument("--model", type=str, default=MODEL_ID,
                        help=f"Model ID or local path to evaluate (default: {MODEL_ID})")
    parser.add_argument(
        "--model-preset",
        choices=["auto", *SUPPORTED_PRESETS],
        default="auto",
        help="Model calibration preset metadata; auto infers from --model.",
    )
    parser.add_argument(
        "--reasoning-mode",
        choices=list(REASONING_MODES),
        default="required",
        help="Prompt mode to match training/eval style.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Generation token budget for evaluation.")
    parser.add_argument("--omit-samples", action="store_true",
                        help="Do not include per-sample rows in output json (smaller file).")
    args = parser.parse_args()

    # Allow CLI override of model — used to eval post-RL merged model
    eval_model_id = args.model
    preset = get_preset(eval_model_id, args.model_preset)
    system_prompt, user_template = prompt_templates(args.reasoning_mode)

    if args.dry_run:
        print("DRY-RUN mode: using stub response fn (correct format, fixed answer).\n")
        model, tokenizer = None, None
        response_fn = lambda m, t, q, **kw: "<answer>42</answer><confidence>0.7</confidence>"  # noqa: E731
    else:
        model, tokenizer = load_model(eval_model_id, args.device)
        response_fn = None  # use real generate_response

    conditions = {}
    total = len(DOMAINS) * len(DIFFICULTIES)
    idx = 0

    for domain in DOMAINS:
        for difficulty in DIFFICULTIES:
            idx += 1
            key = f"{domain}_{difficulty}"
            print(f"[{idx}/{total}] Evaluating {key} ({args.samples} samples) ...")
            t0 = time.time()
            result = evaluate_condition(
                model, tokenizer, domain, difficulty, args.samples,
                system_prompt=system_prompt,
                user_template=user_template,
                max_new_tokens=args.max_new_tokens,
                verbose=args.verbose,
                response_fn=response_fn,
            )
            elapsed = time.time() - t0
            print(
                f"      acc={result['accuracy']:.1%}  fmt={result['format_rate']:.1%}  "
                f"reward={result['mean_reward']:.3f}  elapsed={elapsed:.1f}s"
            )
            conditions[key] = result

    # --- global aggregation ---
    all_confidences, all_correctness = [], []
    all_rewards = []
    total_correct = 0
    total_format_valid = 0
    total_samples = 0
    for c in conditions.values():
        for s in c.get("samples", []):
            total_samples += 1
            total_format_valid += 1 if s["format_valid"] else 0
            total_correct += 1 if s["correct"] is True else 0
            all_rewards.append(float(s["reward"]))
            if s["confidence"] is not None:
                all_confidences.append(s["confidence"])
                all_correctness.append(1 if s["correct"] else 0)

    overall = {
        "accuracy": round((total_correct / total_samples) if total_samples else 0.0, 4),
        "format_rate": round((total_format_valid / total_samples) if total_samples else 0.0, 4),
        "ece": round(compute_ece(all_confidences, all_correctness), 4),
        "ace": round(compute_ace(all_confidences, all_correctness), 4),
        "mce": round(compute_mce(all_confidences, all_correctness), 4),
        "brier": round(compute_brier(all_confidences, all_correctness), 4),
        "mean_reward": round((sum(all_rewards) / len(all_rewards)) if all_rewards else 0.0, 4),
        "n_answered": len(all_confidences),
        "n_total": total_samples,
    }

    if args.omit_samples:
        for c in conditions.values():
            c.pop("samples", None)

    output = {
        "model": eval_model_id,
        "preset": preset.name,
        "reasoning_mode": args.reasoning_mode,
        "max_new_tokens": args.max_new_tokens,
        "n_samples_per_condition": args.samples,
        "conditions": conditions,
        "overall": overall,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}\n")

    print_summary(conditions)
    print(f"\nOverall: accuracy={overall['accuracy']:.1%}  ECE={overall['ece']:.4f}  Brier={overall['brier']:.4f}  Reward={overall['mean_reward']:.3f}")


if __name__ == "__main__":
    main()
