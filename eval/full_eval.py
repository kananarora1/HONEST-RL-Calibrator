"""Full evaluation pipeline for HONEST-RL-Calibrator.

Runs in-distribution (math/code/logic) and OOD (medical/legal) evaluation,
computes the full metric suite (Brier, ECE, ACE, MCE) before and after GRPO,
and produces side-by-side reliability diagrams.

Usage:
    # After baseline_eval has been run to produce baseline_results.json:
    python eval/full_eval.py \\
        --baseline-results eval/baseline_results.json \\
        --adapter-path ./honest-qwen-3b-grpo/final_adapters \\
        --ood-dir eval/ood \\
        --output eval/full_results.json

    # Dry-run (no model needed — uses stub responses):
    python eval/full_eval.py --dry-run
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from server.reward import (  # noqa: E402
    FORMAT_BONUS,
    compute_reward,
    parse_action,
    parse_action_lenient,
)
from server.verifier import verify_mcq                           # noqa: E402
from eval.metrics import compute_brier, compute_ece, compute_ace, compute_mce  # noqa: E402
from server.generators import code_gen, logic_gen, math_gen      # noqa: E402
from calibration_profiles import (  # noqa: E402
    REASONING_MODES,
    SUPPORTED_PRESETS,
    get_preset,
    prompt_templates,
)


# ---------------------------------------------------------------------------
# OOD-specific prompt
#
# The default HONEST prompt is domain-agnostic and leaves the answer format
# open-ended. OOD datasets (MMLU / AGIEval) are strictly multiple-choice, so
# we give the model a narrower contract: reasoning + a single letter answer
# + a confidence value, all wrapped in the same XML tags the strict parser
# already expects. This raises strict-format rate without changing any
# training-time behaviour.
# ---------------------------------------------------------------------------

OOD_USER_TEMPLATE = (
    "{question}\n\n"
    "This is a multiple-choice question. Choose ONE option.\n"
    "Respond in EXACTLY this format:\n"
    "<reasoning>one or two sentences</reasoning>\n"
    "<answer>LETTER</answer>\n"
    "<confidence>0.XX</confidence>\n"
    "LETTER must be a single letter from A-E. "
    "Confidence must be a number between 0.0 and 1.0."
)

GENERATORS = {
    "math":  math_gen.generate,
    "code":  code_gen.generate,
    "logic": logic_gen.generate,
}

DOMAINS      = ["math", "code", "logic"]
DIFFICULTIES = [1, 2, 3, 4, 5]

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str, adapter_path: Optional[str], device: str):
    """Load model + optional LoRA adapter. Returns (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer: {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map=device,
        trust_remote_code=True,
    )

    if adapter_path and Path(adapter_path).exists():
        print(f"Loading LoRA adapter from {adapter_path} ...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("Adapter merged.")

    model.eval()
    print("Model ready.\n")
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
    import torch
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_template.format(question=question)},
    ]
    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Per-condition evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_records(records: list) -> dict:
    """Aggregate a list of per-sample dicts into condition-level metrics."""
    answered   = [r for r in records if r["correct"] is not None]
    confs      = [r["confidence"] for r in answered if r["confidence"] is not None]
    corrects   = [1 if r["correct"] else 0 for r in answered if r["confidence"] is not None]
    n          = len(records)

    return {
        "n_samples":       n,
        "accuracy":        round(len([r for r in answered if r["correct"]]) / n, 4) if n else 0.0,
        "format_rate":     round(sum(1 for r in records if r["format_valid"]) / n, 4) if n else 0.0,
        "mean_confidence": round(sum(confs) / len(confs), 4) if confs else 0.0,
        "mean_reward":     round(sum(r["reward"] for r in records) / n, 4) if n else 0.0,
        "brier":           round(compute_brier(confs, corrects), 4),
        "ece":             round(compute_ece(confs, corrects), 4),
        "ace":             round(compute_ace(confs, corrects), 4),
        "mce":             round(compute_mce(confs, corrects), 4),
        "n_correct":       len([r for r in answered if r["correct"]]),
        "n_answered":      len(answered),
        "n_malformed":     sum(1 for r in records if r["parsed_type"] == "malformed"),
        "n_abstain":       sum(1 for r in records if r["parsed_type"] == "abstain"),
        "samples":         records,
    }


def run_indist_eval(
    model,
    tokenizer,
    n_samples: int,
    system_prompt: str,
    user_template: str,
    max_new_tokens: int,
    response_fn=None,
) -> dict:
    """Run in-distribution evaluation across all 15 (domain, difficulty) conditions."""
    _generate = response_fn or generate_response
    conditions = {}

    total = len(DOMAINS) * len(DIFFICULTIES)
    idx   = 0
    for domain in DOMAINS:
        for difficulty in DIFFICULTIES:
            idx += 1
            key = f"{domain}_{difficulty}"
            print(f"  [{idx}/{total}] In-dist {key} ({n_samples} samples)...", end=" ", flush=True)
            t0 = time.time()

            records = []
            for i in range(n_samples):
                seed     = (difficulty * 1000) + i
                question, ground_truth = GENERATORS[domain](difficulty, seed=seed)
                raw = _generate(
                    model,
                    tokenizer,
                    question,
                    system_prompt=system_prompt,
                    user_template=user_template,
                    max_new_tokens=max_new_tokens,
                )
                parsed   = parse_action(raw)
                correct: Optional[bool] = None
                confidence: Optional[float] = None

                reward, correct = compute_reward(parsed, ground_truth, difficulty, domain=domain)

                if parsed["type"] == "answer":
                    confidence = parsed["confidence"]
                elif parsed["type"] == "abstain":
                    confidence = 0.0

                records.append({
                    "question":     question[:120],
                    "ground_truth": ground_truth,
                    "raw_response": raw[:200],
                    "parsed_type":  parsed["type"],
                    "correct":      correct,
                    "confidence":   confidence,
                    "reward":       reward,
                    "format_valid": parsed["type"] in ("answer", "abstain"),
                })

            result = _evaluate_records(records)
            elapsed = time.time() - t0
            print(f"acc={result['accuracy']:.1%}  fmt={result['format_rate']:.1%}  "
                  f"reward={result['mean_reward']:.3f}  [{elapsed:.1f}s]")
            conditions[key] = result

    return conditions


def run_ood_eval(
    model,
    tokenizer,
    ood_dir: Path,
    system_prompt: str,
    user_template: str,  # kept for backward-compat; OOD uses its own template
    max_new_tokens: int,
    response_fn=None,
) -> dict:
    """Run OOD evaluation on medical and legal jsonl files.

    Differences vs in-distribution eval (intentional, OOD-only):
      * Uses an MCQ-specific user template so the model knows to emit a
        single letter answer in the HONEST XML format.
      * Parses with ``parse_action_lenient`` — still tries the strict
        contract first, then falls back to a best-effort recovery so prose
        answers still contribute to accuracy / calibration metrics.
      * Grades with ``verify_mcq`` so letter-vs-index mismatches between
        the jsonl ground-truth and the model's answer don't count as wrong.
      * ``format_valid`` remains defined as "strict parser succeeded", so
        the format-rate metric stays honest and comparable with training.
    """
    _generate = response_fn or generate_response
    results   = {}

    for domain, fname in [("medical", "medqa_sample.jsonl"), ("legal", "lsat_sample.jsonl")]:
        fpath = ood_dir / fname
        if not fpath.exists():
            print(f"  OOD file not found: {fpath} — run eval/ood/fetch_ood_data.py first.")
            continue

        with open(fpath, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

        print(f"  OOD [{domain}] {len(rows)} samples...", end=" ", flush=True)
        t0 = time.time()
        records = []

        for row in rows:
            question     = row["question"]
            ground_truth = row["answer"]
            raw = _generate(
                model,
                tokenizer,
                question,
                system_prompt=system_prompt,
                user_template=OOD_USER_TEMPLATE,
                max_new_tokens=max_new_tokens,
            )

            parsed = parse_action_lenient(raw)
            parsed_mode = parsed.get("parsed_mode")  # "strict" | "lenient" | "lenient_default_conf" | None
            strict_valid = parsed_mode == "strict" and parsed["type"] in ("answer", "abstain")

            correct: Optional[bool] = None
            confidence: Optional[float] = None

            if parsed["type"] == "answer":
                correct = verify_mcq(parsed["answer"], ground_truth)
                confidence = float(parsed["confidence"])
                target = 1.0 if correct else 0.0
                brier = -1.5 * ((confidence - target) ** 2)
                # Only grant the strict-format bonus when the strict parser accepted the output.
                reward = brier + (FORMAT_BONUS if strict_valid else 0.0)
            elif parsed["type"] == "abstain":
                confidence = 0.0
                reward = 0.0
            else:
                reward = -1.0  # MALFORMED_PENALTY, preserves prior semantics

            records.append({
                "question":     question[:200],
                "ground_truth": ground_truth,
                "raw_response": raw[:200],
                "parsed_type":  parsed["type"],
                "parsed_mode":  parsed_mode,  # extra bookkeeping; safe to ignore
                "recovered":    parsed_mode in ("lenient", "lenient_default_conf"),
                "correct":      correct,
                "confidence":   confidence,
                "reward":       reward,
                "format_valid": strict_valid,
                "source":       row.get("source", domain),
            })

        result  = _evaluate_records(records)
        # Lightweight recovery stats for printed summary (non-breaking).
        n_recovered = sum(1 for r in records if r.get("recovered"))
        elapsed = time.time() - t0
        print(
            f"acc={result['accuracy']:.1%}  fmt(strict)={result['format_rate']:.1%}  "
            f"recovered={n_recovered}/{len(records)}  "
            f"brier={result['brier']:.4f}  ece={result['ece']:.4f}  [{elapsed:.1f}s]"
        )
        results[domain] = result

    return results


# ---------------------------------------------------------------------------
# Summary + diff table
# ---------------------------------------------------------------------------

def print_comparison(baseline: dict, after: dict, section: str = "In-Distribution"):
    """Print a before/after comparison table."""
    print(f"\n{'─'*80}")
    print(f"  {section}")
    print(f"{'─'*80}")
    header = f"{'Condition':<18} {'Brier(before)':>14} {'Brier(after)':>13} "  \
             f"{'ECE(before)':>12} {'ECE(after)':>11} {'ΔBrier':>8}"
    print(header)
    print("─" * 80)

    for key in sorted(set(list(baseline.keys()) + list(after.keys()))):
        b = baseline.get(key, {})
        a = after.get(key, {})
        b_brier = b.get("brier", float("nan"))
        a_brier = a.get("brier", float("nan"))
        b_ece   = b.get("ece",   float("nan"))
        a_ece   = a.get("ece",   float("nan"))
        delta   = a_brier - b_brier
        symbol  = "↓" if delta < 0 else "↑"
        print(
            f"{key:<18} {b_brier:>14.4f} {a_brier:>13.4f} "
            f"{b_ece:>12.4f} {a_ece:>11.4f} {delta:>+7.4f}{symbol}"
        )
    print("─" * 80)


# ---------------------------------------------------------------------------
# Reliability diagram generation
# ---------------------------------------------------------------------------

def generate_reliability_plots(full_results: dict, output_dir: Path):
    """Generate before/after reliability diagrams using plot_reliability.py."""
    try:
        from eval.plot_reliability import plot_comparison
        baseline_path = full_results.get("_baseline_path")
        after_path    = full_results.get("_after_path")
        if baseline_path and after_path:
            out = plot_comparison(
                baseline_path, after_path,
                label_before="Before GRPO (Baseline)",
                label_after="After GRPO Training",
            )
            print(f"\nReliability diagram saved: {out}")
    except Exception as e:
        print(f"\n(Skipping reliability plot: {e})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Full HONEST-RL evaluation pipeline")
    parser.add_argument("--model-id",          type=str,  default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-path",      type=str,  default=None,
                        help="Path to trained LoRA adapter dir (merged into base model)")
    parser.add_argument("--baseline-results",  type=str,  default="eval/baseline_results.json",
                        help="Existing baseline_results.json for comparison table")
    parser.add_argument("--ood-dir",           type=str,  default="eval/ood")
    parser.add_argument("--output",            type=str,  default="eval/full_results.json")
    parser.add_argument("--samples",           type=int,  default=20)
    parser.add_argument("--device",            type=str,  default="auto")
    parser.add_argument("--max-new-tokens",    type=int,  default=512)
    parser.add_argument(
        "--model-preset",
        choices=["auto", *SUPPORTED_PRESETS],
        default="auto",
        help="Model calibration preset metadata; auto infers from --model-id.",
    )
    parser.add_argument(
        "--reasoning-mode",
        choices=list(REASONING_MODES),
        default="required",
        help="Prompt mode used at evaluation time.",
    )
    parser.add_argument("--skip-indist",       action="store_true",
                        help="Skip in-distribution eval (OOD only)")
    parser.add_argument("--skip-ood",         action="store_true")
    parser.add_argument("--dry-run",           action="store_true",
                        help="Use stub inferencer — no GPU needed")
    args = parser.parse_args()
    preset = get_preset(args.model_id, args.model_preset)
    system_prompt, user_template = prompt_templates(args.reasoning_mode)

    # Stub response function for dry-run
    if args.dry_run:
        print("DRY-RUN mode (stub responses)\n")
        model, tokenizer = None, None
        response_fn = lambda m, t, q, **kw: "<answer>A</answer><confidence>0.7</confidence>"
    else:
        model, tokenizer = load_model(args.model_id, args.adapter_path, args.device)
        response_fn = None

    output = {
        "model_id":     args.model_id,
        "preset":       preset.name,
        "reasoning_mode": args.reasoning_mode,
        "max_new_tokens": args.max_new_tokens,
        "adapter_path": args.adapter_path,
        "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # ── In-distribution ───────────────────────────────────────────────────────
    if not args.skip_indist:
        print("\n── In-distribution evaluation ──────────────────────────────────────")
        indist = run_indist_eval(
            model,
            tokenizer,
            args.samples,
            system_prompt=system_prompt,
            user_template=user_template,
            max_new_tokens=args.max_new_tokens,
            response_fn=response_fn,
        )
        output["in_distribution"] = indist

        # Load baseline for comparison
        bp = Path(args.baseline_results)
        if bp.exists():
            with open(bp) as f:
                baseline_data = json.load(f)
            baseline_conds = baseline_data.get("conditions", {})
            baseline_mode = baseline_data.get("reasoning_mode")
            if baseline_mode and baseline_mode != args.reasoning_mode:
                print(
                    f"⚠ Baseline reasoning_mode={baseline_mode} differs from current "
                    f"run reasoning_mode={args.reasoning_mode}. Comparison may be biased."
                )
            print_comparison(baseline_conds, indist)
        else:
            print(f"\n(No baseline file at {bp} — skipping comparison table)")

    # ── OOD ──────────────────────────────────────────────────────────────────
    if not args.skip_ood:
        print("\n── OOD evaluation ──────────────────────────────────────────────────")
        ood_results = run_ood_eval(
            model,
            tokenizer,
            Path(args.ood_dir),
            system_prompt=system_prompt,
            user_template=user_template,
            max_new_tokens=args.max_new_tokens,
            response_fn=response_fn,
        )
        output["ood"] = ood_results

    # ── Overall ───────────────────────────────────────────────────────────────
    all_confs, all_corrects = [], []
    for section in ["in_distribution", "ood"]:
        section_data = output.get(section, {})
        for cond in section_data.values():
            for s in cond.get("samples", []):
                if s["confidence"] is not None:
                    all_confs.append(s["confidence"])
                    all_corrects.append(1 if s["correct"] else 0)

    if all_confs:
        all_rewards = []
        total = 0
        total_correct = 0
        total_format_valid = 0
        for section in ["in_distribution", "ood"]:
            section_data = output.get(section, {})
            for cond in section_data.values():
                for s in cond.get("samples", []):
                    total += 1
                    total_correct += 1 if s["correct"] is True else 0
                    total_format_valid += 1 if s["format_valid"] else 0
                    all_rewards.append(float(s["reward"]))
        output["overall"] = {
            "n_samples": len(all_confs),
            "brier":     round(compute_brier(all_confs, all_corrects), 4),
            "ece":       round(compute_ece(all_confs, all_corrects), 4),
            "ace":       round(compute_ace(all_confs, all_corrects), 4),
            "mce":       round(compute_mce(all_confs, all_corrects), 4),
            "accuracy": round((total_correct / total) if total else 0.0, 4),
            "format_rate": round((total_format_valid / total) if total else 0.0, 4),
            "mean_reward": round((sum(all_rewards) / len(all_rewards)) if all_rewards else 0.0, 4),
        }
        o = output["overall"]
        print(f"\n── Overall ─ n={o['n_samples']}  "
              f"Acc={o['accuracy']:.1%}  Fmt={o['format_rate']:.1%}  "
              f"Brier={o['brier']:.4f}  ECE={o['ece']:.4f}  "
              f"ACE={o['ace']:.4f}  MCE={o['mce']:.4f}  Reward={o['mean_reward']:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull results saved -> {out_path}")

    # ── Reliability plots ─────────────────────────────────────────────────────
    bp = Path(args.baseline_results)
    if bp.exists():
        output["_baseline_path"] = str(bp)
        output["_after_path"]    = str(out_path)
        generate_reliability_plots(output, out_path.parent)


if __name__ == "__main__":
    main()