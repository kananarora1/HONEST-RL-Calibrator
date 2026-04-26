# HONEST · Operational Runbook

End-to-end pipeline: **data → RL training → OOD eval → comparison →
success metrics → MCP deployment → self-learning verification**.

Every step is a single command. Every command produces a JSON or
markdown artifact that the next step consumes. There are no implicit
dependencies between steps; each one fails fast if its inputs are
missing.

```
┌───────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│ 1 · Ingestion │─▶│ 2 · Baseline   │─▶│ 3 · GRPO train │─▶│ 4 · Full eval  │
│   (data/)     │  │   (anchor)     │  │   (LoRA out)   │  │  (ID + OOD)    │
└───────────────┘  └────────────────┘  └────────────────┘  └────────────────┘
                                                                    │
┌───────────────┐  ┌────────────────┐  ┌────────────────┐           │
│ 7 · Self-     │◀─│ 6 · MCP serve  │◀─│ 5 · Compare    │◀──────────┘
│   learning    │  │   (deploy)     │  │   (Δ + CI)     │
└───────────────┘  └────────────────┘  └────────────────┘
```

---

## 0 · Prerequisites

```bash
# From the project root
python3 -m venv venv
venv/bin/pip install -r requirements.txt

# Optional: log in to W&B and HuggingFace
venv/bin/wandb login
venv/bin/huggingface-cli login

# Activate (we'll prefix commands with ./venv/bin/python below)
source venv/bin/activate          # POSIX shells
```

Verify the environment is healthy before running anything expensive:

```bash
make test                # 438 unit + integration tests should pass
make smoke-train         # train_grpo --dry-run --hindsight --replay-priority --self-mutate --self-play
make mcp-smoke           # offline MCP self-test
```

---

## 1 · Data ingestion

The unified sampler reads `data/processed/{math,code_mbpp,code_apps,logic}.jsonl`.
If any of those files is missing, `train_grpo.py` fails fast at startup
with a clear error.

```bash
# Math (Hendrycks MATH, 7 subjects, 5 difficulty levels)
PYTHONPATH=. python data/ingestion/ingest_hendrycks_math.py
# → data/processed/math.jsonl  (~12.5k problems)

# Code · MBPP (sandboxed verifier)
PYTHONPATH=. python data/ingestion/ingest_mbpp.py
# → data/processed/code_mbpp.jsonl  (~427 problems)

# Code · APPS (streamed JSONL shards from HuggingFace)
PYTHONPATH=. python -m data.ingestion.ingest_apps
# → data/processed/code_apps.jsonl

# Logic · ZebraLogic-style CSP puzzles (regenerated, unique-solution)
PYTHONPATH=. python data/ingestion/regenerate_zebralogic.py
# → data/processed/logic.jsonl  (~75 problems)
```

Every ingestion script prints a JSON summary on the last line — capture
it if you want to assert dataset shape in CI.

### Sanity check

```bash
./venv/bin/python -c "
from data.sampler.unified_sampler import get_sampler
s = get_sampler()
print('total:', s.total_count())
print('buckets (domain, difficulty):')
for k, v in sorted(s.bucket_counts().items()):
    print(' ', k, '=', v)
"
```

You should see ~13k problems spread across (math, 1..5),
(code, 3..5), (logic, 1..5). A bucket with `0` count will *not* error —
the controller will just never dispense that condition.

### OOD data (medical + legal)

OOD data is fetched from public HuggingFace datasets and is **never**
seen during training. Run this once per project setup.

```bash
PYTHONPATH=. python eval/ood/fetch_ood_data.py --n 200
# → eval/ood/medqa_sample.jsonl   (200 medical MCQs from MMLU professional_medicine)
# → eval/ood/lsat_sample.jsonl    (200 LSAT-LR logical-reasoning MCQs)
```

---

## 2 · Baseline characterization

Before training, anchor the **pre-RL** behaviour of the same model on
the same evaluation conditions. Without this anchor, the post-RL
numbers are uninterpretable.

```bash
# Default: 100 samples × 3 domains × 5 difficulties (~1500 generations)
./venv/bin/python eval/baseline_eval.py \
    --model         Qwen/Qwen2.5-3B-Instruct \
    --model-preset  qwen3b \
    --output        eval/baseline_results.json

# Headline run (200 samples per condition, ~3000 generations)
./venv/bin/python eval/baseline_eval.py \
    --model         Qwen/Qwen2.5-3B-Instruct \
    --model-preset  qwen3b \
    --samples       200 \
    --output        eval/baseline_results.json
```

Outputs (`eval/baseline_results.json`):

```jsonc
{
  "model_id":        "Qwen/Qwen2.5-3B-Instruct",
  "preset":          "qwen3b",
  "n_samples":       100,
  "metrics": {
    "ece": 0.18,    "ace": 0.21,    "mce": 0.42,
    "brier": 0.27,  "nll": 0.71,
    "auroc": 0.62,  "auprc": 0.55,
    "format_rate": 0.82,    "abstain_rate": 0.04
  },
  "per_domain":  { "math": {...}, "code": {...}, "logic": {...} },
  "per_difficulty": { "1": {...}, ..., "5": {...} }
}
```

If `format_rate < 0.70`, the strict XML parser is rejecting too many
completions. **Run the optional Stage-2 format SFT** first:

```bash
./venv/bin/python training/format_sft.py \
    --model-id     Qwen/Qwen2.5-3B-Instruct \
    --output-dir   ./honest-qwen-3b-sft
```

Then resume from the SFT adapter (`--resume-from ./honest-qwen-3b-sft`)
in Step 3.

---

## 3 · GRPO calibration training

Single command. Defaults come from `calibration_profiles.py`; CLI flags
override per-run.

### 3a · Vanilla GRPO (recommended starting point)

```bash
./venv/bin/python training/train_grpo.py \
    --model-preset    qwen3b \
    --colab-profile   l4 \
    --max-steps       350 \
    --output-dir      ./honest-qwen-3b-grpo
```

### 3b · GRPO + self-learning (recommended for the headline result)

```bash
./venv/bin/python training/train_grpo.py \
    --model-preset    qwen3b \
    --colab-profile   l4 \
    --max-steps       350 \
    --hindsight \
    --self-mutate \
    --replay-priority \
    --output-dir      ./honest-qwen-3b-grpo
```

`--self-play` is supported but disabled by default — turn it on only
after the first three pillars show a clean Δ ECE.

### 3c · What you get

```
honest-qwen-3b-grpo/
├── final_adapters/             ← LoRA adapter (load with --adapter-path)
├── difficulty_state.json       ← rolling controller state per domain
├── smc_state.json              ← (if --self-mutate) ceiling per domain
├── replay_state.json           ← (if --replay-priority) buffer snapshot
├── checkpoint-50/              ← intermediate checkpoints
├── checkpoint-100/
└── trainer_state.json
```

### 3d · Live monitoring

W&B (or stdout if `--no-wandb`) reports the metrics that matter for a
healthy GRPO run:

| Metric                             | Healthy range                      |
| ---------------------------------- | ---------------------------------- |
| `train/reward`                     | rising, plateauing > 0             |
| `train/reward_std`                 | > 1e-4 (else dead-batch guard fires)|
| `train/kl`                         | < 0.05 (else AdaptiveBetaCallback) |
| `train/grad_norm`                  | < `max_grad_norm = 1.0`            |
| `controller/<domain>/target`       | tracking accuracy band (0.30–0.70) |
| `smc/<domain>/max_unlocked`        | promotes only when ready (≥ d=6+)  |
| `replay/buffer_size`               | rises to ~ buffer_size after warmup|
| `replay/priority_entropy`          | > 1.0 (else replay disables itself)|

### 3e · Render committed training plots

After training, regenerate the committed evidence PNGs from
`trainer_state.json`:

```bash
make plots TRAINER_STATE=./honest-qwen-3b-grpo/trainer_state.json
# or directly:
./venv/bin/python bin/plot_training_curves.py \
    --trainer-state ./honest-qwen-3b-grpo/trainer_state.json \
    --out docs/training \
    --label "qwen3b · 350 steps · L4"
```

This overwrites `docs/training/loss_curve.png`, `reward_curve.png`,
and `kl_curve.png` — the same images embedded in the project README.
Commit them so the submission carries real training evidence:

```bash
git add docs/training/*.png
git commit -m "docs: refresh training curves from completed run"
```

### 3f · Multi-model recipe

For Llama-3B and Phi-4-mini (~3.8B), use L4 instead of A100:

```bash
# Llama-3.2-3B
./venv/bin/python training/train_grpo.py \
    --model-preset llama3b --colab-profile l4 \
    --max-steps 400 --hindsight --self-mutate \
    --output-dir ./honest-llama-3b-grpo

# Phi-4-mini-instruct
./venv/bin/python training/train_grpo.py \
    --model-preset phi4mini --colab-profile l4 \
    --max-steps 400 --hindsight --self-mutate \
    --output-dir ./honest-phi4mini-grpo
```

---

## 4 · Full evaluation (in-distribution + OOD)

Same metrics battery as Step 2, run on the **trained adapter**, with an
extra OOD pass on medical and legal questions.

```bash
./venv/bin/python eval/full_eval.py \
    --model-id          Qwen/Qwen2.5-3B-Instruct \
    --adapter-path      ./honest-qwen-3b-grpo/final_adapters \
    --baseline-results  eval/baseline_results.json \
    --ood-dir           eval/ood \
    --samples           100 \
    --output            eval/full_results.json
```

Output (`eval/full_results.json`):

```jsonc
{
  "model_id": "Qwen/Qwen2.5-3B-Instruct",
  "adapter":  "./honest-qwen-3b-grpo/final_adapters",
  "in_distribution": {
    "metrics":      { "ece": ..., "brier": ..., "auroc": ..., ... },
    "per_domain":     { "math": {...}, "code": {...}, "logic": {...} },
    "per_difficulty": { "1": {...}, ..., "5": {...} }
  },
  "ood": {
    "medical": { "metrics": { "ece": ..., "brier": ... } },
    "legal":   { "metrics": { "ece": ..., "brier": ... } }
  },
  "baseline_metrics": { ... }   // mirrored from Step 2 for the comparator
}
```

To skip ID or OOD individually: `--skip-indist` / `--skip-ood`.
To debug locally without a GPU: `--dry-run`.

---

## 5 · Comparison & success metrics

```bash
./venv/bin/python eval/compare_runs.py \
    --baseline    eval/baseline_results.json \
    --after       eval/full_results.json \
    --output      eval/comparison.md \
    --plot --plot-output eval/plots/comparison.png
```

`comparison.md` is the deliverable artefact — paste it directly into a
report or pitch deck. It contains:

1. Headline table (ECE, Brier, AUROC) before vs after, with Δ and a
   95% bootstrap CI on Δ Brier.
2. Per-domain breakdown (math / code / logic).
3. Per-difficulty breakdown (1..5, plus 6+ if SMC promoted the ceiling).
4. OOD slice (medical, legal).
5. Reliability diagram PNG (`comparison.png`).

### Success criteria

A pillar/run "ships" only if **all** the following are met. These are
the pass/fail gates for a publishable headline.

| Gate                                | Threshold                          |
| ----------------------------------- | ---------------------------------- |
| **Δ ECE (in-distribution)**         | ≤ -0.03 (lower is better)          |
| **Δ Brier (in-distribution)**       | ≤ -0.02, with 95% CI excluding 0   |
| **Δ ECE (OOD, both medical + legal)**| ≤ -0.02 (transfer requirement)    |
| **AUROC (in-distribution)**         | ≥ 0.65 (no discrimination collapse)|
| **Format rate**                     | ≥ 0.90 (parsing did not regress)   |
| **Abstain rate at d=5**             | > abstain rate at d=1              |

If a gate fails on the headline run but passes per-domain (e.g. math
improves but code regresses), report per-domain and investigate the
losing slice — usually a verifier sharpness or a curriculum balance
issue, not a fundamental training failure.

---

## 6 · MCP deployment

The trained adapter is a self-contained artifact: it can run in any
process that can load the base model + adapter. The MCP server is a
thin, **stateless** wrapper that exposes that artifact to MCP clients.

### 6a · Pre-flight (offline, no GPU)

```bash
make mcp-smoke         # offline self-test (no model load)
make mcp-health        # config preflight: are model + adapter + calibration present?
```

`make mcp-health` should print:

```
[ok] model_id                Qwen/Qwen2.5-3B-Instruct
[ok] adapter_path            ./honest-qwen-3b-grpo/final_adapters
[ok] calibration_info        eval/full_results.json
```

### 6b · Generate a Claude Desktop config

```bash
HONEST_MODEL_ID=Qwen/Qwen2.5-3B-Instruct \
HONEST_ADAPTER_PATH=$PWD/honest-qwen-3b-grpo/final_adapters \
HONEST_CALIBRATION_INFO=$PWD/eval/full_results.json \
make mcp-config
```

Paste the printed JSON snippet into Claude Desktop's
`~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows).

For Cursor and LangGraph, see `mcp_server/README.md`.

### 6c · Launch (manual)

```bash
./venv/bin/python -m mcp_server \
    --model-id          Qwen/Qwen2.5-3B-Instruct \
    --adapter-path      ./honest-qwen-3b-grpo/final_adapters \
    --calibration-info  eval/full_results.json
```

The server speaks MCP over stdio (the standard transport). Once
connected, the client can call:

```jsonc
// ask_with_calibrated_confidence
{
  "answer":           "42",
  "confidence":       0.83,
  "calibration_note": "Confidence is empirically calibrated: ECE = 0.04, Brier = 0.18.",
  "abstained":        false,
  "malformed":        false,
  "raw":              "<reasoning>...</reasoning><answer>42</answer><confidence>0.83</confidence>"
}

// get_calibration_info
{
  "available": true,
  "model":     "Qwen/Qwen2.5-3B-Instruct",
  "preset":    "qwen3b",
  "metrics":   { "ece": 0.04, "brier": 0.18, "auroc": 0.71 },
  "ood":       { "medical": {...}, "legal": {...} }
}
```

### 6d · One-shot installer

```bash
bin/install-mcp.sh
# - installs serving deps if missing
# - runs smoke + health
# - prints a paste-ready Claude Desktop config
```

---

## 7 · Self-learning verification

If you ran Step 3b (with self-learning flags), confirm each pillar
contributed and didn't degrade the run.

### 7a · Hindsight (HCR)

In `eval/full_results.json` look for the per-completion hindsight
emission rate. The model should emit `<hindsight>` tags on at least
**~30 %** of episodes by step 200, and the average `|hindsight − correctness|`
should be lower than `|confidence − correctness|`.

```bash
./venv/bin/python -c "
import json
r = json.load(open('eval/full_results.json'))
print('hindsight emission rate:', r['in_distribution']['metrics'].get('hindsight_rate'))
print('mean |c-y|:           ', r['in_distribution']['metrics'].get('mean_calibration_error'))
print('mean |hindsight-y|:   ', r['in_distribution']['metrics'].get('mean_hindsight_error'))
"
```

### 7b · Replay (CPR)

```bash
./venv/bin/python -c "
import json
s = json.load(open('honest-qwen-3b-grpo/replay_state.json'))
print('buffer_size:        ', s['size'])
print('priority entropy:   ', s['priority_entropy'])   # > 1.0 → diverse, healthy
print('top-1 priority share:', s['top1_share'])         # < 0.05 → no over-replay
"
```

### 7c · Self-mutating curriculum (SMC)

```bash
./venv/bin/python -c "
import json
s = json.load(open('honest-qwen-3b-grpo/smc_state.json'))
for d, st in s.items():
    print(f'{d}: max_unlocked = {st[\"max_unlocked_difficulty\"]}, '
          f'episodes_at_max = {st[\"episodes_at_max\"]}')
"
```

A pillar "shipped" if its dedicated metric moved in the right direction
**and** the headline ECE/Brier didn't regress vs the vanilla run.

---

## 8 · Notebook variant

For interactive Colab / Kaggle runs, `training/train_colab.ipynb`
mirrors Step 3 with cell-by-cell narration. The notebook respects all
the CLI flags, just edit the `args = "--model-preset qwen3b ..."` cell
at the top.

---

## 9 · Reproducibility

* Every random sampler is seeded from `--seed` (default 42) →
  difficulty controller, sampler shuffle, replay buffer init,
  generator stub.
* Eval is seeded by `eval/eval_seeds.json`; pin the seeds before any
  comparison run.
* The adapter directory contains both `trainer_state.json` and the
  full set of CLI args under `training_args.json`.
* Re-running the full pipeline from a fresh checkout against the same
  seeds reproduces the headline numbers within 95 % bootstrap CI.

---

## 10 · Troubleshooting

| Symptom                                              | First thing to check                                                |
| ---------------------------------------------------- | -------------------------------------------------------------------- |
| `Unified sampler is empty` at training startup       | Re-run **§1** ingestion scripts.                                     |
| `format_rate` < 0.70 in baseline                     | Run optional Stage-2 format SFT (`training/format_sft.py`) first.    |
| `train/reward_std` flat near 0                       | `RewardHealthCallback` already disabled the bad batch — keep going.  |
| `train/kl` blowing up (> 0.2)                        | `AdaptiveBetaCallback` will raise β. If still bad, lower `--learning-rate`. |
| OOM at step 0                                        | Drop `--num-generations` by 2 or `--max-completion-length` by 128.   |
| MCP client says "tool not found"                     | `make mcp-health` and confirm the adapter path is absolute.          |
| Δ ECE positive after training                        | Curriculum imbalance — pin `--domain-weights 0.5,0.35,0.15` and retry.|
| Replay buffer sampling the same prompt repeatedly    | `priority_entropy < log(2)` → CPR auto-disables for 50 steps.        |
| SMC ceiling oscillating                              | Increase `--smc-min-episodes-at-max` from 20 to 40.                  |

For server-specific MCP issues see the troubleshooting section in
`mcp_server/README.md`.
