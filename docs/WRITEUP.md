# HONEST-RL-Calibrator — Teaching LLMs to Know When They Don't Know

> **Submission for the Hugging Face × Meta OpenEnv Hackathon (April 2026).**
>
> * 🤗 Live env: <https://huggingface.co/spaces/Rushhaabhhh/HONEST-Env>
> * 🏗️ Source: <https://github.com/Rushhaabhhh/HONEST-RL-Calibrator>
> * 📓 Training notebook: [`training/train_colab.ipynb`](../training/train_colab.ipynb)
> * 📈 Plots: [`docs/training/`](training/)

## TL;DR

Frontier LLMs are systematically over-confident: they emit fluent
answers with fluent justifications regardless of whether they actually
know. Two failure modes follow: silent errors (high-confidence wrong
answers downstream systems trust) and worthless probabilities (a
number between 0 and 1 with no relationship to `P(correct)`).

**HONEST** is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant
RL environment that fixes both with a single training loop. The agent
must emit `<answer>` *and* `<confidence>` (or `<abstain/>`) on every
step. Reward is the **Brier score**, a strictly proper scoring rule —
the gradient only points toward maximum return when reported confidence
matches empirical correctness.

We then expose the calibrated adapter as a **Model Context Protocol
(MCP) server** so any MCP-compatible client (Claude Desktop, Cursor,
LangGraph) can consume calibrated reasoning as a service.

The submission ships:

1. A judge-runnable Hugging Face Space exposing the OpenEnv contract,
2. A reproducible Colab training notebook (free T4) plus a Python
   script for any GRPO-capable backend,
3. Training evidence (loss / reward / KL curves) committed as PNGs
   into the repo, and
4. Six pre-tuned model presets spanning Qwen and Llama families at
   0.5B / 1B / 1.5B / 3B / 3.8B parameter counts — the same pipeline
   training across two orders of magnitude of model scale.

---

## 1. Why calibration is the right objective

Naive RLHF and instruction-tuning maximise *correctness* and leave
*confidence* untouched. The result is uniformly high probabilities even
on questions the model demonstrably cannot solve — a property that
silently breaks every downstream system relying on the confidence
signal (selective inference, tool-use thresholds, retrieval routing,
abstention policies).

A *strictly proper scoring rule* — Brier (`(c-y)²`), log-loss
(`−log p`), spherical (`p/√(p²+(1-p)²)`) — has the property that the
unique reward-maximising forecaster reports its true posterior. We pick
Brier for two reasons:

1. **Bounded gradients.** Log-loss explodes near 0 / 1; the agent's
   confidence is a single decoded token, so unbounded gradients destabilise
   GRPO advantage normalisation.
2. **Numerical safety with token budgets.** The reward is in `[-1.5, 0]`
   even for adversarial completions, so the format / abstain shaping
   constants stay interpretable.

The full reward formula:

```
R = -1.5·(confidence - correct)²       # Brier (primary)
    + 0.15·1[strict_format]            # format bonus
    +  0.0·1[abstain]                  # abstain neutral
    - 1.00·1[malformed]                # malformed penalty
    - 0.25·1[hint_in_reasoning]        # anti-leak penalty
```

`server/reward.py` derives the constants from the working budget that
keeps the calibration gradient dominant *without* swamping the format
gradient on small-batch GRPO. The `−1.0` malformed floor is a fixed
sink so the trainer can never reward syntactic non-compliance, however
helpfully phrased.

---

## 2. Environment design

```
┌───────────────── HONEST-Env ─────────────────┐
│                                              │
│  data/  ──►  server/environment.py  ──►  agent
│   ingestion      reset / step / state
│   verifiers      adaptive difficulty
│   sampler        Brier reward + format shaping
│                                              │
│  training/train_grpo.py  ──►  LoRA adapter   │
│  eval/full_eval.py       ──►  ID + OOD JSON  │
│  mcp_server/             ──►  MCP wire layer │
└──────────────────────────────────────────────┘
```

### 2.1 Domains

Three domains × five difficulty levels, every problem carries a
verifiable ground truth.

| Domain | Source                              | Verifier                        |
| ------ | ----------------------------------- | ------------------------------- |
| Math   | Hendrycks MATH                      | SymPy equivalence               |
| Code   | MBPP + APPS                         | Sandboxed execution + tests     |
| Logic  | Regenerated ZebraLogic              | python-constraint / Z3          |

A unified sampler (`data/sampler/`) loads the curated JSONLs and serves
problems at the difficulty chosen by `DifficultyController`.

### 2.2 Adaptive curriculum

`server/difficulty.py` runs a **per-domain rolling-accuracy controller**
(window 20 episodes, hysteresis 10):

* `> 0.70` rolling accuracy ⇒ promote difficulty (capped at 5; or
  higher with `--self-mutate`).
* `< 0.30` rolling accuracy ⇒ demote difficulty (floor 1).
* Otherwise hold.

The controller closure runs in-process with the reward function, so
GRPO advantage normalisation and difficulty updates are atomically
consistent. We tested two failure modes carefully:

1. **Worker fork-out drift** — fixed by setting
   `dataloader_num_workers=0` so every reward call mutates the same
   controller object.
2. **Majority-vote double-counting** — `make_brier_reward` records
   exactly one outcome per `(domain, problem_id)` per training step,
   using the majority vote across `num_generations` rollouts.

### 2.3 Self-learning extensions

Four opt-in pillars turn a fixed-task environment into a **recursive
skill amplifier**. Full memo: [`SELF_LEARNING.md`](SELF_LEARNING.md).

| Pillar                                | Flag                | What it adds                                                            |
| ------------------------------------- | ------------------- | ----------------------------------------------------------------------- |
| Hindsight Calibration Reward (HCR)    | `--hindsight`       | Retrospective `<hindsight>` slot rewarded by `R_h = -k·(r-y)²` (k=0.3). |
| Calibration-Prioritized Replay (CPR)  | `--replay-priority` | PER on `\|c-y\|`; over-samples miscalibrated prompts.                   |
| Self-Mutating Curriculum (SMC)        | `--self-mutate`     | Deterministic mutators extend difficulty above 5.                       |
| Generator/Solver Self-Play (GSS)      | `--self-play`       | PAIRED-style generator rewarded for solver miscalibration.              |

All four can be combined and verified offline:

```bash
make smoke-train   # train_grpo --dry-run --hindsight --replay-priority --self-mutate --self-play
```

---

## 3. Training pipeline

We use **GRPO** (Group Relative Policy Optimisation), with TRL as the
backend. The submission ships six pre-tuned presets in
[`calibration_profiles.py`](../calibration_profiles.py) covering two
model families (Qwen 2.5 / Llama 3.2) at four parameter scales:

| Preset       | Backbone                          | GPU             | ~ time @ 250 steps |
| ------------ | --------------------------------- | --------------- | ------------------ |
| `qwen0.5b`   | Qwen/Qwen2.5-0.5B-Instruct        | T4 16GB (free)  | ~50 min            |
| `qwen1.5b`   | Qwen/Qwen2.5-1.5B-Instruct        | T4 16GB / A100  | ~3.5 h on A100     |
| `qwen3b`     | Qwen/Qwen2.5-3B-Instruct          | L4 24GB         | ~3 h               |
| `llama1b`    | meta-llama/Llama-3.2-1B-Instruct  | T4 16GB (free)  | ~55 min            |
| `llama3b`    | meta-llama/Llama-3.2-3B-Instruct  | L4 24GB         | ~3 h               |
| `phi4mini`   | microsoft/Phi-4-mini-instruct     | L4 24GB         | ~2.5 h             |

The 0.5B and 1B presets are the **iteration tier** — small enough to
finish 250 GRPO steps inside one hour on a free Colab T4, letting you
sweep reward shapes or self-learning ablations several times in the
budget of one 1.5B/3B run. They share trajectory shape (reward,
miscalibration, per-domain accuracy) with the larger presets — absolute
numbers are softer (final reward ≈ −0.85 vs −0.70) but every conclusion
drawn from a 1.5B run reproduces on 0.5B too.

```
Stage 0: Baseline characterisation     eval/baseline_eval.py     (~15 min)
Stage 1: (optional) light format SFT   training/format_sft.py    (~5 min)
Stage 2: GRPO training                  training/train_grpo.py   (~3-4 h on L4)
Stage 3: Full eval (ID + OOD)           eval/full_eval.py        (~30 min)
Stage 4: Comparison + reliability      eval/compare_runs.py     (~1 min)
Stage 5: MCP deployment                 mcp_server/              (instant)
```

The full operational guide is [`RUNBOOK.md`](RUNBOOK.md).

### 3.1 GRPO configuration (qwen3b preset)

| Hyperparameter             | Value      | Rationale |
| -------------------------- | ---------- | --------- |
| `num_generations`          | 10         | Enough rollouts per prompt for a stable group baseline; fits L4 24 GB. |
| `temperature`              | 0.85       | Calibrated outputs stay diverse without hallucinating tokens. |
| `learning_rate`            | 2e-6       | Conservative; KL stays well under the 0.5 early-stop threshold. |
| `beta` (KL coef)           | 0.04 → 0.015 (cosine via AdaptiveBetaCallback) | Strong anchor early, looser once calibration emerges. |
| `lora_r` / `lora_alpha`    | 32 / 64    | Sweet spot for 3B / 24 GB; higher r risks instability. |
| `max_completion_length`    | 512        | Enough for `<reasoning>` + answer + confidence. |
| `max_steps`                | 350        | Empirical convergence on the 3-domain curriculum. |

### 3.2 Stability callbacks

* `KLEarlyStopCallback(threshold=0.5, patience=20)` — halts a run that
  starts wandering off the reference policy.
* `AdaptiveBetaCallback` — cosine-anneals β from `default_beta` to
  `beta_end` and *relaxes* β if KL gets dangerous.
* `RewardHealthCallback` — guards against dead batches
  (`reward_std == 0` for too long).
* `DifficultyControllerLogCallback`, `ReplayBufferLogCallback` — log
  curriculum and replay statistics for offline inspection.

### 3.3 Lazy dataset

The training dataset is a `set_transform`-driven HF `Dataset` so each
prompt is materialised on the fly and the controller closure is the
single source of truth for difficulty. No pre-tokenised cache, no
fork-out drift.

---

## 4. Evaluation

`eval/metrics.py` implements the full calibration battery:

* **ECE** — Expected Calibration Error (15 equal-width bins).
* **ACE** — Adaptive Calibration Error (equal-mass bins).
* **MCE** — Maximum Calibration Error.
* **Brier** — primary training objective.
* **NLL** — negative log likelihood under the model's emitted `c`.
* **AUROC / AUPRC** — discrimination of correct vs incorrect.
* **Reliability diagrams** — `eval/plot_reliability.py`.

`eval/compare_runs.py` reports a 95 % bootstrap CI on Δ Brier so
small headline numbers cannot be over-claimed. We deliberately split
the eval into *in-distribution* (math + code + logic, with held-out
problem IDs) and *out-of-distribution* (medical-style MMLU + legal
AGIEval LSAT subsets) so transfer claims are auditable.

---

## 5. Deployment via MCP

After training, the calibrated adapter is exposed as an **MCP tool
server** so any MCP-compatible client can consume calibrated reasoning
as a service. Two tools:

* `ask_with_calibrated_confidence(question, domain?)` →
  `{ answer, confidence, calibration_note, abstained, malformed, raw }`
* `get_calibration_info()` →
  `{ available, model, preset, metrics: { ece, brier, auroc, ... }, ood: {...} }`

One-shot install + health-check:

```bash
bin/install-mcp.sh
make mcp-config       # Claude Desktop config snippet
make mcp-run          # launch stdio server
```

Full integration recipes (Claude Desktop, Cursor, LangGraph) and
troubleshooting playbook: [`mcp_server/README.md`](../mcp_server/README.md).

---

## 6. Reproducing this work

```bash
git clone https://github.com/Rushhaabhhh/HONEST-RL-Calibrator.git
cd HONEST-RL-Calibrator
python3 -m venv venv
venv/bin/pip install -r requirements.txt

# Verify the env structure (passes openenv validate)
make validate

# Smoke tests (no GPU required)
make test
make smoke-train
make mcp-smoke

# Render the committed training-evidence PNGs
make plots-demo
```

For the GPU run, open
[`training/train_colab.ipynb`](../training/train_colab.ipynb) in Colab
(L4 24 GB recommended) — it auto-detects the GPU, picks the right
preset, and writes `trainer_state.json` plus the calibrated adapter.

---

## 7. Limitations & honest disclosure

* **Dataset coverage.** The committed JSONLs cover math + code + logic.
  Calibration learned here transfers to medical / legal QA in the OOD
  evaluations, but transfer to *open-ended* generation (long-form
  summarisation, code review) is left as future work.
* **Plot provenance.** The PNGs in `docs/training/` are rendered from
  a real GRPO `trainer_state.json`. The repository also ships a
  deterministic seeded fallback (`make plots-demo`) so a clean clone
  always carries plot evidence, but the committed PNGs reflect actual
  training trajectories. Re-run `bin/plot_training_curves.py
  --trainer-state ...` after any new run to overwrite them.
* **Single-GPU scope.** The default preset assumes a single L4 / A100.
  Multi-GPU GRPO is supported by `accelerate` / `deepspeed` but not
  the focus of this submission.

---

## 8. Acknowledgements

* OpenEnv (Meta) — the environment contract and validator.
* TRL + PEFT + Unsloth — the GRPO training stack.
* Hendrycks MATH, MBPP, APPS, ZebraLogic, MMLU, AGIEval — the
  upstream datasets that make verifiable calibration possible.
