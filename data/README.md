# Data Layer

This layer ingests external reasoning datasets and produces a unified JSONL
format shared across the project. Each ingestion script under `ingestion/`
downloads raw data into `data/raw/`, runs a domain-specific verifier from
`verifiers/` to confirm the reference solution is correct, and writes
normalized records into `data/processed/<domain>/`. Every record carries the
same envelope — `id`, `domain`, `prompt`, `answer`, `metadata` — so
downstream consumers do not need to special-case the upstream source.

The unified sampler in `sampler/unified_sampler.py` is the consumer: it
loads the processed shards and serves problems to the environment behind
the same interface the procedural generators in `server/generators/`
expose today. It supports domain mixing and difficulty weighting, and is
designed as a drop-in replacement so the environment, reward, and training
code do not need to change when swapping procedural problems for real
dataset problems.

Source datasets per domain:

- **Math** — Hendrycks MATH (`ingest_hendrycks_math.py`), verified with
  `verifiers/math_verifier.py` (SymPy-based equivalence).
- **Code** — MBPP (`ingest_mbpp.py`) and APPS (`ingest_apps.py`), verified
  with `verifiers/code_verifier.py` (sandboxed execution against tests).
- **Logic** — Regenerated ZebraLogic-style CSP puzzles
  (`regenerate_zebralogic.py`), verified with `verifiers/logic_verifier.py`
  (python-constraint / Z3).
