# HONEST · convenience Makefile
#
# Common workflows. Override PYTHON to use a non-default interpreter:
#   make test PYTHON=./venv/bin/python
#
# We default to `python3` (universally present on modern Linux/macOS).
# Fall back to `python` if `python3` is missing (rare; Windows mostly).

PYTHON ?= $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python3)
PYTEST ?= $(PYTHON) -m pytest

.PHONY: help test test-fast lint smoke-train plots plots-demo validate mcp-smoke mcp-health mcp-config mcp-install mcp-run

help:
	@echo "HONEST Makefile targets"
	@echo ""
	@echo "  test           Full pytest suite (tests/ + data/tests/)"
	@echo "  test-fast      Just the unit tests (tests/)"
	@echo "  smoke-train    Dry-run train_grpo with all four self-learning pillars"
	@echo "  validate       Run 'openenv validate' against the project root"
	@echo ""
	@echo "  plots-demo     Regenerate docs/training/*.png from the demo trace"
	@echo "  plots          Render docs/training/*.png from a real run"
	@echo "                 (set TRAINER_STATE=path/to/trainer_state.json)"
	@echo ""
	@echo "  mcp-smoke      Offline MCP self-test (no model load)"
	@echo "  mcp-health     MCP config preflight"
	@echo "  mcp-config     Print a ready-to-paste Claude Desktop config"
	@echo "  mcp-install    Install MCP serving deps + run smoke + health"
	@echo "  mcp-run        Launch the MCP stdio server (uses HONEST_* env vars)"
	@echo ""
	@echo "End-to-end pipeline runbook: docs/RUNBOOK.md"
	@echo "Self-learning research memo:  docs/SELF_LEARNING.md"

test:
	$(PYTEST) tests/ data/tests/

test-fast:
	$(PYTEST) tests/

smoke-train:
	$(PYTHON) training/train_grpo.py --dry-run --hindsight --replay-priority --self-mutate --self-play

validate:
	$(PYTHON) -m openenv.cli validate --verbose

# Regenerate the committed demo plots (deterministic; safe to run anytime).
plots-demo:
	$(PYTHON) bin/plot_training_curves.py --demo --out docs/training \
	    --label "qwen3b · 350 steps · L4 (demo)"

# Render plots from a real trainer_state.json. Override TRAINER_STATE to
# point at a different run directory.
TRAINER_STATE ?= ./honest-qwen3b-grpo/trainer_state.json
plots:
	$(PYTHON) bin/plot_training_curves.py --trainer-state $(TRAINER_STATE) \
	    --out docs/training

mcp-smoke:
	$(PYTHON) -m mcp_server --smoke-test

mcp-health:
	$(PYTHON) -m mcp_server --health

mcp-config:
	@PYTHON=$(PYTHON) bin/install-mcp.sh --print-claude-config

mcp-install:
	PYTHON=$(PYTHON) bin/install-mcp.sh

mcp-run:
	$(PYTHON) -m mcp_server
