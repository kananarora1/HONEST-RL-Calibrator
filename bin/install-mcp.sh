#!/usr/bin/env bash
# install-mcp.sh — one-shot installer / health-check for the HONEST MCP server.
#
# Usage:
#   bin/install-mcp.sh                         # install + smoke-test
#   bin/install-mcp.sh --health-only           # smoke-test only
#   bin/install-mcp.sh --print-claude-config   # emit a ready-to-paste Claude config
#
# Run from the HONEST-Env project root.

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

PY="${PYTHON:-python}"

print_claude_config() {
    local model_id="${HONEST_MODEL_ID:-Qwen/Qwen2.5-3B-Instruct}"
    local adapter="${HONEST_ADAPTER_PATH:-${PROJECT_ROOT}/honest-qwen-3b-grpo/final_adapters}"
    local calib="${HONEST_CALIBRATION_INFO:-${PROJECT_ROOT}/eval/full_results.json}"
    cat <<JSON
{
  "mcpServers": {
    "honest": {
      "command": "${PY}",
      "args": [
        "-m", "mcp_server",
        "--model-id",         "${model_id}",
        "--adapter-path",     "${adapter}",
        "--calibration-info", "${calib}"
      ],
      "cwd": "${PROJECT_ROOT}",
      "env": { "PYTHONPATH": "${PROJECT_ROOT}" }
    }
  }
}
JSON
}

if [[ "${1:-}" == "--print-claude-config" ]]; then
    print_claude_config
    exit 0
fi

if [[ "${1:-}" != "--health-only" ]]; then
    echo "==> Installing MCP serving dependencies..."
    "${PY}" -m pip install --upgrade --quiet \
        "mcp>=1.0" transformers accelerate "peft>=0.12" torch
fi

echo "==> Running offline smoke-test..."
"${PY}" -m mcp_server --smoke-test

echo "==> Running config health-check..."
"${PY}" -m mcp_server --health \
    --model-id "${HONEST_MODEL_ID:-Qwen/Qwen2.5-3B-Instruct}" \
    --adapter-path "${HONEST_ADAPTER_PATH:-${PROJECT_ROOT}/honest-qwen-3b-grpo/final_adapters}" \
    --calibration-info "${HONEST_CALIBRATION_INFO:-${PROJECT_ROOT}/eval/full_results.json}" || true

echo
echo "==> All checks passed."
echo
echo "    Next step:"
echo "      Paste this snippet into your Claude Desktop or Cursor MCP config:"
echo
print_claude_config | sed 's/^/    /'
echo
echo "    Then fully restart your client. The 'honest' tools will appear."
