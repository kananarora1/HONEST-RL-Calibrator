#!/usr/bin/env bash
# run_server.sh — start the HONEST-Env FastAPI server locally for development/testing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/venv"

if [[ ! -d "$VENV" ]]; then
    echo "ERROR: virtualenv not found at $VENV. Run: python3 -m venv venv && venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"
LOG_LEVEL="${LOG_LEVEL:-info}"

echo "Starting HONEST-Env server on http://${HOST}:${PORT} ..."
echo "  Docs:     http://localhost:${PORT}/docs"
echo "  Health:   http://localhost:${PORT}/health"
echo "  Metadata: http://localhost:${PORT}/metadata"
echo "  Schema:   http://localhost:${PORT}/schema"
echo ""

cd "$SCRIPT_DIR"
exec "$VENV/bin/uvicorn" server.app:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level "$LOG_LEVEL" \
    --reload
