# Hugging Face Spaces / OpenEnv container for HONEST-Env.
#
# Layout:
#   * Slim Python 3.11 base
#   * Non-root user (uid 1000) — required by HF Spaces
#   * pip install --no-cache-dir from requirements.txt (CPU-only, server side)
#   * Copy the OpenEnv server stack (server/ + models/ + data/ + client/)
#   * uvicorn at port 8000 (matches openenv.yaml)
#
# Notes
# -----
# - The training stack (torch, trl, peft, unsloth) lives in
#   ``[project.optional-dependencies].training`` and is NOT installed here.
#   The container is the *serving* image — small, deterministic, GPU-free.
# - ``data/processed/*.jsonl`` is committed to the repo so the unified
#   sampler can boot without any ingestion step at runtime.

FROM python:3.11-slim

# Set up a non-root user (required by HF Spaces; also good hygiene).
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR $HOME/app

# Install dependencies first for layer-cache friendliness.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Source code — only the runtime/server pieces are needed for the OpenEnv
# container. The training/ and eval/ packages are pulled in on the GPU box.
COPY --chown=user models/                models/
COPY --chown=user server/                server/
COPY --chown=user client/                client/
COPY --chown=user data/                  data/
COPY --chown=user calibration_profiles.py .
COPY --chown=user openenv.yaml           .
COPY --chown=user pyproject.toml         .
COPY --chown=user README.md              .

EXPOSE 8000

# Health-check hits OpenEnv's auto-mounted /health endpoint; failure marks
# the container unhealthy so HF Spaces / orchestrators can redeploy it.
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; r=urllib.request.urlopen('http://localhost:8000/health',timeout=4); sys.exit(0 if r.status==200 else 1)" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
