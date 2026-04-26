"""FastAPI server entry-point for the HONEST environment.

`create_fastapi_app` from openenv-core auto-registers the standard runtime
contract endpoints on the returned app:

  * ``POST /reset``      — start a new episode
  * ``POST /step``       — submit an action, receive observation + reward
  * ``GET  /state``      — read current internal state (simulation mode)
  * ``GET  /schema``     — combined action / observation / state JSON schemas
  * ``GET  /metadata``   — environment name, description, version, README
  * ``GET  /health``     — ``{"status": "healthy"}`` for load-balancers
  * ``POST /mcp``        — JSON-RPC 2.0 MCP entry point
  * ``WS   /ws`` , ``WS /mcp`` — persistent session websockets
  * ``GET  /openapi.json`` — OpenAPI 3 schema (FastAPI default)
  * ``GET  /docs`` , ``/redoc`` — interactive API docs

Custom metadata (name, description, version) is provided by overriding
:py:meth:`HonestEnvironment.get_metadata` — this keeps the FastAPI layer
thin and lets every deployment mode (Docker, ``uv run server``, direct
``uvicorn``) report consistent information.
"""

import os

from openenv.core.env_server.http_server import create_fastapi_app

from models.models import HonestAction, HonestObservation
from server.environment import HonestEnvironment

# ---------------------------------------------------------------------------
# Build the OpenEnv FastAPI app
# create_fastapi_app expects a factory callable, not an instance — a fresh
# environment is created per session so concurrent /reset calls don't share
# state.
# ---------------------------------------------------------------------------

app = create_fastapi_app(
    env=HonestEnvironment,
    action_cls=HonestAction,
    observation_cls=HonestObservation,
    max_concurrent_envs=32,
)


def main() -> None:
    """Console entry-point used by ``[project.scripts]`` and direct execution.

    Enables every deployment mode the OpenEnv toolchain expects::

        uv run --project . server          # via [project.scripts]
        python -m server.app               # via __main__
        uvicorn server.app:app             # direct, suitable for production

    Bind address and port are read from the ``HOST`` and ``PORT`` env vars
    (with sensible defaults) so this entry-point can be invoked with no
    arguments — the form ``[project.scripts]`` console-scripts use.
    """
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
