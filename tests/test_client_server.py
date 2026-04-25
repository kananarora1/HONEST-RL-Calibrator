"""Integration tests: client ↔ server over a real WebSocket connection.

The test suite spins up the FastAPI server in a subprocess for the duration
of the session, runs all tests, then tears it down.

Run manually:
    ./venv/bin/pytest tests/test_client_server.py -v
Or with a pre-running server:
    ./run_server.sh &
    sleep 3
    ./venv/bin/pytest tests/test_client_server.py -v
"""

import asyncio
import subprocess
import sys
import time
from typing import AsyncGenerator, Generator

import httpx
import pytest
import pytest_asyncio  # type: ignore[import]

from client.client import HonestEnv
from models.models import HonestAction

BASE_URL = "http://localhost:18765"   # use a non-default port to avoid clashes
WS_URL = "ws://localhost:18765"

# ---------------------------------------------------------------------------
# Server fixture — starts once for the whole test session
# ---------------------------------------------------------------------------


def _wait_for_server(url: str, timeout: float = 15.0) -> bool:
    """Poll /health until the server responds or timeout is exceeded."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{url}/health", timeout=2.0)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


@pytest.fixture(scope="session")
def server_process() -> Generator[subprocess.Popen, None, None]:
    """Start the uvicorn server once for the full test session."""
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "server.app:app",
            "--host", "127.0.0.1",
            "--port", "18765",
            "--log-level", "warning",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    ready = _wait_for_server(BASE_URL)
    if not ready:
        proc.terminate()
        stderr_out = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        pytest.fail(f"Server did not start in time.\nStderr:\n{stderr_out}")

    yield proc

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---------------------------------------------------------------------------
# Per-test async client fixture
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture()
async def client(server_process) -> AsyncGenerator[HonestEnv, None]:
    """Fresh connected client per test."""
    async with HonestEnv(base_url=BASE_URL) as env:
        yield env


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

WELL_FORMED = "<reasoning>think</reasoning><answer>42</answer><confidence>0.5</confidence>"
MALFORMED = "no tags at all"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reset_returns_observation(client: HonestEnv):
    result = await client.reset()
    obs = result.observation
    assert obs is not None
    assert isinstance(obs.question, str) and obs.question
    assert obs.domain in {"math", "code", "logic"}
    assert 1 <= obs.difficulty <= 5
    assert obs.episode_step == 0
    assert obs.terminal is False


@pytest.mark.asyncio
async def test_reset_done_is_false(client: HonestEnv):
    result = await client.reset()
    assert result.done is False


@pytest.mark.asyncio
async def test_step_with_well_formed_action(client: HonestEnv):
    await client.reset()
    action = HonestAction(raw_text=WELL_FORMED)
    result = await client.step(action)
    obs = result.observation
    assert result.reward is not None
    assert -1.0 <= result.reward <= 0.1
    # non-terminal step should still provide a question
    assert obs is not None


@pytest.mark.asyncio
async def test_step_with_malformed_action_gives_minus_half(client: HonestEnv):
    await client.reset()
    action = HonestAction(raw_text=MALFORMED)
    result = await client.step(action)
    from server.reward import MALFORMED_PENALTY
    assert result.reward == pytest.approx(MALFORMED_PENALTY)


@pytest.mark.asyncio
async def test_episode_terminates_after_five_steps(client: HonestEnv):
    await client.reset()
    for i in range(5):
        action = HonestAction(raw_text=WELL_FORMED)
        result = await client.step(action)
        if i < 4:
            assert not result.done, f"Ended too early at step {i+1}"
    assert result.done is True
    assert result.observation.terminal is True


@pytest.mark.asyncio
async def test_query_convenience_method(client: HonestEnv):
    info = await client.query()
    assert set(info.keys()) == {"question", "domain", "difficulty"}
    assert info["domain"] in {"math", "code", "logic"}
    assert 1 <= info["difficulty"] <= 5
    assert isinstance(info["question"], str) and info["question"]


@pytest.mark.asyncio
async def test_multiple_resets_are_independent(client: HonestEnv):
    r1 = await client.reset()
    r2 = await client.reset()
    # After second reset episode_step is 0 again
    assert r2.observation.episode_step == 0
    # Two unseeded resets may or may not return the same domain — just check structure
    assert r1.observation.domain in {"math", "code", "logic"}
    assert r2.observation.domain in {"math", "code", "logic"}


@pytest.mark.asyncio
async def test_import_from_package():
    """Sanity: ensure `from client.client import HonestEnv` works."""
    from client.client import HonestEnv as HE   # noqa: F401
    assert HE is HonestEnv
