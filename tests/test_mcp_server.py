"""Offline tests for the HONEST MCP wrapper.

These tests do NOT load a real model; they exercise the wrapper logic
(parser routing, calibration metadata loader, server graph construction).
That makes them safe to run in CI / on a tiny laptop without GPU access.

The full end-to-end stdio test belongs in an integration suite — out of
scope for this hackathon.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp_server.honest_mcp import (  # noqa: E402
    CalibratedModel,
    _build_calibration_note,
    load_calibration_info,
    _smoke_test,
    _MCP_AVAILABLE,
)


# ---------------------------------------------------------------------------
# load_calibration_info
# ---------------------------------------------------------------------------


class TestLoadCalibrationInfo:
    def test_no_path_returns_unavailable(self):
        info = load_calibration_info(None)
        assert info["available"] is False
        assert "no calibration file" in info["reason"]

    def test_missing_file_returns_unavailable(self, tmp_path):
        info = load_calibration_info(str(tmp_path / "does_not_exist.json"))
        assert info["available"] is False
        assert "file not found" in info["reason"]

    def test_invalid_json_returns_unavailable(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{{{not json")
        info = load_calibration_info(str(bad))
        assert info["available"] is False
        assert "failed to parse" in info["reason"]

    def test_valid_json_lifts_overall_block(self, tmp_path):
        valid = tmp_path / "ok.json"
        valid.write_text(json.dumps({
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "preset": "qwen7b",
            "reasoning_mode": "required",
            "overall": {
                "n_samples": 600,
                "ece": 0.041,
                "brier": 0.067,
                "auroc": 0.832,
                "accuracy": 0.612,
                "format_rate": 0.991,
            },
            "ood": {
                "medical": {
                    "n_samples": 200, "ece": 0.072, "brier": 0.183,
                    "auroc": 0.781, "accuracy": 0.524,
                },
            },
        }))
        info = load_calibration_info(str(valid))
        assert info["available"] is True
        assert info["model"] == "Qwen/Qwen2.5-7B-Instruct"
        assert info["preset"] == "qwen7b"
        assert info["metrics"]["ece"] == pytest.approx(0.041)
        assert info["metrics"]["brier"] == pytest.approx(0.067)
        assert "medical" in info["ood"]
        assert info["ood"]["medical"]["n"] == 200


# ---------------------------------------------------------------------------
# CalibratedModel construction (no .load())
# ---------------------------------------------------------------------------


class TestCalibratedModelLazy:
    def test_init_does_not_load_model(self):
        m = CalibratedModel(
            model_id="dummy/model",
            adapter_path=None,
            reasoning_mode="required",
        )
        assert m.model is None
        assert m.tokenizer is None
        assert m.system_prompt          # non-empty string
        assert "{question}" in m.user_template

    def test_reasoning_modes(self):
        from calibration_profiles import REASONING_MODES
        for mode in REASONING_MODES:
            m = CalibratedModel(
                model_id="dummy", adapter_path=None, reasoning_mode=mode,
            )
            assert isinstance(m.system_prompt, str)
            assert m.system_prompt.strip()


# ---------------------------------------------------------------------------
# Calibration note
# ---------------------------------------------------------------------------


class TestBuildCalibrationNote:
    def test_unavailable_falls_back(self):
        note = _build_calibration_note({"available": False, "reason": "x"})
        assert "HONEST RL environment" in note
        assert "metrics not yet" in note.lower()

    def test_available_includes_metrics(self):
        note = _build_calibration_note({
            "available": True,
            "metrics": {
                "ece": 0.041, "brier": 0.067, "auroc": 0.832,
                "accuracy": 0.612, "n_samples": 600,
            },
        })
        assert "ECE=0.0410" in note
        assert "Brier=0.0670" in note
        assert "AUROC=0.8320" in note


# ---------------------------------------------------------------------------
# Smoke test entry point
# ---------------------------------------------------------------------------


class TestSmokeTest:
    def test_smoke_test_runs_clean(self, capsys):
        # Use a deliberately-missing file so calibration_info reports
        # available=False — the smoke test still passes (it asserts the
        # *contract* is satisfied, not that the file is present).
        info = load_calibration_info("does_not_exist.json")
        rc = _smoke_test(info)
        captured = capsys.readouterr()
        assert "smoke-test" in captured.out
        assert rc == 0


# ---------------------------------------------------------------------------
# build_server (only when mcp installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _MCP_AVAILABLE, reason="mcp package not installed")
class TestBuildServer:
    def test_server_exposes_two_tools(self):
        from mcp_server.honest_mcp import build_server
        m = CalibratedModel(model_id="dummy", adapter_path=None)
        server = build_server(m, {"available": False})
        # The MCP server stashes registered handlers on .request_handlers.
        from mcp.types import ListToolsRequest
        assert ListToolsRequest in server.request_handlers
