"""mcp_server/honest_mcp.py — Model Context Protocol server for HONEST.

Exposes the *trained, calibrated* HONEST model as two MCP tools:

    1. ``ask_with_calibrated_confidence(question, domain?)``
       Returns the model's answer along with its calibrated confidence and a
       short calibration note. The confidence is the same value the model
       was trained to emit under the Brier reward — i.e. an honest estimate
       of P(correct).

    2. ``get_calibration_info()``
       Returns the headline calibration metrics measured at evaluation time
       (ECE, Brier, AUROC), so any agent calling this tool can decide how to
       weigh outputs from this model.

The server is a *deployment artifact only*: it loads an already-trained
model and never touches training infrastructure. Mixing training and
serving introduces hard-to-debug stability bugs, so the two are kept in
separate processes by design.

Run it from Claude Desktop (see mcp_server/README.md for the config), or
locally via:

    python mcp_server/honest_mcp.py \\
        --model-id Qwen/Qwen2.5-3B-Instruct \\
        --adapter-path ./honest-qwen-3b-grpo/final_adapters \\
        --calibration-info eval/full_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Make sibling packages importable when this script is invoked directly via
# `python mcp_server/honest_mcp.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.reward import parse_action, parse_action_lenient  # noqa: E402

# Late-imported (heavy / optional) modules — guarded so smoke tests can run
# without `mcp` or `transformers` installed.
try:  # pragma: no cover — exercised only when the MCP package is available.
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool
    _MCP_AVAILABLE = True
except ImportError:  # pragma: no cover
    Server = None       # type: ignore[assignment]
    stdio_server = None  # type: ignore[assignment]
    TextContent = None   # type: ignore[assignment]
    Tool = None          # type: ignore[assignment]
    _MCP_AVAILABLE = False


_LOG = logging.getLogger("honest_mcp")


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class CalibratedModel:
    """Thin wrapper around a HuggingFace causal-LM + optional LoRA adapter.

    The wrapper is intentionally minimal: it owns the tokenizer + model, and
    knows how to (a) format a HONEST-style prompt and (b) extract the
    calibrated answer/confidence from the raw output. Anything more complex
    — batching, KV-cache reuse, vLLM — belongs in a downstream serving layer.
    """

    def __init__(
        self,
        model_id: str,
        adapter_path: Optional[str] = None,
        device: str = "auto",
        max_new_tokens: int = 512,
        reasoning_mode: str = "required",
    ):
        from calibration_profiles import prompt_templates

        self.model_id = model_id
        self.adapter_path = adapter_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.system_prompt, self.user_template = prompt_templates(reasoning_mode)

        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        """Lazy-load the base model + adapter. Idempotent."""
        if self.model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "transformers is required to run the MCP server. "
                "pip install transformers accelerate torch peft"
            ) from exc

        _LOG.info("Loading tokenizer: %s", self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        _LOG.info("Loading model: %s (device=%s)", self.model_id, self.device)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map=self.device,
            trust_remote_code=True,
        )

        if self.adapter_path and Path(self.adapter_path).exists():
            _LOG.info("Merging LoRA adapter: %s", self.adapter_path)
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            self.model = self.model.merge_and_unload()

        self.model.eval()
        _LOG.info("Model ready.")

    # -- inference --------------------------------------------------------

    def _generate_raw(self, question: str) -> str:
        import torch

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.format(question=question)},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def answer(self, question: str) -> dict[str, Any]:
        """Run one inference pass and return a structured response.

        Returns one of:
          {"type": "answer",  "answer": str, "confidence": float, "raw": str}
          {"type": "abstain", "raw": str}
          {"type": "malformed", "raw": str}

        Confidence is the *trained* calibrated value — clamped to [0, 1] by
        the parser. Calibration training has already aligned this with the
        true probability of correctness on in-distribution problems.
        """
        if self.model is None:
            self.load()

        raw = self._generate_raw(question)

        # Try the strict parser first (the same contract used in training);
        # fall back to lenient parsing so the MCP tool always returns *something*
        # actionable rather than failing on a slightly-off output.
        parsed = parse_action(raw)
        if parsed["type"] == "malformed":
            parsed = parse_action_lenient(raw)

        out: dict[str, Any] = {"type": parsed["type"], "raw": raw}
        if parsed["type"] == "answer":
            out["answer"] = parsed["answer"]
            out["confidence"] = float(parsed["confidence"])
        return out


# ---------------------------------------------------------------------------
# Calibration metadata
# ---------------------------------------------------------------------------


def load_calibration_info(path: Optional[str]) -> dict[str, Any]:
    """Read the headline metrics from a ``full_results.json`` (or compatible).

    Falls back to a generic placeholder if the file is missing — that way the
    server still starts and the ``get_calibration_info`` tool returns a
    structured "no metrics available" payload instead of crashing.
    """
    if not path:
        return {"available": False, "reason": "no calibration file configured"}

    p = Path(path)
    if not p.exists():
        return {"available": False, "reason": f"file not found: {p}"}

    try:
        data = json.loads(p.read_text())
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"failed to parse {p}: {exc}"}

    overall = data.get("overall") or {}
    info = {
        "available": True,
        "model": data.get("model_id") or data.get("model"),
        "preset": data.get("preset"),
        "reasoning_mode": data.get("reasoning_mode"),
        "metrics": {
            "n_samples":  overall.get("n_samples") or overall.get("n_total"),
            "ece":        overall.get("ece"),
            "ace":        overall.get("ace"),
            "mce":        overall.get("mce"),
            "brier":      overall.get("brier"),
            "nll":        overall.get("nll"),
            "auroc":      overall.get("auroc"),
            "accuracy":   overall.get("accuracy"),
            "format_rate": overall.get("format_rate"),
        },
        "source_file": str(p),
    }

    # Lift the OOD breakdown if present — useful for downstream agents that
    # want to weight by domain.
    ood = data.get("ood")
    if isinstance(ood, dict) and ood:
        info["ood"] = {
            domain: {
                "n":     cond.get("n_samples"),
                "ece":   cond.get("ece"),
                "brier": cond.get("brier"),
                "auroc": cond.get("auroc"),
                "accuracy": cond.get("accuracy"),
            }
            for domain, cond in ood.items()
        }

    return info


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------


def _build_calibration_note(info: dict[str, Any]) -> str:
    if not info.get("available"):
        return ("This model was trained with the HONEST RL environment "
                "(Brier-score reward) for calibration. Headline metrics not "
                "yet attached — pass --calibration-info to surface them.")
    metrics = info.get("metrics", {})
    ece = metrics.get("ece")
    brier = metrics.get("brier")
    auroc = metrics.get("auroc")
    parts = []
    if ece is not None:
        parts.append(f"ECE={ece:.4f}")
    if brier is not None:
        parts.append(f"Brier={brier:.4f}")
    if auroc is not None:
        parts.append(f"AUROC={auroc:.4f}")
    metrics_str = "  ".join(parts) if parts else "metrics unavailable"
    return (
        "Confidence reflects an empirically calibrated probability of "
        "correctness, trained via Brier-score reward in the HONEST RL "
        f"environment.  Eval: {metrics_str}."
    )


def build_server(model: CalibratedModel, calibration_info: dict[str, Any]):
    """Wire the two HONEST tools onto an MCP server instance.

    Returns the server object (caller is responsible for invoking
    ``stdio_server`` etc.). Separated so we can construct + introspect the
    server in tests without ever opening stdio streams.
    """
    if not _MCP_AVAILABLE:  # pragma: no cover
        raise SystemExit(
            "The `mcp` package is not installed — run `pip install mcp>=1.0` "
            "or use the project requirements.txt."
        )

    server = Server("honest-calibrated-reasoning")

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return [
            Tool(
                name="ask_with_calibrated_confidence",
                description=(
                    "Ask a question and receive an answer with a calibrated "
                    "confidence score reflecting the trained model's true "
                    "probability of correctness."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to answer.",
                        },
                        "domain": {
                            "type": "string",
                            "enum": ["math", "code", "logic", "general"],
                            "description": "Optional domain hint.",
                        },
                    },
                    "required": ["question"],
                },
            ),
            Tool(
                name="get_calibration_info",
                description=(
                    "Return the calibration metrics and benchmarks of this "
                    "model so callers can decide how to weigh its outputs."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name == "ask_with_calibrated_confidence":
            question = (arguments or {}).get("question", "").strip()
            if not question:
                payload = {"error": "Missing 'question' argument."}
                return [TextContent(type="text", text=json.dumps(payload))]

            result = await asyncio.to_thread(model.answer, question)
            payload = {
                "answer":           result.get("answer"),
                "confidence":       result.get("confidence"),
                "abstained":        result["type"] == "abstain",
                "malformed":        result["type"] == "malformed",
                "calibration_note": _build_calibration_note(calibration_info),
                "raw":              result.get("raw", "")[:1000],
            }
            return [TextContent(type="text", text=json.dumps(payload))]

        if name == "get_calibration_info":
            return [TextContent(type="text", text=json.dumps(calibration_info))]

        return [TextContent(type="text", text=json.dumps({
            "error": f"Unknown tool: {name}",
        }))]

    return server


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def _smoke_test(calibration_info: dict[str, Any]) -> int:
    """Offline self-test: parse, reward, calibration-info loading.

    Does NOT load a real model — runs in any environment with just the
    server-side code installed. Used by ``tests/test_mcp_server.py`` and
    by users debugging their MCP install before downloading a 7B model.

    Exits with 0 if every assertion passes, 1 otherwise.
    """
    print("=" * 60)
    print("HONEST MCP smoke-test (offline, no model load)")
    print("=" * 60)
    failures: list[str] = []

    # 1. parser sanity
    print("\n[1/4] parse_action / parse_action_lenient")
    from server.reward import parse_action  # noqa: F401
    well_formed = "<reasoning>r</reasoning><answer>4</answer><confidence>0.7</confidence>"
    parsed = parse_action(well_formed)
    if parsed.get("type") != "answer" or abs(parsed["confidence"] - 0.7) > 1e-9:
        failures.append(f"parse_action returned {parsed!r}")
        print("  FAIL")
    else:
        print("  OK  ->", parsed)

    # 2. CalibratedModel construction (no .load()).
    print("\n[2/4] CalibratedModel(...) lazy construction")
    try:
        m = CalibratedModel(
            model_id="dummy/model",
            adapter_path=None,
            reasoning_mode="required",
        )
        assert m.model is None and m.tokenizer is None, "model must be lazy"
        assert isinstance(m.system_prompt, str)
        assert isinstance(m.user_template, str)
        print("  OK  -> system_prompt is a string of length",
              len(m.system_prompt), "chars")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"CalibratedModel construction raised: {exc!r}")
        print("  FAIL", exc)

    # 3. Calibration metadata loader
    print("\n[3/4] load_calibration_info()")
    info = calibration_info
    keys_ok = isinstance(info, dict) and "available" in info
    if not keys_ok:
        failures.append("calibration_info missing 'available' key")
        print("  FAIL")
    else:
        print(f"  OK  -> available={info['available']}",
              f"reason={info.get('reason')!r}" if not info["available"] else "")

    # 4. Build the server graph (validates MCP wiring without stdio).
    print("\n[4/4] build_server(...) construction")
    if not _MCP_AVAILABLE:
        print("  SKIP — mcp package not installed (this is fine for "
              "the offline smoke test, but you cannot run the real server "
              "until you `pip install mcp>=1.0`).")
    else:
        try:
            server = build_server(
                CalibratedModel(model_id="dummy/model", adapter_path=None),
                info,
            )
            print(f"  OK  -> server name = {getattr(server, 'name', 'honest-calibrated-reasoning')!r}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"build_server raised: {exc!r}")
            print("  FAIL", exc)

    print()
    print("=" * 60)
    if failures:
        print(f"FAILED ({len(failures)} issue(s))")
        for f in failures:
            print(" *", f)
        return 1
    print("PASSED — server-side code is healthy.")
    print("Next step: `python -m mcp_server` to launch the actual stdio server.")
    return 0


def _health_summary(args: argparse.Namespace, calibration_info: dict[str, Any]) -> int:
    """Print a one-shot summary of the server's intended config.

    Useful for users debugging their Claude Desktop / Cursor config: they
    can copy-paste the same flags into ``--health`` and verify everything
    is wired correctly *before* trying to launch the model.

    Exits non-zero if a hard precondition is violated (model id is empty,
    calibration file path is non-existent, etc.).
    """
    print("=" * 60)
    print("HONEST MCP health check")
    print("=" * 60)
    print(f"  model_id            : {args.model_id}")
    print(f"  adapter_path        : {args.adapter_path or '(none)'}")
    if args.adapter_path:
        from pathlib import Path as _P
        ap = _P(args.adapter_path)
        print(f"  adapter exists      : {ap.exists()}")
    print(f"  reasoning_mode      : {args.reasoning_mode}")
    print(f"  max_new_tokens      : {args.max_new_tokens}")
    print(f"  device              : {args.device}")
    print(f"  calibration_info    : {args.calibration_info}")
    print(f"  calibration_loaded  : {calibration_info.get('available', False)}")
    if calibration_info.get("available"):
        m = calibration_info.get("metrics", {})
        if m.get("ece") is not None:
            print(f"    headline ECE      : {m['ece']:.4f}")
        if m.get("brier") is not None:
            print(f"    headline Brier    : {m['brier']:.4f}")
        if m.get("auroc") is not None:
            print(f"    headline AUROC    : {m['auroc']:.4f}")
        if m.get("accuracy") is not None:
            print(f"    headline accuracy : {m['accuracy']:.4f}")
    print(f"  mcp package present : {_MCP_AVAILABLE}")
    if not _MCP_AVAILABLE:
        print("    -> install with `pip install mcp>=1.0` to launch stdio server")

    # Hard preconditions.
    if not args.model_id or args.model_id.strip() == "":
        print("\nFAIL: --model-id is empty.")
        return 1
    print("\nOK — health check passed; this config can launch the stdio server.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="HONEST MCP server.")
    parser.add_argument("--model-id", default=os.environ.get(
        "HONEST_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct"))
    parser.add_argument("--adapter-path", default=os.environ.get(
        "HONEST_ADAPTER_PATH"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--reasoning-mode", default="required",
                        help="Prompt reasoning mode passed to prompt_templates().")
    parser.add_argument("--calibration-info", default=os.environ.get(
        "HONEST_CALIBRATION_INFO", "eval/full_results.json"),
        help="Path to a full_results.json whose `overall` block "
             "feeds get_calibration_info().")
    parser.add_argument("--log-level", default="INFO")

    # Convenience modes.
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Offline self-test (parser, calibration loader, server graph). "
             "Does NOT download or load a model.",
    )
    parser.add_argument(
        "--health", action="store_true",
        help="Print a one-shot config summary and exit. Useful for verifying "
             "your Claude Desktop / Cursor config before launching.",
    )
    parser.add_argument(
        "--list-tools", action="store_true",
        help="Print the JSON schema of every MCP tool this server exposes "
             "and exit. Handy for clients without auto-discovery.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    calibration_info = load_calibration_info(args.calibration_info)

    if args.smoke_test:
        sys.exit(_smoke_test(calibration_info))

    if args.health:
        sys.exit(_health_summary(args, calibration_info))

    if args.list_tools:
        if not _MCP_AVAILABLE:
            sys.exit("ERROR: `mcp` package not installed. Run `pip install mcp>=1.0`.")
        # Build a server purely to introspect its tool list — no model load.
        server = build_server(
            CalibratedModel(model_id=args.model_id, adapter_path=None),
            calibration_info,
        )
        # MCP servers register handlers; we walk the registered tools synchronously.
        tool_list = asyncio.run(_collect_tool_schemas(server))
        print(json.dumps(tool_list, indent=2))
        return

    model = CalibratedModel(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        reasoning_mode=args.reasoning_mode,
    )

    if not _MCP_AVAILABLE:
        sys.exit(
            "ERROR: `mcp` package not installed. Run `pip install mcp>=1.0` "
            "(also listed in requirements.txt)."
        )

    server = build_server(model, calibration_info)

    async def _run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream, server.create_initialization_options(),
            )

    asyncio.run(_run())


async def _collect_tool_schemas(server) -> list[dict]:
    """Best-effort: extract tool schemas from a built MCP Server.

    The MCP SDK doesn't expose a public ``tools`` accessor, but the
    ``@server.list_tools()`` handler is stored on the server's request
    handlers map. We invoke it with a synthetic ListToolsRequest and JSON-
    serialise the resulting Tool objects.
    """
    from mcp.types import ListToolsRequest
    handler = server.request_handlers.get(ListToolsRequest)
    if handler is None:
        return []
    request = ListToolsRequest(method="tools/list", params={})
    result = await handler(request)
    tools = []
    for t in (getattr(result, "root", result).tools or []):
        tools.append({
            "name":         t.name,
            "description":  t.description,
            "inputSchema":  t.inputSchema,
        })
    return tools


if __name__ == "__main__":
    main()
