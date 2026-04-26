# HONEST MCP Server

A production-ready [Model Context Protocol](https://modelcontextprotocol.io)
wrapper around the **trained, calibrated** HONEST model. Exposes the trained
model as two MCP tools so any MCP-compatible client (Claude Desktop, Cursor,
LangGraph agents, etc.) can consume calibrated reasoning as a service.

This is a **deployment artifact**: it loads an already-trained model and never
participates in training. Mixing serving and training in the same process
introduces hard-to-debug stability bugs, so the two are kept separate by design.

> **TL;DR**:
> ```bash
> python -m mcp_server --smoke-test         # offline self-test (no model load)
> python -m mcp_server --health             # config preflight
> python -m mcp_server                      # launch the stdio server
> ```

---

## What you get

| Tool                                | Returns                                                               |
| ----------------------------------- | --------------------------------------------------------------------- |
| `ask_with_calibrated_confidence`    | `{ answer, confidence, calibration_note, abstained, malformed, raw }` |
| `get_calibration_info`              | `{ available, model, preset, metrics: { ece, brier, auroc, ... }, ood: {...} }` |

Both tools are auto-discoverable via the standard MCP `tools/list` handshake.

```bash
python -m mcp_server --list-tools     # dump full JSON schema
```

---

## 1 · Tool: `ask_with_calibrated_confidence`

**Input**

```json
{
  "question": "What is 7^4 mod 11?",
  "domain":   "math"            // optional: "math" | "code" | "logic" | "general"
}
```

**Output (success)**

```json
{
  "answer":           "3",
  "confidence":       0.74,
  "abstained":        false,
  "malformed":        false,
  "calibration_note": "Confidence reflects an empirically calibrated probability of correctness, trained via Brier-score reward in the HONEST RL environment. Eval: ECE=0.0410  Brier=0.0670  AUROC=0.8320.",
  "raw":              "<reasoning>...</reasoning><answer>3</answer><confidence>0.74</confidence>"
}
```

**Output (abstain)**

```json
{
  "answer":           null,
  "confidence":       null,
  "abstained":        true,
  "malformed":        false,
  "calibration_note": "...",
  "raw":              "<abstain/>"
}
```

`abstained: true` is **not an error** — it's a deliberate, calibrated signal
that the model's confidence-of-correctness is below threshold. Downstream
agents should respect it.

---

## 2 · Tool: `get_calibration_info`

**Output (when `--calibration-info` is wired)**

```json
{
  "available": true,
  "model":     "Qwen/Qwen2.5-3B-Instruct",
  "preset":    "qwen3b",
  "reasoning_mode": "required",
  "metrics": {
    "n_samples": 600,
    "ece":       0.041,
    "ace":       0.039,
    "mce":       0.114,
    "brier":     0.067,
    "nll":       0.282,
    "auroc":     0.832,
    "accuracy":  0.612,
    "format_rate": 0.991
  },
  "ood": {
    "medical": { "n": 200, "ece": 0.072, "brier": 0.183, "auroc": 0.781, "accuracy": 0.524 },
    "legal":   { "n": 200, "ece": 0.064, "brier": 0.151, "auroc": 0.804, "accuracy": 0.578 }
  },
  "source_file": "eval/full_results.json"
}
```

Use this to decide *how much to trust the model's confidence* before acting
on it. Headline rule of thumb: ECE < 0.05 → trust the bin; AUROC > 0.80 →
the confidence ranking is meaningful.

If no calibration JSON is configured, the tool returns
`{ "available": false, "reason": "..." }` rather than failing.

---

## 3 · Install

```bash
# All-in-one (project root requirements):
pip install -r requirements.txt

# Or just the MCP serving subset:
pip install "mcp>=1.0" transformers accelerate torch peft
```

---

## 4 · Run locally (stdio)

```bash
python -m mcp_server \
  --model-id        Qwen/Qwen2.5-3B-Instruct \
  --adapter-path    ./honest-qwen-3b-grpo/final_adapters \
  --calibration-info eval/full_results.json
```

Environment-variable equivalents are honoured (handy for Claude Desktop):

| Flag                  | Env var                  |
| --------------------- | ------------------------ |
| `--model-id`          | `HONEST_MODEL_ID`        |
| `--adapter-path`      | `HONEST_ADAPTER_PATH`    |
| `--calibration-info`  | `HONEST_CALIBRATION_INFO`|

### Convenience modes

```bash
# Offline self-test — does NOT load a model. Run this first.
python -m mcp_server --smoke-test

# Preflight summary of the config you're about to launch.
python -m mcp_server --health \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --adapter-path ./honest-qwen-3b-grpo/final_adapters

# Dump the JSON tool schema — handy when debugging a non-auto-discovery client.
python -m mcp_server --list-tools
```

---

## 5 · Claude Desktop integration

Add to your Claude Desktop config:

* **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
* **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
* **Linux**:   `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "honest": {
      "command": "python",
      "args": [
        "-m", "mcp_server",
        "--model-id",          "Qwen/Qwen2.5-3B-Instruct",
        "--adapter-path",      "/absolute/path/to/HONEST-Env/honest-qwen-3b-grpo/final_adapters",
        "--calibration-info",  "/absolute/path/to/HONEST-Env/eval/full_results.json"
      ],
      "cwd": "/absolute/path/to/HONEST-Env",
      "env": {
        "PYTHONPATH": "/absolute/path/to/HONEST-Env"
      }
    }
  }
}
```

Fully restart Claude Desktop (quit, don't just close the window). The
`honest` tools should now appear in the tool picker (the wrench icon).

### Verifying

After restarting Claude Desktop, ask the assistant:

> Use the `get_calibration_info` tool from the `honest` server.

If everything is wired correctly, you'll see a JSON dump of your model's
calibration metrics inline. If not:

1. Run `python -m mcp_server --health` from the same `cwd` Claude Desktop
   uses, with the same `--model-id` / `--adapter-path` flags. This shows
   the exact config the server will see.
2. Check Claude Desktop's MCP logs:
   * macOS: `~/Library/Logs/Claude/mcp*.log`
   * Windows: `%LOCALAPPDATA%\Claude\Logs\mcp*.log`

---

## 6 · Cursor integration

Cursor reuses Claude Desktop's MCP config format. Open
**Cursor Settings → Features → MCP** and add the same JSON snippet as
section 5. Restart Cursor.

To call the tools from a Cursor agent, mention the server name:

> @honest answer "What is 7^4 mod 11?" using the calibrated tool

---

## 7 · LangGraph / custom agents

Any standard MCP client works. Minimal Python example:

```python
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
import asyncio

async def ask(question: str):
    params = StdioServerParameters(
        command="python",
        args=["-m", "mcp_server",
              "--model-id", "Qwen/Qwen2.5-3B-Instruct",
              "--adapter-path", "./honest-qwen-3b-grpo/final_adapters",
              "--calibration-info", "./eval/full_results.json"],
        env={"PYTHONPATH": "."},
    )
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            result = await s.call_tool(
                "ask_with_calibrated_confidence",
                {"question": question},
            )
            return result.content[0].text

print(asyncio.run(ask("What is 7^4 mod 11?")))
```

---

## 8 · Architecture & invariants

* The MCP server **only loads the trained model**. It does not connect to
  the training environment, the difficulty controller, the replay buffer,
  or W&B. Mixing training and serving is a canonical source of
  training-stability bugs, so we keep them in separate processes.
* Confidence values are emitted by the model under a Brier-score reward,
  so they are an *empirically* calibrated estimate of `P(correct)` — not a
  post-hoc Platt scaling. See `eval/full_results.json` for the
  per-condition breakdown.
* The strict XML parser is the same one used during RL training
  (`server.reward.parse_action`). On a slightly malformed completion we
  fall back to the lenient parser so the tool still returns *something*
  actionable rather than failing. The `malformed: true` flag tells the
  caller when this fallback fired.

---

## 9 · Troubleshooting

| Symptom                                                        | Fix                                                                                       |
| -------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `python -m mcp_server` immediately exits with `mcp not installed` | `pip install "mcp>=1.0"` (or `pip install -r requirements.txt`).                        |
| `--smoke-test` step 4 says "SKIP — mcp package not installed"  | Same as above; smoke test still passes — the MCP wiring just couldn't be exercised.       |
| Claude Desktop shows the server but no tools                   | Re-run `python -m mcp_server --list-tools` with the *exact same flags* and confirm two tools appear. If not, the model_id is unreachable from your `cwd` / `PYTHONPATH`. |
| `available: false, reason: file not found`                     | Pass `--calibration-info /absolute/path/to/eval/full_results.json` (relative paths break in Claude Desktop because its `cwd` may differ). |
| Confidence values are always `0.5`                             | The adapter wasn't loaded. Check `--adapter-path` exists; run `--health` to confirm. |
| Confidence values are wildly miscalibrated                     | Re-run `eval/full_eval.py` to refresh `full_results.json`. The `calibration_note` reflects the metrics file, not live calibration. |
| MCP tool returns `malformed: true` for sensible inputs         | The model's chat template is mismatched — make sure `--model-id` matches the adapter's base model exactly. |

---

## 10 · Why this exists (research framing)

Most LLM tools are uncalibrated: they emit answers with no quantified
uncertainty. HONEST trains a model to emit `(answer, confidence)` pairs
where `confidence` is a strictly proper estimate of `P(correct)` under a
Brier-score reward. This MCP server packages that calibrated model so
**other agents can route around it**:

* Use the `confidence` to gate downstream actions (`if confidence < 0.6: ask_user`).
* Use `get_calibration_info` to set the gating threshold *per domain*
  based on the OOD breakdown.
* Use `abstained: true` as a signal to consult a different model
  ("calibrated cascade").

This is the deployment-side complement to the self-learning RL pipeline
described in `SELF_LEARNING.md` — the model learns to be calibrated, the
MCP wrapper makes that calibration available to other systems.
