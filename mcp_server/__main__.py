"""Module entry-point for the HONEST MCP server.

Lets you run the server as a module:

    python -m mcp_server                                         # stdio server
    python -m mcp_server --smoke-test                            # offline check
    python -m mcp_server --health                                # health summary
    python -m mcp_server --model-id Qwen/Qwen2.5-3B-Instruct ...

The actual server logic lives in ``mcp_server.honest_mcp``. This wrapper
just makes ``python -m mcp_server`` work, which is what every MCP client
config example uses (Claude Desktop, Cursor, etc.).
"""

from __future__ import annotations

from .honest_mcp import main


if __name__ == "__main__":
    main()
