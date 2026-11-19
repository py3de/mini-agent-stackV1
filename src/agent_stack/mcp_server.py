"""MCP server exposing the same tools to any MCP client (Claude Desktop, IDEs, other agents).

The tools are not reimplemented here: the server wraps the existing LangChain tools, so a tool
is written once and is available to the LangGraph agent and over MCP.

Governance: MCP clients have no approval gate, so the server is **read-only by default** and only
exposes tools classified as `read`. Set MCP_ALLOW_WRITES=true to also expose write tools. Every MCP
tool call is recorded in the audit trail under the run ID "mcp".

Run (stdio):  python -m agent_stack.mcp_server
Claude Desktop config example is in the README.
"""

from __future__ import annotations

import inspect
import os

from mcp.server.mcpserver import MCPServer

from agent_stack.governance import get_audit, permission_for
from agent_stack.tools import get_tools

ALLOW_WRITES = os.getenv("MCP_ALLOW_WRITES", "false").lower() in ("1", "true", "yes")

server = MCPServer(
    "mini-agent-stack-crm",
    instructions="CRM tools: search contacts, deals and the knowledge base"
    + ("; log meeting notes and move deal stages." if ALLOW_WRITES else ". Read-only."),
)


def _register(lc_tool) -> None:
    """Expose one LangChain tool as an MCP tool with the same name, description and arguments."""

    def call(**kwargs):
        output = str(lc_tool.invoke(kwargs))
        get_audit().record("tool_call", "mcp", tool=lc_tool.name, args=kwargs,
                           permission=permission_for(lc_tool.name), decision="allowed", result=output[:300])
        return output

    fields = lc_tool.args_schema.model_fields
    call.__name__ = lc_tool.name
    call.__doc__ = lc_tool.description
    # Give the wrapper the tool's real signature so MCP generates the right input schema.
    call.__signature__ = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, annotation=f.annotation) for name, f in fields.items()],
        return_annotation=str,
    )
    call.__annotations__ = {name: f.annotation for name, f in fields.items()} | {"return": str}
    server.tool(name=lc_tool.name, description=lc_tool.description)(call)


for _tool in get_tools("crm"):
    if ALLOW_WRITES or permission_for(_tool.name) == "read":
        _register(_tool)


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
