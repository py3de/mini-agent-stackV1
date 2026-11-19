"""MCP server exposing the same CRM tools to any MCP client (Claude Desktop, IDEs, other agents).

The tools are not reimplemented here: the server wraps the existing LangChain tools, so a tool
is written once and is available to the LangGraph agent and over MCP.

Run (stdio):  python -m agent_stack.mcp_server
Claude Desktop config example is in the README.
"""

from __future__ import annotations

import inspect

from mcp.server.mcpserver import MCPServer

from agent_stack.tools import get_tools

server = MCPServer(
    "mini-agent-stack-crm",
    instructions="CRM tools: search contacts and deals, log meeting notes, move deal stages.",
)


def _register(lc_tool) -> None:
    """Expose one LangChain tool as an MCP tool with the same name, description and arguments."""

    def call(**kwargs):
        return lc_tool.invoke(kwargs)

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
    _register(_tool)


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
