"""Tool registry.

To add a tool: write it in a module under tools/, then add it to a toolset below. The agent,
the MCP server and the API pick it up automatically.
"""

from agent_stack.tools.crm import CRM_TOOLS
from agent_stack.tools.knowledge import KNOWLEDGE_TOOLS

TOOLSETS = {
    "crm": CRM_TOOLS + KNOWLEDGE_TOOLS,
    "crm_only": CRM_TOOLS,
    "knowledge": KNOWLEDGE_TOOLS,
}


def get_tools(toolset: str = "crm"):
    return list(TOOLSETS[toolset])
