
"""Central registry for Person B's project tools."""

from tools.calculator import calculator
from tools.corpus import search_corpus
from tools.quote import quote_exact
from tools.messaging import send_message


TOOL_REGISTRY = {
    "calculator": calculator,
    "search_corpus": search_corpus,
    "quote_exact": quote_exact,
    "send_message": send_message,
}


def execute_tool(name: str, arguments: dict) -> dict:
    """Execute a registered tool by name with keyword arguments."""
    tool = TOOL_REGISTRY.get(name)

    if tool is None:
        return {
            "error": f"Unknown tool: {name}",
            "available_tools": list(TOOL_REGISTRY.keys()),
        }

    if not isinstance(arguments, dict):
        return {"error": "Tool arguments must be a dictionary."}

    try:
        result = tool(**arguments)
        return result if isinstance(result, dict) else {"result": str(result)}
    except Exception as exc:
        return {"error": f"Tool execution failed: {exc}"}