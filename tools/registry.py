"""
Central Tool Registry and Tool Calling Schemas for Predictors Agentic Assistant.
Defines OpenAI-compatible tool specifications and execution dispatcher.
"""

from typing import Dict, Any, List
from tools.calculator import calculator
from tools.corpus import search_corpus
from tools.quote import quote_exact
from tools.messaging import send_message
from tools.outline import list_chapters

TOOL_REGISTRY = {
    "search_corpus": search_corpus,
    "calculator": calculator,
    "quote_exact": quote_exact,
    "send_message": send_message,
    "list_chapters": list_chapters,
}

# OpenAI-compatible function calling schemas
TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": (
                "Search 'The Crystal Ball Instruction Manual' (Volume One) using hybrid FAISS dense + BM25 keyword retrieval. "
                "Use this to find relevant textbook passages, explanations, definitions, and code examples."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language question or search query keywords."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve (default: 5, max: 10).",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Safely evaluate arithmetic and mathematical expressions. "
                "Always use this tool for arithmetic, calculations, page differences, averages, ratios, etc., "
                "instead of calculating in your head."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate, e.g. '155 - 43', '(120 + 74) / 2', 'sqrt(144) * 3'."
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "quote_exact",
            "description": (
                "Retrieve the exact full text of a specific page from a named document in the corpus. "
                "Use this to verify a passage, inspect surrounding context for adjacent sections, or quote exact wording."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {
                        "type": "integer",
                        "description": "The page number to read (1-based)."
                    },
                    "document": {
                        "type": "string",
                        "description": "The filename of the document (default: 'crystal1 (1).pdf').",
                        "default": "crystal1 (1).pdf"
                    },
                    "passage": {
                        "type": "string",
                        "description": "Optional specific sentence or phrase to check for presence on this page."
                    }
                },
                "required": ["page"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": (
                "Post an answer, summary, or notification to a Slack or Teams channel via MCP / Webhook. "
                "WARNING: Irreversible external action. Always requires explicit human approval before sending."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "The Slack or Teams channel name (e.g. 'general', 'data-science')."
                    },
                    "message": {
                        "type": "string",
                        "description": "The exact message text to send."
                    },
                    "approved": {
                        "type": "boolean",
                        "description": "True only if human user has explicitly confirmed/approved sending this message.",
                        "default": False
                    }
                },
                "required": ["channel", "message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_chapters",
            "description": (
                "Returns the table of contents and page numbers for all 29 chapters in The Crystal Ball manual. "
                "Useful when orienting where topics are located or calculating chapter page spans."
            ),
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]


def execute_tool(name: str, arguments: dict) -> dict:
    """Execute a registered tool by name with keyword arguments."""
    tool = TOOL_REGISTRY.get(name)

    if tool is None:
        return {
            "status": "error",
            "error": f"Unknown tool: '{name}'",
            "available_tools": list(TOOL_REGISTRY.keys()),
        }

    if not isinstance(arguments, dict):
        return {
            "status": "error",
            "error": "Tool arguments must be a dictionary.",
            "received_type": type(arguments).__name__
        }

    try:
        result = tool(**arguments)
        return result if isinstance(result, dict) else {"status": "success", "result": str(result)}
    except Exception as exc:
        return {
            "status": "error",
            "error": f"Tool execution failed: {type(exc).__name__}: {exc}"
        }