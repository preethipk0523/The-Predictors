"""Agent core: think -> act -> observe loop (Person A).

Usage:
    from app.agent import run_agent
    result = run_agent("What is ...?")

Quick test:
    python -m app.agent "Use the echo tool to say hello"
"""

import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from app.logger import RunLogger
from app.prompts import (
    GIVE_UP_HINT,
    REPEAT_MESSAGE,
    STEP_CAP_PROMPT,
    SYSTEM_PROMPT,
)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Explicitly load .env from the project root
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH, override=True)

# Import configuration only after .env has been loaded
from app.config import OPENAI_API_KEY, CHAT_MODEL

MODEL = CHAT_MODEL
MAX_STEPS = 6
MAX_NO_RESULTS = 3

_client = None


def get_client() -> OpenAI:
    """Create the OpenAI client once and reuse it."""
    global _client

    if _client is None:
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is missing. Add it to the project root .env file."
            )

        if OPENAI_API_KEY == "your_api_key_here":
            raise ValueError(
                "OPENAI_API_KEY is still using the placeholder value."
            )

        _client = OpenAI(api_key=OPENAI_API_KEY)

    return _client


# --------------------------------------------------------------------------
# Tool registry
# Use Person B/C's real tools if present, otherwise fake tools.
# --------------------------------------------------------------------------

def _echo(text: str = "") -> str:
    return f"ECHO: {text}"


def _fake_send_message(text: str = "") -> str:
    return "SENT (fake)"


_FAKE_TOOLS = {
    "echo": {
        "fn": _echo,
        "schema": {
            "name": "echo",
            "description": "Testing tool. Repeats the given text back.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string"
                    }
                },
                "required": ["text"],
            },
        },
    },

    "send_message": {
        "fn": _fake_send_message,
        "schema": {
            "name": "send_message",
            "description": (
                "Posts a message to the team Slack channel. "
                "Only use when the user explicitly asks to post "
                "or share the answer. Requires human approval."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string"
                    }
                },
                "required": ["text"],
            },
        },
    },
}


def load_tools() -> dict:
    """Load real tools when available, otherwise use fake tools."""
    try:
        from app.tools import TOOLS
        return TOOLS

    except Exception as e:
        print(
            f"[agent] Using fake tools "
            f"({type(e).__name__}: {e})"
        )
        return _FAKE_TOOLS


def _schemas(tools: dict) -> list:
    """Convert the tool registry into OpenAI tool schemas."""
    output = []

    for tool in tools.values():
        schema = tool["schema"]

        if schema.get("type") == "function":
            output.append(schema)
        else:
            output.append(
                {
                    "type": "function",
                    "function": schema,
                }
            )

    return output


# --------------------------------------------------------------------------
# Safe tool execution
# --------------------------------------------------------------------------

def run_tool(tools: dict, name: str, args) -> str:
    """Run a tool safely. Never raise an exception."""

    if name not in tools:
        return (
            f"ERROR: unknown tool '{name}'. "
            f"Available: {list(tools.keys())}"
        )

    if not isinstance(args, dict):
        return "ERROR: invalid arguments, expected a JSON object"

    try:
        return str(tools[name]["fn"](**args))

    except TypeError as e:
        return f"ERROR: bad arguments: {e}"

    except Exception as e:
        return f"ERROR: {e}"


def _parse_args(raw: str):
    """Safely parse tool-call JSON arguments."""

    try:
        args = json.loads(raw or "{}")

        if isinstance(args, dict):
            return args

        return None

    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------
# Citation extraction
# --------------------------------------------------------------------------

ANSWER_CITE = re.compile(
    r"\(([^()\n]+?),\s*page\s*(\d+)\)",
    re.IGNORECASE,
)

SEARCH_CITE = re.compile(
    r"Doc:\s*(.+?)\s*\|\s*Page:\s*(\d+)",
    re.IGNORECASE,
)


def extract_citations(answer: str, trajectory: list) -> list:
    """Extract document/page citations."""

    seen = set()
    citations = []

    def add(doc, page):
        key = (doc.strip(), int(page))

        if key not in seen:
            seen.add(key)

            citations.append(
                {
                    "doc": key[0],
                    "page": key[1],
                }
            )

    # First try citations from the final answer
    for doc, page in ANSWER_CITE.findall(answer or ""):
        add(doc, page)

    # Fallback to search results
    if not citations:
        for item in trajectory:

            if item.get("tool") == "search_corpus":

                for doc, page in SEARCH_CITE.findall(
                    item.get("result", "")
                ):
                    add(doc, page)

    return citations


# --------------------------------------------------------------------------
# Main agent loop
# --------------------------------------------------------------------------

def run_agent(question: str) -> dict:
    """Run the think -> act -> observe agent."""

    tools = load_tools()
    schemas = _schemas(tools)

    log = RunLogger(question)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    answer = ""
    stopped_reason = "answered"
    pending_message = None

    steps = 0

    seen_calls = set()
    consecutive_empty = 0

    try:

        for step in range(1, MAX_STEPS + 1):

            steps = step

            # Last allowed model call
            forced = step == MAX_STEPS

            if forced:
                messages.append(
                    {
                        "role": "user",
                        "content": STEP_CAP_PROMPT,
                    }
                )

            kwargs = {
                "model": MODEL,
                "temperature": 0,
                "messages": messages,
                "tools": schemas,
            }

            if forced:
                kwargs["tool_choice"] = "none"

            # --------------------------------------------------------------
            # Model call
            # --------------------------------------------------------------

            response = get_client().chat.completions.create(**kwargs)

            log.add_usage(response.usage)

            message = response.choices[0].message

            # --------------------------------------------------------------
            # No tool requested -> final answer
            # --------------------------------------------------------------

            if not message.tool_calls:

                answer = message.content or ""

                if forced:
                    stopped_reason = "step_cap"

                break

            # Keep assistant message including tool_calls
            messages.append(
                message.model_dump(exclude_none=True)
            )

            # --------------------------------------------------------------
            # Execute requested tools
            # --------------------------------------------------------------

            for call in message.tool_calls:

                name = call.function.name

                args = _parse_args(
                    call.function.arguments
                )

                # ----------------------------------------------------------
                # Human approval for send_message
                # ----------------------------------------------------------

                if name == "send_message":

                    text = (args or {}).get("text", "")

                    log.log_call(
                        step,
                        name,
                        args,
                        "PENDING_APPROVAL: not sent",
                    )

                    pending_message = text

                    stopped_reason = "pending_approval"

                    answer = (
                        "I've drafted a message "
                        "for your approval."
                    )

                    break

                # ----------------------------------------------------------
                # Repeated identical tool call
                # ----------------------------------------------------------

                key = (
                    name,
                    json.dumps(
                        args,
                        sort_keys=True,
                        default=str,
                    ),
                )

                if args is not None and key in seen_calls:

                    result = REPEAT_MESSAGE

                elif args is None:

                    result = (
                        "ERROR: invalid arguments, "
                        "expected JSON"
                    )

                else:

                    seen_calls.add(key)

                    result = run_tool(
                        tools,
                        name,
                        args,
                    )

                # ----------------------------------------------------------
                # NO_RESULTS recovery
                # ----------------------------------------------------------

                if result.startswith("NO_RESULTS"):

                    consecutive_empty += 1

                    if consecutive_empty >= MAX_NO_RESULTS:
                        result += GIVE_UP_HINT

                elif not result.startswith("ERROR"):

                    consecutive_empty = 0

                # ----------------------------------------------------------
                # Log tool call
                # ----------------------------------------------------------

                log.log_call(
                    step,
                    name,
                    args,
                    result,
                )

                # Give tool result back to model
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    }
                )

            if stopped_reason == "pending_approval":
                break

    # ----------------------------------------------------------------------
    # Never crash the UI
    # ----------------------------------------------------------------------

    except Exception as e:

        answer = (
            f"Sorry, the agent failed: "
            f"{type(e).__name__}: {e}"
        )

        stopped_reason = "error"

    # ----------------------------------------------------------------------
    # Final processing
    # ----------------------------------------------------------------------

    citations = extract_citations(
        answer,
        log.trajectory,
    )

    ui_trajectory = log.ui_trajectory()

    log.finish(
        answer,
        stopped_reason,
        steps,
    )

    return {
        "answer": answer,
        "citations": citations,
        "trajectory": ui_trajectory,
        "steps": steps,
        "stopped_reason": stopped_reason,
        "pending_message": pending_message,
    }


# --------------------------------------------------------------------------
# Command-line testing
# --------------------------------------------------------------------------

if __name__ == "__main__":

    question = (
        " ".join(sys.argv[1:])
        or "Use the echo tool to say hello."
    )

    result = run_agent(question)

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )