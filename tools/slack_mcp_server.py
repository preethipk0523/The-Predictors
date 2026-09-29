"""Minimal MCP server exposing an approval-gated Slack send_message tool.

Run with:
    python -m tools.slack_mcp_server

Configure SLACK_BOT_TOKEN in the environment. The calling client must pass
approved=True only after a human has explicitly reviewed and approved the
message. This server does not itself display an approval prompt.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from tools.messaging import send_message

mcp = FastMCP("predictors-slack")


@mcp.tool()
def slack_send_message(channel: str, message: str, approved: bool = False) -> dict:
    """Send a Slack message only after explicit human approval.

    Args:
        channel: Slack channel name (for example #project-updates) or channel ID.
        message: Text to send.
        approved: Must be true only after a human has approved this exact message.
    """
    return send_message(channel=channel, message=message, approved=approved)


if __name__ == "__main__":
    mcp.run()
