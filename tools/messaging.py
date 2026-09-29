"""Messaging integration seam. Connect this function to the course MCP Slack tool."""
import os

def send_message(channel: str, message: str, approved: bool = False) -> dict:
    """Do not send without explicit approval and a configured integration."""
    if not approved:
        return {"sent": False, "result": "Human approval is required before sending."}
    endpoint = os.getenv("SLACK_MCP_ENDPOINT", "").strip()
    if not endpoint:
        return {
            "sent": False,
            "result": "Slack/MCP is not configured. Set SLACK_MCP_ENDPOINT and connect the course MCP client.",
        }
    # Intentionally not making a network request: wire in your provided MCP SDK/client here.
    return {
        "sent": False,
        "result": "Approval received, but the MCP client is not implemented in this starter.",
        "channel": channel,
    }
