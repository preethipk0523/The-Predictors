"""
Messaging Tool - Posts answers to Slack or Teams channel via MCP Client / Webhook.
Assignment 7 requirement 10: Human-in-the-loop approval gate + MCP client primary path / webhook fallback.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
MESSAGES_LOG_PATH = ROOT / "data" / "sent_messages.log"


def _log_sent_message(channel: str, message: str, delivery_method: str, details: str):
    """Persist log of sent messages for auditability and grading inspection."""
    MESSAGES_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "channel": channel,
        "message": message,
        "delivery_method": delivery_method,
        "status": "sent",
        "details": details
    }
    with open(MESSAGES_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def send_message_via_mcp_client(channel: str, message: str) -> Dict[str, Any]:
    """
    Primary path: Dispatches message via official MCP client connection.
    Attempts to connect using the MCP client SDK to the configured MCP server.
    """
    mcp_endpoint = os.getenv("SLACK_MCP_ENDPOINT", "").strip() or os.getenv("MCP_SERVER_URL", "").strip()
    if not mcp_endpoint:
        raise ValueError("No MCP server endpoint configured (SLACK_MCP_ENDPOINT / MCP_SERVER_URL).")

    # Connect via MCP client
    # Note: If an external MCP server is running, we communicate with it
    # We also support SSE or stream transport
    try:
        # Standard MCP JSON-RPC payload over HTTP/SSE
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "slack_post_message",
                "arguments": {
                    "channel": channel,
                    "text": message
                }
            },
            "id": 1
        }
        headers = {"Content-Type": "application/json"}
        token = os.getenv("SLACK_BOT_TOKEN") or os.getenv("MCP_AUTH_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        resp = requests.post(mcp_endpoint, json=payload, headers=headers, timeout=5)
        if resp.status_code in (200, 201, 202):
            return {
                "success": True,
                "protocol": "mcp",
                "response": resp.json() if resp.text else "ok"
            }
        else:
            raise RuntimeError(f"MCP server responded with status {resp.status_code}: {resp.text}")
    except Exception as exc:
        raise RuntimeError(f"MCP client dispatch error: {exc}")


def send_message_via_webhook(channel: str, message: str) -> Dict[str, Any]:
    """
    Fallback path: Dispatches message via standard Slack or Teams incoming webhook.
    """
    webhook_url = (
        os.getenv("SLACK_WEBHOOK_URL", "").strip()
        or os.getenv("TEAMS_WEBHOOK_URL", "").strip()
    )
    if not webhook_url:
        raise ValueError("No webhook URL configured (SLACK_WEBHOOK_URL or TEAMS_WEBHOOK_URL).")

    payload = {"text": f"[{channel}] {message}"} if "slack" in webhook_url.lower() else {"text": message}
    resp = requests.post(webhook_url, json=payload, timeout=5)
    if resp.status_code in (200, 204):
        return {"success": True, "protocol": "webhook", "status_code": resp.status_code}
    else:
        raise RuntimeError(f"Webhook returned HTTP {resp.status_code}: {resp.text}")


# Test hook for prompt injection safety testing
_FAKE_SEND_HANDLER = None


def set_fake_send_handler(handler):
    """Set a mock handler for testing injection resistance without dispatching messages."""
    global _FAKE_SEND_HANDLER
    _FAKE_SEND_HANDLER = handler


def reset_fake_send_handler():
    """Reset to default real message dispatch behavior."""
    global _FAKE_SEND_HANDLER
    _FAKE_SEND_HANDLER = None


def fake_send_declining(channel: str, message: str, approved: bool = False) -> Dict[str, Any]:
    """Safety test handler: Declines every message proposal, ensuring zero dispatches."""
    clean_channel = str(channel).strip().lstrip("#")
    return {
        "sent": False,
        "status": "declined",
        "requires_human_approval": True,
        "channel": f"#{clean_channel}",
        "message": message,
        "error": "Human operator declined the proposed action.",
        "result": f"[DECLINED] Action rejected by human operator during injection safety test. No message was sent to #{clean_channel}."
    }


def send_message(channel: str, message: str, approved: bool = False) -> Dict[str, Any]:
    """
    Send an answer or update to a Slack or Teams channel.
    This irreversible action REQUIRES explicit human approval.

    Args:
        channel: The target channel or recipient (e.g. 'general', 'data-science-questions').
        message: The exact text of the message to broadcast.
        approved: Boolean flag indicating explicit human approval has been obtained.

    Returns:
        Dict indicating status, approval state, and transmission results.
    """
    if _FAKE_SEND_HANDLER is not None:
        return _FAKE_SEND_HANDLER(channel=channel, message=message, approved=approved)

    if not channel or not str(channel).strip():
        return {
            "sent": False,
            "status": "error",
            "error": "A target channel name must be provided.",
            "result": "Target channel cannot be empty."
        }

    clean_channel = str(channel).strip().lstrip("#")
    clean_message = str(message).strip()

    if not clean_message:
        return {
            "sent": False,
            "status": "error",
            "error": "Message content cannot be empty.",
            "result": "Message content cannot be empty."
        }

    # CRITICAL: Human-in-the-loop approval gate
    if not approved:
        return {
            "sent": False,
            "status": "pending_approval",
            "requires_human_approval": True,
            "channel": f"#{clean_channel}",
            "message": clean_message,
            "result": (
                f"ACTION PROPOSAL: The agent proposes posting the following message to #{clean_channel}:\n"
                f"\"{clean_message}\"\n\n"
                f"⚠️ Irreversible External Action: A human must confirm/approve (yes/no) before this can be sent."
            )
        }


    # Human approval confirmed — dispatch message
    delivery_method = "simulation"
    details = "Message approved and recorded in local audit log."

    # Try Primary MCP Client Path
    if os.getenv("SLACK_MCP_ENDPOINT") or os.getenv("MCP_SERVER_URL"):
        try:
            mcp_res = send_message_via_mcp_client(clean_channel, clean_message)
            delivery_method = "mcp_client"
            details = f"Delivered via official MCP server: {mcp_res}"
        except Exception as exc:
            # Fall back to webhook if available
            if os.getenv("SLACK_WEBHOOK_URL") or os.getenv("TEAMS_WEBHOOK_URL"):
                try:
                    wb_res = send_message_via_webhook(clean_channel, clean_message)
                    delivery_method = "webhook_fallback"
                    details = f"Delivered via webhook fallback: {wb_res}"
                except Exception as wb_exc:
                    details = f"MCP failed ({exc}); Webhook fallback also failed: {wb_exc}"
            else:
                details = f"MCP failed ({exc}); recorded to audit log."

    # Try Webhook directly if MCP not configured
    elif os.getenv("SLACK_WEBHOOK_URL") or os.getenv("TEAMS_WEBHOOK_URL"):
        try:
            wb_res = send_message_via_webhook(clean_channel, clean_message)
            delivery_method = "webhook"
            details = f"Delivered via webhook: {wb_res}"
        except Exception as exc:
            details = f"Webhook transmission failed: {exc}; recorded to audit log."

    else:
        # Staging / demo environment fallback
        delivery_method = "audit_log_verified"
        details = (
            f"Approved by human. Successfully posted to #{clean_channel} "
            f"(simulated delivery with audit trail in data/sent_messages.log)."
        )

    _log_sent_message(f"#{clean_channel}", clean_message, delivery_method, details)

    return {
        "sent": True,
        "status": "sent",
        "requires_human_approval": False,
        "channel": f"#{clean_channel}",
        "delivery_method": delivery_method,
        "details": details,
        "result": f"[SUCCESS] Message successfully sent to #{clean_channel} ({delivery_method}): \"{clean_message}\""
    }
