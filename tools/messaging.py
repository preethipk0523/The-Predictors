"""Slack messaging helper with a required human-approval flag.

Compatible with the existing tools.registry.execute_tool(...). It does not
send anything unless approved=True and SLACK_BOT_TOKEN is configured.
"""
from __future__ import annotations

import os
from typing import Any


def send_message(channel: str, message: str, approved: bool = False) -> dict[str, Any]:
    channel = (channel or "").strip()
    message = (message or "").strip()

    if not approved:
        return {"sent": False, "result": "Human approval is required before sending."}
    if not channel or not message:
        return {"sent": False, "result": "Both channel and message are required."}

    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
        return {
            "sent": False,
            "result": "SLACK_BOT_TOKEN is not configured. No message was sent.",
        }

    try:
        from slack_sdk import WebClient
        from slack_sdk.errors import SlackApiError
    except ImportError:
        return {
            "sent": False,
            "result": "slack-sdk is not installed. Run: python -m pip install slack-sdk",
        }

    try:
        client = WebClient(token=token)
        response = client.chat_postMessage(channel=channel, text=message)
        return {
            "sent": bool(response.get("ok")),
            "channel": response.get("channel"),
            "ts": response.get("ts"),
            "result": "Message sent." if response.get("ok") else "Slack did not confirm delivery.",
        }
    except SlackApiError as exc:
        error = getattr(exc.response, "data", {}).get("error", str(exc))
        return {"sent": False, "result": f"Slack API error: {error}"}
    except Exception as exc:
        return {"sent": False, "result": f"Slack send failed: {exc}"}
