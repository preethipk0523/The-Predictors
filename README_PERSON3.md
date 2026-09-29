# Person 3 — UI, Slack MCP tool, human approval, docs and demo

## Files included
- `ui.py` — Gradio UI with a RAG tab and an approval-gated Slack tab.
- `tools/messaging.py` — Slack send helper; compatible with the existing tool registry.
- `tools/slack_mcp_server.py` — MCP server exposing `slack_send_message`.
- `.env.example` — environment variable example (no real token).
- `requirements-person3.txt` — extra dependencies.
- `DEMO_PERSON3.md` — demo steps and expected behavior.

## Add files to the repository
Copy `ui.py` into the repository root.
Copy `tools/messaging.py` and `tools/slack_mcp_server.py` into the existing `tools/` folder.
Do not replace `tools/registry.py` or the existing pipeline files.

## Install dependencies
Activate your project virtual environment, then run:

```powershell
python -m pip install -r requirements-person3.txt
```

If the project already has a `requirements.txt`, you can instead add these dependencies to it:
- `gradio`
- `slack-sdk`
- `mcp`

## Configure Slack safely
1. Create/configure a Slack app and add the `chat:write` bot token scope.
2. Install the app to the workspace and invite the bot to the target channel if needed.
3. Set the bot token in the current PowerShell session (do not paste it into code or Git):

```powershell
$env:SLACK_BOT_TOKEN = "xoxb-your-token-here"
```

Do not commit `.env` files or tokens. Never share a real token in screenshots or chat.

## Run the UI
From the repository root:

```powershell
python ui.py
```

Open http://127.0.0.1:7860.

The RAG tab assumes `app.pipeline.RAGPipeline()` and `pipeline.run(question, top_k=7)` work in the existing project. If the project API differs, adapt those calls to match it.

## Run the MCP server
In a separate terminal, with the virtual environment active and `SLACK_BOT_TOKEN` set:

```powershell
python -m tools.slack_mcp_server
```

Connect this server to your MCP client using that client's documented stdio/server configuration. The exact configuration depends on the MCP client your team uses.

## Human approval behavior
- `approved=False` must not send a message.
- The Gradio Slack tab requires a checkbox to be selected before calling the send helper.
- The MCP tool requires the caller to pass `approved=True` after a human has reviewed the exact message.
- This is an approval flag, not an independent identity/authorization system. The MCP client must implement the human review step; do not automatically set `approved=True`.

## Verify without sending a real message
```powershell
python -c "from tools.messaging import send_message; print(send_message('test', 'Hello'))"
```
Expected: `sent` is `False` and the result says human approval is required.

Without a configured token, even an approved test returns `sent: False` and says no message was sent.

## Existing registry compatibility
Your existing `tools.registry` imports `send_message` from `tools.messaging`; this implementation preserves that function name and signature. Person B should retest the registry after this file is copied in.
