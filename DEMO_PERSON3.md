# Person 3 demo checklist

1. Start the app with `python ui.py`.
2. Open the RAG Assistant tab and submit a sample question. Show the returned pipeline result.
3. Open the Slack Message tab and enter a channel and message.
4. Leave the approval checkbox unchecked and click Send. Expected: “Not sent: select the human-approval checkbox first.”
5. In the terminal, run:
   `python -c "from tools.messaging import send_message; print(send_message('test', 'Hello'))"`
   Expected: human approval required.
6. If demonstrating a real Slack send, use a test channel, verify the message text and channel, check the approval box, and send only after the human explicitly approves. Never show the bot token in the demo.
7. For the MCP demo, show the `slack_send_message` tool definition and demonstrate `approved=False`. A live MCP connection requires configuration in the team's MCP client.
