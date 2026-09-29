"""
Web UI for both projects, in one app:
    Tab 1 "RAG Assistant"   - the existing Tuesday UI (app/main.py, unchanged)
    Tab 2 "Agent"           - the agentic assistant: answer + citations, the tool calls it made
                              (which tools, in what order), and the human approval step for send_message

Run:  python app/agent_app.py      -> http://127.0.0.1:7860
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr  # noqa: E402

from app.agent import MAX_STEPS, deliver_message, run_agent  # noqa: E402
from app.main import CUSTOM_CSS, build_app  # noqa: E402

AGENT_EXAMPLES = [
    "What are the four levels of the data-to-wisdom hierarchy?",                     # 1 search
    "Which documents can you search?",                                              # list_documents
    "What is 17.5% of 2,340, rounded to 2 decimals?",                                # calculator only
    "How many levels does the data-to-wisdom hierarchy have? Multiply that number by 12.5.",  # search + calculate
    "Who won the 2024 T20 Cricket World Cup?",                                      # refuse, no tool
    "Explain Simpson's Paradox from the book and post a short summary to our team channel.",  # search + send (approval)
]


def format_trace(trace, steps, cost, seconds):
    if not trace:
        return (f"**No tool calls** — answered directly · {steps} step(s) / max {MAX_STEPS} · "
                f"${cost:.5f} · {seconds}s")
    lines = [f"**{len(trace)} tool call(s)** · {steps} model step(s) / max {MAX_STEPS} · "
             f"${cost:.5f} · {seconds}s\n",
             "| # | Step | Tool | Arguments | Result (preview) |",
             "|---|---|---|---|---|"]
    for i, t in enumerate(trace, 1):
        try:
            args = ", ".join(f"{k}={v!r}" for k, v in json.loads(t["arguments"] or "{}").items())
        except json.JSONDecodeError:
            args = t["arguments"]
        preview = t["result"][:140].replace("\n", " ").replace("|", "/")
        lines.append(f"| {i} | {t['step']} | `{t['tool']}` | {args.replace('|', '/')} | {preview}… |")
    return "\n".join(lines)


def ask_agent(question, chat_history):
    if not question or not question.strip():
        return chat_history, "", "", None, gr.update(visible=False), ""
    # keep the last 3 exchanges as context for follow-up questions
    history = []
    for m in (chat_history or [])[-6:]:
        history.append({"role": m["role"], "content": m["content"]})
    res = run_agent(question.strip(), history=history)
    chat_history = (chat_history or []) + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": res["answer"]},
    ]
    trace_md = format_trace(res["trace"], res["steps"], res["cost_usd"], res["seconds"])
    pending = res["pending_message"]
    if pending:
        box = f"### ✋ Approval needed\nThe agent wants to post this to the team channel:\n\n> {pending.replace(chr(10), chr(10) + '> ')}"
        return chat_history, "", trace_md, pending, gr.update(visible=True), box
    return chat_history, "", trace_md, None, gr.update(visible=False), ""


def approve(pending):
    if not pending:
        return None, gr.update(visible=False), "Nothing to send."
    return None, gr.update(visible=False), f"**Approved by user.** {deliver_message(pending)}"


def reject(pending):
    return None, gr.update(visible=False), "**Rejected by user.** Nothing was sent."


def build_agent_tab():
    with gr.Blocks() as tab:
        gr.Markdown("## 🤖 Agentic Assistant\nThe agent decides which tools to use: "
                    "`search_corpus`, `calculator`, `list_documents`, `send_message` (needs your approval).")
        pending_state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=3):
                try:  # Gradio 4/5 need type="messages"; Gradio 6 uses messages by default
                    chatbot = gr.Chatbot(label="Conversation", height=420, type="messages")
                except TypeError:
                    chatbot = gr.Chatbot(label="Conversation", height=420)
                question = gr.Textbox(label="Ask the agent", lines=2,
                                      placeholder="e.g. Find the value in the book, then calculate ...")
                with gr.Row():
                    ask_btn = gr.Button("Ask agent", variant="primary")
                    clear_btn = gr.Button("Clear")
                gr.Examples(AGENT_EXAMPLES, inputs=question)
            with gr.Column(scale=2):
                gr.Markdown("### 🛠️ Tool calls (in order)")
                trace_box = gr.Markdown("*The tools the agent calls will appear here.*")
                with gr.Group(visible=False) as approval_group:
                    approval_box = gr.Markdown()
                    with gr.Row():
                        yes_btn = gr.Button("✅ Approve & send", variant="primary")
                        no_btn = gr.Button("❌ Reject")
                send_status = gr.Markdown()

        outputs = [chatbot, question, trace_box, pending_state, approval_group, approval_box]
        ask_btn.click(ask_agent, [question, chatbot], outputs)
        question.submit(ask_agent, [question, chatbot], outputs)
        yes_btn.click(approve, [pending_state], [pending_state, approval_group, send_status])
        no_btn.click(reject, [pending_state], [pending_state, approval_group, send_status])
        clear_btn.click(lambda: ([], "", "*The tools the agent calls will appear here.*", None,
                                 gr.update(visible=False), ""),
                        outputs=[chatbot, question, trace_box, pending_state, approval_group, send_status])
    return tab


if __name__ == "__main__":
    demo = gr.TabbedInterface([build_app(), build_agent_tab()],
                              ["📚 RAG Assistant", "🤖 Agent"],
                              title="Predictors | Crystal Ball Assistant")
    demo.launch(server_name="127.0.0.1", server_port=7860, css=CUSTOM_CSS)
