"""Gradio UI for The Predictors with explicit human approval for Slack messages.

Run from the repository root:
    python ui.py

This UI calls the existing RAG pipeline. Slack sending is optional and only
available when Slack credentials are configured and the approval checkbox is
selected.
"""
from __future__ import annotations

import json
import os
import gradio as gr

from app.pipeline import RAGPipeline
from tools.messaging import send_message

pipeline = RAGPipeline()


def ask_question(question: str):
    question = (question or "").strip()
    if not question:
        return "Please enter a question."
    try:
        result = pipeline.run(question, top_k=7)
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False, indent=2, default=str)
        return str(result)
    except Exception as exc:
        return f"RAG pipeline error: {exc}"


def slack_send(channel: str, message: str, approved: bool):
    channel = (channel or "").strip()
    message = (message or "").strip()
    if not channel or not message:
        return "Enter both a Slack channel and a message."
    if not approved:
        return "Not sent: select the human-approval checkbox first."
    result = send_message(channel=channel, message=message, approved=True)
    return json.dumps(result, ensure_ascii=False, indent=2, default=str)


with gr.Blocks(title="The Predictors") as demo:
    gr.Markdown("# The Predictors")
    gr.Markdown("Ask questions using the existing RAG pipeline, or prepare a Slack message.")

    with gr.Tab("RAG Assistant"):
        question = gr.Textbox(label="Question", placeholder="Ask about the project corpus...", lines=3)
        ask_button = gr.Button("Ask", variant="primary")
        answer = gr.Textbox(label="Answer / pipeline result", lines=14, interactive=False)
        ask_button.click(ask_question, inputs=question, outputs=answer)
        question.submit(ask_question, inputs=question, outputs=answer)

    with gr.Tab("Slack Message (approval required)"):
        gr.Markdown(
            "Messages are sent only when you explicitly approve them. "
            "Configure `SLACK_BOT_TOKEN` in your environment and install `slack-sdk` first."
        )
        channel = gr.Textbox(label="Slack channel", placeholder="e.g. #project-updates or channel ID")
        message = gr.Textbox(label="Message", lines=4, placeholder="Write the message to send...")
        approved = gr.Checkbox(label="I have reviewed this message and approve sending it.")
        send_button = gr.Button("Send approved Slack message")
        send_result = gr.Textbox(label="Result", lines=5, interactive=False)
        send_button.click(slack_send, inputs=[channel, message, approved], outputs=send_result)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
