"""
Gradio Web Interface for The Predictors — Crystal Ball RAG System.
Clean, Minimal, Expandable UI with Always-Visible Chatbot.

Key Features:
- Dual Engine: Autonomous Multi-Tool Agent (ReAct loop) vs Direct Hybrid RAG
- Always-Visible Chatbot with full conversation history and answer citations
- Expandable Navigation Sidebar with selectable sections (hidden by default):
    1. Document / Corpus Information (metadata, statistics, 29-chapter directory)
    2. Retrieval Details (hybrid search architecture, ReAct agent loop, real-time trajectory)
    3. Sources and Citations (live evidence cards, page numbers, similarity scores, grounding)
    4. Evaluation Results (recall benchmarks, refusal rates, latency, category breakdown)
    5. System Status / Settings (budget spend, infrastructure health, message dispatch audit log)
- Human-in-the-Loop Approval Interface for Slack / Teams actions
- Clean, minimal, glassmorphic dark theme with responsive styling
"""

import os
import sys
import re
import json
import time
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

# Ensure root workspace is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
from app.pipeline import pipeline
from app.agent import agent
from app.cost_tracker import cost_tracker
from app.chat_memory import chat_memory
from app.feedback import save_feedback
from app.confidence import calculate_retrieval_confidence
from app.faithfulness import verify_faithfulness
from app.query_rewriter import rewrite_query
from app.ab_comparison import run_retrieval_ab_comparison, format_ab_comparison_html
from app.metrics_dashboard import load_or_run_benchmark, run_benchmark_evaluation, format_metrics_dashboard_html
from tools.messaging import send_message
from tools.outline import CHAPTERS

CUSTOM_CSS = """
/* Modern Dark Glassmorphic Theme with Clean Spacing & Typography */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --primary: #6366f1;
    --primary-hover: #4f46e5;
    --primary-glow: rgba(99, 102, 241, 0.35);
    --bg-dark: #0f172a;
    --card-bg: rgba(30, 41, 59, 0.7);
    --card-border: rgba(99, 102, 241, 0.2);
    --border-subtle: rgba(255, 255, 255, 0.08);
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-cyan: #06b6d4;
    --accent-purple: #a855f7;
    --accent-rose: #f43f5e;
}

body, .gradio-container {
    background: radial-gradient(ellipse at 15% 0%, #1e1b4b 0%, #0f172a 45%, #020617 100%) !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: var(--text-main) !important;
    min-height: 100vh;
}

/* Custom Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: rgba(15, 23, 42, 0.6);
}
::-webkit-scrollbar-thumb {
    background: rgba(99, 102, 241, 0.4);
    border-radius: 9999px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(99, 102, 241, 0.7);
}

/* Top Header Bar */
#header-bar {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 16px 24px;
    margin-bottom: 16px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 0 20px rgba(99, 102, 241, 0.12);
    backdrop-filter: blur(16px);
}

.title-badge {
    display: inline-block;
    background: linear-gradient(90deg, #6366f1, #06b6d4);
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 3px 10px;
    border-radius: 9999px;
    margin-bottom: 4px;
}

.mode-badge {
    display: inline-block;
    background: rgba(168, 85, 247, 0.15);
    color: #c084fc;
    border: 1px solid rgba(168, 85, 247, 0.35);
    font-size: 0.75rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 6px;
    margin-left: 8px;
}

.budget-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.15);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.3);
    font-size: 0.75rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 9999px;
}

/* Main App Row Layout */
#main-app-row {
    display: flex !important;
    gap: 18px !important;
    align-items: flex-start !important;
    width: 100% !important;
}

/* Sidebar Navigation Column - Narrow Left Sidebar */
#sidebar-nav-col {
    flex: 0 0 200px !important;
    width: 200px !important;
    min-width: 180px !important;
    max-width: 220px !important;
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 16px !important;
    padding: 16px 12px !important;
    backdrop-filter: blur(16px) !important;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5) !important;
}

.nav-header-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin: 8px 0 6px 4px;
}

/* Navigation Buttons */
.nav-btn {
    background: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #cbd5e1 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    border-radius: 10px !important;
    padding: 8px 10px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    display: flex !important;
    align-items: center !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    margin-bottom: 6px !important;
    box-shadow: none !important;
}

.nav-btn:hover {
    background: rgba(99, 102, 241, 0.18) !important;
    border-color: rgba(99, 102, 241, 0.45) !important;
    color: #ffffff !important;
    transform: translateX(3px) !important;
}

.nav-btn-active {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.85) 0%, rgba(6, 182, 212, 0.85) 100%) !important;
    border-color: #818cf8 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 14px var(--primary-glow) !important;
}

/* Engine Mode Radio in Narrow Sidebar */
#engine-selector .wrap, #engine-selector label {
    font-size: 0.75rem !important;
    line-height: 1.3 !important;
    padding: 6px 8px !important;
    background: rgba(30, 41, 59, 0.45) !important;
    border-radius: 8px !important;
    margin-bottom: 4px !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
}

#engine-selector label span {
    font-size: 0.75rem !important;
    color: #cbd5e1 !important;
}

/* Chatbot Container - Large, Spacious Main Column */
#chatbot-main-col {
    flex: 1 1 0% !important;
    min-width: 0 !important;
    width: 100% !important;
    background: rgba(15, 23, 42, 0.7) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 16px !important;
    padding: 20px 24px !important;
    backdrop-filter: blur(16px) !important;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5) !important;
}

/* Chatbot Window */
#main-chatbot {
    border-radius: 14px !important;
    background: rgba(11, 17, 33, 0.85) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.35) !important;
    backdrop-filter: blur(16px) !important;
}

/* User Message: Distinct Right-Aligned Bubble */
#main-chatbot [data-testid="user"],
#main-chatbot .message.user,
#main-chatbot .user {
    background: linear-gradient(135deg, rgba(79, 70, 229, 0.35) 0%, rgba(99, 102, 241, 0.45) 100%) !important;
    border: 1px solid rgba(129, 140, 248, 0.4) !important;
    border-radius: 16px 16px 4px 16px !important;
    color: #ffffff !important;
    padding: 12px 18px !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    line-height: 1.55 !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2) !important;
}

/* Assistant Message: Left-Aligned, Clean Slate, Comfortable Reading */
#main-chatbot [data-testid="bot"],
#main-chatbot .message.bot,
#main-chatbot .bot {
    background: rgba(20, 28, 48, 0.9) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-left: 4px solid #6366f1 !important;
    border-radius: 4px 16px 16px 16px !important;
    color: #f1f5f9 !important;
    padding: 18px 22px !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
}

/* Markdown Headings inside Assistant Answers */
#main-chatbot .prose h1, #main-chatbot h1,
#main-chatbot .prose h2, #main-chatbot h2,
#main-chatbot .prose h3, #main-chatbot h3,
#main-chatbot .prose h4, #main-chatbot h4 {
    color: #f8fafc !important;
    font-weight: 700 !important;
    margin-top: 1.1em !important;
    margin-bottom: 0.45em !important;
    line-height: 1.35 !important;
}

#main-chatbot h1 { font-size: 1.35rem !important; border-bottom: 1px solid rgba(255, 255, 255, 0.1); padding-bottom: 6px; }
#main-chatbot h2 { font-size: 1.2rem !important; }
#main-chatbot h3 { font-size: 1.05rem !important; color: #a5b4fc !important; }
#main-chatbot h4 { font-size: 0.95rem !important; color: #94a3b8 !important; }

/* Paragraphs & Spacing */
#main-chatbot .prose p, #main-chatbot p {
    color: #e2e8f0 !important;
    margin-top: 0 !important;
    margin-bottom: 0.9em !important;
    line-height: 1.7 !important;
    font-size: 0.95rem !important;
}

#main-chatbot .prose p:last-child, #main-chatbot p:last-child {
    margin-bottom: 0 !important;
}

/* Bold / Strong Emphasis */
#main-chatbot strong, #main-chatbot b {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Lists: Bulleted and Numbered */
#main-chatbot .prose ul, #main-chatbot ul {
    list-style-type: disc !important;
    padding-left: 1.6em !important;
    margin-top: 0.5em !important;
    margin-bottom: 0.9em !important;
}

#main-chatbot .prose ol, #main-chatbot ol {
    list-style-type: decimal !important;
    padding-left: 1.6em !important;
    margin-top: 0.5em !important;
    margin-bottom: 0.9em !important;
}

#main-chatbot li {
    margin-bottom: 0.4em !important;
    line-height: 1.6 !important;
    color: #e2e8f0 !important;
}

#main-chatbot li::marker {
    color: #818cf8 !important;
}

/* Inline Code & Code Blocks */
#main-chatbot code {
    font-family: 'JetBrains Mono', Consolas, Monaco, monospace !important;
    font-size: 0.88rem !important;
    background: rgba(0, 0, 0, 0.45) !important;
    color: #38bdf8 !important;
    padding: 2px 7px !important;
    border-radius: 5px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
}

#main-chatbot pre {
    background: rgba(8, 12, 24, 0.95) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
    margin: 1em 0 !important;
    overflow-x: auto !important;
}

#main-chatbot pre code {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    color: #f1f5f9 !important;
    font-size: 0.86rem !important;
    line-height: 1.55 !important;
}

/* Mathematical Notation (KaTeX) */
#main-chatbot .katex, #main-chatbot .katex-display {
    color: #facc15 !important;
    font-size: 1.05em !important;
}

#main-chatbot .katex-display {
    margin: 1em 0 !important;
    padding: 8px 12px !important;
    background: rgba(0, 0, 0, 0.25) !important;
    border-radius: 8px !important;
    border-left: 3px solid #facc15 !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
}

/* Blockquotes */
#main-chatbot blockquote {
    border-left: 3px solid #06b6d4 !important;
    background: rgba(6, 182, 212, 0.08) !important;
    padding: 8px 16px !important;
    border-radius: 0 8px 8px 0 !important;
    margin: 1em 0 !important;
    color: #cbd5e1 !important;
    font-style: italic !important;
}

/* Tables inside Markdown */
#main-chatbot table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 1em 0 !important;
    font-size: 0.9rem !important;
}

#main-chatbot th {
    background: rgba(30, 41, 59, 0.8) !important;
    color: #94a3b8 !important;
    padding: 8px 12px !important;
    border-bottom: 2px solid rgba(99, 102, 241, 0.3) !important;
    text-align: left !important;
}

#main-chatbot td {
    padding: 8px 12px !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
    color: #e2e8f0 !important;
}

#main-chatbot hr {
    border: none !important;
    border-top: 1px solid rgba(255, 255, 255, 0.12) !important;
    margin: 16px 0 !important;
}

/* Expandable Detail Section (Opens smoothly below chat when selected) */
#expandable-detail-section {
    background: rgba(15, 23, 42, 0.95) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 14px !important;
    padding: 18px 22px !important;
    margin-top: 18px !important;
    backdrop-filter: blur(20px) !important;
    box-shadow: 0 12px 36px -8px rgba(0, 0, 0, 0.6) !important;
    max-height: 560px !important;
    overflow-y: auto !important;
    animation: fadeInDown 0.25s cubic-bezier(0.16, 1, 0.3, 1);
}

#detail-panel-header, #detail-section-header {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding-bottom: 8px !important;
    margin-bottom: 14px !important;
}

@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.close-panel-btn {
    background: rgba(244, 63, 94, 0.15) !important;
    border: 1px solid rgba(244, 63, 94, 0.4) !important;
    color: #fda4af !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    border-radius: 8px !important;
    padding: 4px 10px !important;
    transition: all 0.2s ease !important;
}

.close-panel-btn:hover {
    background: rgba(244, 63, 94, 0.3) !important;
    color: #ffffff !important;
}

/* Cards & Badges inside Detail Sections */
.info-card {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 12px;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    margin-bottom: 14px;
}

.metric-pill {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 10px 12px;
}

.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-value {
    font-size: 1.15rem;
    font-weight: 700;
    margin-top: 2px;
}

.trajectory-step {
    background: rgba(15, 23, 42, 0.8) !important;
    border-left: 4px solid var(--accent-purple) !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    margin-bottom: 10px !important;
    border: 1px solid var(--border-subtle);
}

.tool-badge {
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 5px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.tool-search { background: rgba(99, 102, 241, 0.25); color: #a5b4fc; border: 1px solid #6366f1; }
.tool-calc { background: rgba(16, 185, 129, 0.25); color: #6ee7b7; border: 1px solid #10b981; }
.tool-quote { background: rgba(245, 158, 11, 0.25); color: #fcd34d; border: 1px solid #f59e0b; }
.tool-msg { background: rgba(244, 63, 94, 0.25); color: #fda4af; border: 1px solid #f43f5e; }
.tool-toc { background: rgba(6, 182, 212, 0.25); color: #67e8f9; border: 1px solid #06b6d4; }

.evidence-card {
    background: rgba(15, 23, 42, 0.7) !important;
    border-left: 4px solid #6366f1 !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    margin-bottom: 10px !important;
    border: 1px solid var(--border-subtle);
}

.approval-card {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(220, 38, 38, 0.15) 100%);
    border: 1px solid rgba(245, 158, 11, 0.5);
    border-radius: 12px;
    padding: 14px;
    margin: 10px 0;
}

button.primary-btn {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px var(--primary-glow) !important;
}

button.primary-btn:hover {
    background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
    transform: translateY(-1px);
}

button.approval-btn {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
}

/* Styled Table for Chapters and Evaluations */
.glass-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    margin-top: 8px;
}
.glass-table th {
    text-align: left;
    padding: 8px 10px;
    background: rgba(30, 41, 59, 0.8);
    color: #94a3b8;
    font-weight: 600;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}
.glass-table td {
    padding: 7px 10px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    color: #e2e8f0;
}
.glass-table tr:hover td {
    background: rgba(99, 102, 241, 0.08);
}
"""

# Titles for each selectable section
SECTION_TITLES = {
    "chat": "",
    "doc": "📚 Document / Corpus Information",
    "retrieval": "🔍 Retrieval Details & Agent Trajectory",
    "sources": "📑 Sources, Evidence & Citations",
    "eval": "📊 Evaluation Benchmarks & Metrics",
    "status": "⚙️ System Status, Budget & Health",
}


def format_trajectory_html(
    trajectory_data: Any,
    stop_reason: Optional[str] = None,
    total_duration_ms: Optional[float] = None
) -> str:
    """
    Formats the complete agent execution trace into rich, readable cards:
    - Summary banner showing total steps, total execution latency, and stop reason
    - Step-by-step cards detailing step sequence, tool/action, inputs, observations, statuses, and timings
    - Distinct visual styling for success, warning, pending approval, and failure states
    - Scrollable viewport container for long traces
    """
    if isinstance(trajectory_data, dict):
        trajectory = trajectory_data.get("trajectory", [])
        stop_reason = trajectory_data.get("stop_reason", stop_reason)
        total_duration_ms = trajectory_data.get("duration_ms", total_duration_ms)
        tool_count = trajectory_data.get("tool_calls_count", len(trajectory_data.get("tool_sequence", [])))
    elif isinstance(trajectory_data, list):
        trajectory = trajectory_data
        tool_count = sum(len(s.get("tool_calls", [])) for s in trajectory)
    else:
        trajectory = []
        tool_count = 0

    if not trajectory:
        return "<p style='color: #94a3b8; font-style: italic;'>No agent tool trajectory recorded for this run.</p>"

    # Format Stop Reason
    stop_labels = {
        "completed_task": ("Task Completed Successfully", "#10b981", "rgba(16,185,129,0.15)", "rgba(16,185,129,0.35)"),
        "direct_answer": ("Direct Response (No Tools Needed)", "#a5b4fc", "rgba(99,102,241,0.15)", "rgba(99,102,241,0.35)"),
        "max_step_limit_reached": ("Maximum Step Limit Reached", "#f59e0b", "rgba(245,158,11,0.15)", "rgba(245,158,11,0.35)"),
        "budget_exceeded": ("Budget Exceeded (Halted Safely)", "#f59e0b", "rgba(245,158,11,0.15)", "rgba(245,158,11,0.35)"),
        "repeated_calls_halted": ("Repeated-Call Prevention (Halted)", "#f43f5e", "rgba(244,63,94,0.15)", "rgba(244,63,94,0.35)"),
        "model_error": ("Model Communication Error", "#f43f5e", "rgba(244,63,94,0.15)", "rgba(244,63,94,0.35)"),
    }
    s_label, s_color, s_bg, s_border = stop_labels.get(
        stop_reason or "",
        ("Completed", "#6ee7b7", "rgba(16,185,129,0.15)", "rgba(16,185,129,0.35)")
    )

    html_cards = []
    card_index = 0

    for step_data in trajectory:
        # 1. Render Pre-Execution Planner Card
        if step_data.get("type") == "planner":
            plan_items = step_data.get("plan", [])
            plan_html = "".join([f"<div style='padding: 2px 0;'>{p}</div>" for p in plan_items])
            card = f"""
<div class="trajectory-step" style="border-left: 4px solid #818cf8; background: rgba(99, 102, 241, 0.08); margin-bottom: 10px;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <span style="font-weight: 700; color: #a5b4fc; font-size: 0.85rem;">📋 Pre-Execution Plan</span>
        <span style="font-size: 0.72rem; background: rgba(99, 102, 241, 0.2); color: #c7d2fe; padding: 2px 8px; border-radius: 9999px;">Planner Activated</span>
    </div>
    <div style="font-size: 0.82rem; color: #e2e8f0; line-height: 1.5; font-family: 'JetBrains Mono', monospace;">
        {plan_html}
    </div>
</div>
"""
            html_cards.append(card)
            continue

        # 2. Render Single-Pass Self-Critique Card
        if step_data.get("type") == "self_critique":
            critique = step_data.get("critique", {})
            c_action = critique.get("action", "verified")
            c_status = critique.get("status", "verified")
            card = f"""
<div class="trajectory-step" style="border-left: 4px solid #10b981; background: rgba(16, 185, 129, 0.08); margin-bottom: 10px;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <span style="font-weight: 700; color: #6ee7b7; font-size: 0.85rem;">🔎 Single-Pass Self-Critique</span>
        <span style="font-size: 0.72rem; background: rgba(16, 185, 129, 0.2); color: #a7f3d0; padding: 2px 8px; border-radius: 9999px;">{c_status.title()}</span>
    </div>
    <div style="font-size: 0.82rem; color: #e2e8f0;">
        <strong>Verification Action:</strong> {c_action}
    </div>
</div>
"""
            html_cards.append(card)
            continue

        step_lat = step_data.get("step_latency_ms", 0.0)
        step_events = step_data.get("events", [])
        tool_calls = step_data.get("tool_calls", [])

        if tool_calls:
            for tc in tool_calls:
                card_index += 1
                tool_name = tc.get("tool", "unknown")
                args = tc.get("arguments", {})
                full_res = tc.get("full_result", {})
                status_raw = str(tc.get("status", "success")).lower()
                tc_lat = tc.get("duration_ms", step_lat)

                # Status badge styling
                if status_raw == "pending_approval":
                    status_badge = "<span style='background: rgba(6,182,212,0.2); color: #67e8f9; border: 1px solid rgba(6,182,212,0.4); padding: 2px 8px; border-radius: 6px; font-size: 0.72rem; font-weight: 700;'>🔵 Pending Human Approval</span>"
                    step_border = "border-left: 4px solid #06b6d4 !important;"
                elif status_raw == "cache_hit" or tc.get("cache_hit"):
                    status_badge = "<span style='background: rgba(6,182,212,0.2); color: #67e8f9; border: 1px solid rgba(6,182,212,0.4); padding: 2px 8px; border-radius: 6px; font-size: 0.72rem; font-weight: 700;'>⚡ Cache Hit</span>"
                    step_border = "border-left: 4px solid #06b6d4 !important;"
                elif status_raw in ("warning", "repeated_call_detected"):
                    status_badge = "<span style='background: rgba(245,158,11,0.2); color: #fcd34d; border: 1px solid rgba(245,158,11,0.4); padding: 2px 8px; border-radius: 6px; font-size: 0.72rem; font-weight: 700;'>🟡 Warning / Repeated Call</span>"
                    step_border = "border-left: 4px solid #f59e0b !important;"
                elif status_raw in ("error", "failure"):
                    status_badge = "<span style='background: rgba(244,63,94,0.2); color: #fda4af; border: 1px solid rgba(244,63,94,0.4); padding: 2px 8px; border-radius: 6px; font-size: 0.72rem; font-weight: 700;'>🔴 Error / Failed</span>"
                    step_border = "border-left: 4px solid #f43f5e !important;"
                elif status_raw == "stopped":
                    status_badge = "<span style='background: rgba(148,163,184,0.2); color: #cbd5e1; border: 1px solid rgba(148,163,184,0.4); padding: 2px 8px; border-radius: 6px; font-size: 0.72rem; font-weight: 700;'>⚪ Stopped</span>"
                    step_border = "border-left: 4px solid #94a3b8 !important;"
                else:
                    status_badge = "<span style='background: rgba(16,185,129,0.2); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.4); padding: 2px 8px; border-radius: 6px; font-size: 0.72rem; font-weight: 700;'>🟢 Success</span>"
                    step_border = "border-left: 4px solid #10b981 !important;"

                # Select tool badge style
                if tool_name == "search_corpus":
                    badge_class = "tool-search"
                    icon = "🔍"
                elif tool_name == "calculator":
                    badge_class = "tool-calc"
                    icon = "🧮"
                elif tool_name == "quote_exact":
                    badge_class = "tool-quote"
                    icon = "📖"
                elif tool_name == "send_message":
                    badge_class = "tool-msg"
                    icon = "💬"
                else:
                    badge_class = "tool-toc"
                    icon = "📑"

                args_str = json.dumps(args, indent=2, ensure_ascii=False)

                # Format clean observation snippet
                if status_raw in ("error", "failure") or full_res.get("status") in ("error", "failure"):
                    err_msg = full_res.get("error") or tc.get("result_summary") or full_res.get("formatted") or "Execution failed"
                    obs_summary = f"<span style='color: #fda4af;'>Error: {err_msg}</span>"
                elif tool_name == "search_corpus":
                    matches = full_res.get("matches", [])
                    pages = sorted(list(set(m.get("page") for m in matches if m.get("page"))))
                    top_score = matches[0].get("score", 0.0) if matches else 0.0
                    obs_summary = f"Retrieved <strong>{len(matches)} matching chunks</strong> across pages {pages} (Top relevance: {top_score:.3f})."
                    if full_res.get("status") == "no_matches":
                        obs_summary = f"<span style='color: #fca5a5;'>No matching passages found for query '{args.get('query')}'.</span>"
                elif tool_name == "calculator":
                    calc_res = full_res.get('result', full_res.get('value', tc.get('result_summary')))
                    obs_summary = f"Evaluated expression <code>{args.get('expression')}</code> → <strong style='color: #6ee7b7; font-size: 0.92rem;'>{calc_res}</strong>"
                elif tool_name == "quote_exact":
                    p = full_res.get("page", "?")
                    char_count = full_res.get("char_count", 0)
                    verified = full_res.get("passage_status", "")
                    obs_summary = f"Loaded text of Page {p} ({char_count} chars). {verified}"
                elif tool_name == "send_message":
                    obs_summary = full_res.get("result", tc.get("result_summary", "Message action processed."))
                else:
                    obs_summary = str(full_res.get("formatted", tc.get("result_summary", str(full_res))))[:300]

                card = f"""
<div class="trajectory-step" style="{step_border}">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 6px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-weight: 700; color: #94a3b8; font-size: 0.82rem;">Step {card_index}</span>
            <span class="tool-badge {badge_class}">{icon} {tool_name}</span>
            <span style="background: rgba(255,255,255,0.06); color: #94a3b8; padding: 1px 6px; border-radius: 4px; font-size: 0.72rem;">🔧 Tool Call</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            {status_badge}
            <span style="font-size: 0.75rem; color: #38bdf8; font-family: 'JetBrains Mono', monospace;">⏱️ {tc_lat:.1f} ms</span>
        </div>
    </div>
    <div style="background: rgba(0,0,0,0.35); border-radius: 6px; padding: 6px 10px; margin-bottom: 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; color: #cbd5e1; max-height: 90px; overflow-y: auto;">
        <strong style="color: #94a3b8;">Input Parameters:</strong> {args_str}
    </div>
    <div style="background: rgba(15,23,42,0.5); border-radius: 6px; padding: 7px 10px; font-size: 0.82rem; color: #e2e8f0; line-height: 1.45;">
        <strong style="color: #a5b4fc;">Observation:</strong> {obs_summary}
    </div>
</div>
"""
                html_cards.append(card)

        else:
            # Direct answer / final synthesis step
            card_index += 1
            final_snip = step_data.get("final_content", "")
            if not final_snip and isinstance(trajectory_data, dict):
                final_snip = trajectory_data.get("answer", "")

            is_direct = (card_index == 1)
            step_badge_label = "Final Synthesis / Direct Response" if is_direct else "Synthesized Final Answer"
            action_tag = "direct_answer" if is_direct else "final_synthesis"
            badge_icon = "🧠"

            status_badge = "<span style='background: rgba(16,185,129,0.2); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.4); padding: 2px 8px; border-radius: 6px; font-size: 0.72rem; font-weight: 700;'>🟢 Completed</span>"

            card = f"""
<div class="trajectory-step" style="border-left: 4px solid #6366f1 !important;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 6px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-weight: 700; color: #94a3b8; font-size: 0.82rem;">Step {card_index}</span>
            <span class="tool-badge tool-calc">{badge_icon} {step_badge_label}</span>
            <span style="background: rgba(255,255,255,0.06); color: #94a3b8; padding: 1px 6px; border-radius: 4px; font-size: 0.72rem;">Action: {action_tag}</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            {status_badge}
            <span style="font-size: 0.75rem; color: #38bdf8; font-family: 'JetBrains Mono', monospace;">⏱️ {step_lat:.1f} ms</span>
        </div>
    </div>
    <div style="background: rgba(15,23,42,0.5); border-radius: 6px; padding: 8px 10px; font-size: 0.82rem; color: #e2e8f0; line-height: 1.45;">
        <strong style="color: #a5b4fc;">Synthesized Final Answer:</strong> {final_snip[:400] + ('...' if len(final_snip) > 400 else '')}
    </div>
</div>
"""
            html_cards.append(card)

    tot_steps = len(html_cards)
    tot_time_str = f"{total_duration_ms:,.1f} ms" if total_duration_ms else f"{sum(s.get('step_latency_ms', 0) for s in trajectory):,.1f} ms"

    summary_html = f"""
<div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 10px; padding: 12px 16px; margin-bottom: 12px;">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">⚡ Agent Execution Telemetry</span>
            <span style="background: rgba(99, 102, 241, 0.2); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.4); padding: 2px 8px; border-radius: 6px; font-size: 0.74rem; font-weight: 600;">{tot_steps} Total Steps</span>
        </div>
        <div style="font-size: 0.8rem; color: #94a3b8;">
            Total Duration: <strong style="color: #38bdf8;">{tot_time_str}</strong> | Tool Calls: <strong style="color: #c084fc;">{tool_count}</strong>
        </div>
    </div>
    <div style="margin-top: 8px; font-size: 0.82rem; color: #cbd5e1; display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
        <span style="color: #94a3b8;">Stop Reason:</span>
        <span style="background: {s_bg}; color: {s_color}; border: 1px solid {s_border}; padding: 2px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.76rem;">
            {s_label}
        </span>
    </div>
</div>
"""

    if not html_cards:
        return "<p style='color: #94a3b8; font-style: italic;'>Agent answered directly without requiring external tool execution.</p>"

    cards_block = "\n".join(html_cards)
    return f"""
{summary_html}
<div class="trace-scroll-container" style="max-height: 520px; overflow-y: auto; padding-right: 4px;">
{cards_block}
</div>
"""



def format_pipeline_trace_html(result: dict) -> str:
    """Formats fixed RAG pipeline execution trace into visual cards."""
    r_ms = result.get("retrieval_time_ms", 0.0)
    rerank_ms = result.get("rerank_time_ms", 0.0)
    g_ms = result.get("generation_time_ms", 0.0)
    tot_ms = result.get("total_time_ms", 0.0)
    sources_count = len(result.get("sources", []))
    cited_pages = result.get("cited_pages", [])

    cards = []

    # 1. Query Rewriting Card (if triggered)
    rw_info = result.get("rewritten_query_info", {})
    if rw_info.get("is_rewritten"):
        cards.append(f"""
<div class="trajectory-step" style="border-left: 4px solid #06b6d4; background: rgba(6, 182, 212, 0.08);">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
        <span style="font-weight: 700; color: #67e8f9; font-size: 0.8rem;">💡 Query Rewriter (Chat Memory Context)</span>
        <span style="font-size: 0.72rem; background: rgba(6, 182, 212, 0.2); color: #a5f3fc; padding: 2px 7px; border-radius: 9999px;">Coreference Resolved</span>
    </div>
    <div style="font-size: 0.82rem; color: #e2e8f0; line-height: 1.45;">
        <div style="color: #94a3b8; font-size: 0.78rem;">Original: <em>"{rw_info.get('original_query')}"</em></div>
        <div>Standalone Query: <strong style="color: #38bdf8;">"{rw_info.get('query')}"</strong></div>
    </div>
</div>
""")

    # 2. Stage 1: Hybrid Retrieval
    cards.append(f"""
<div class="trajectory-step">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <div>
            <span style="font-weight: 700; color: #94a3b8; font-size: 0.8rem; margin-right: 8px;">Stage 1</span>
            <span class="tool-badge tool-search">🔍 Hybrid Retrieval (FAISS + BM25)</span>
        </div>
        <span style="font-size: 0.75rem; color: #38bdf8;">{r_ms:.1f} ms</span>
    </div>
    <div style="font-size: 0.82rem; color: #e2e8f0; line-height: 1.4;">
        Retrieved candidate pool using Dense FAISS (65%) and Okapi BM25 (35%).
    </div>
</div>
""")

    # 3. Stage 2: Cross-Encoder Reranking
    cards.append(f"""
<div class="trajectory-step" style="border-left: 4px solid #8b5cf6;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <div>
            <span style="font-weight: 700; color: #94a3b8; font-size: 0.8rem; margin-right: 8px;">Stage 2</span>
            <span class="tool-badge" style="background: rgba(139, 92, 246, 0.2); color: #c4b5fd; border: 1px solid rgba(139, 92, 246, 0.4);">⚡ Cross-Encoder Reranking</span>
        </div>
        <span style="font-size: 0.75rem; color: #c4b5fd;">{rerank_ms:.1f} ms</span>
    </div>
    <div style="font-size: 0.82rem; color: #e2e8f0; line-height: 1.4;">
        Reranked candidate chunks to top <strong>{sources_count} best excerpts</strong> for final generation.
    </div>
</div>
""")

    # 4. Confidence Indicator Card
    conf = result.get("confidence", {})
    if conf:
        cards.append(f"""
<div class="trajectory-step" style="border-left: 4px solid #10b981;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <div>
            <span style="font-weight: 700; color: #94a3b8; font-size: 0.8rem; margin-right: 8px;">Indicator</span>
            <span style="font-size: 0.78rem; font-weight: 700; color: #6ee7b7;">{conf.get('badge', '🟢 High')}</span>
        </div>
        <span style="font-size: 0.75rem; color: #94a3b8;">Score: {conf.get('score', 0.0):.2f}</span>
    </div>
    <div style="font-size: 0.80rem; color: #cbd5e1; line-height: 1.4;">
        {conf.get('explanation', '')}
        <div style="color: #64748b; font-size: 0.72rem; margin-top: 3px;"><em>{conf.get('disclaimer', '')}</em></div>
    </div>
</div>
""")

    # 5. Stage 3: Grounded Generation
    cards.append(f"""
<div class="trajectory-step">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <div>
            <span style="font-weight: 700; color: #94a3b8; font-size: 0.8rem; margin-right: 8px;">Stage 3</span>
            <span class="tool-badge tool-quote">🤖 Grounded Generation &amp; Citation</span>
        </div>
        <span style="font-size: 0.75rem; color: #38bdf8;">{g_ms:.1f} ms</span>
    </div>
    <div style="font-size: 0.82rem; color: #e2e8f0; line-height: 1.4;">
        Synthesized answer with <code>gpt-4o-mini</code>. Cited physical pages: <strong>{cited_pages or 'None'}</strong>.
    </div>
</div>
""")

    # 6. Faithfulness Check Card
    faith = result.get("faithfulness", {})
    if faith:
        unsupported_claims = faith.get("unsupported_claims", [])
        unsupported_cits = faith.get("unsupported_citations", [])
        retry_flag = result.get("retry_occurred", False)

        issues = []
        if unsupported_cits:
            issues.append(f"<div style='color: #fca5a5; font-size: 0.74rem; margin-top: 3px;'>⚠️ Unretrieved Citation(s): [Page {', '.join(map(str, unsupported_cits))}]</div>")
        if unsupported_claims and faith.get("status") != "Faithful":
            issues.append(f"<div style='color: #fdba74; font-size: 0.74rem; margin-top: 2px;'>• {unsupported_claims[0]}</div>")

        retry_html = ""
        if retry_flag:
            retry_html = "<div style='color: #6ee7b7; font-size: 0.74rem; margin-top: 4px; font-weight: 600;'>🔄 Automated 1-step retrieval adjustment executed.</div>"

        cards.append(f"""
<div class="trajectory-step" style="border-left: 4px solid #38bdf8;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
        <span style="font-weight: 700; color: #7dd3fc; font-size: 0.8rem;">🛡️ Faithfulness Check</span>
        <span style="font-size: 0.75rem; font-weight: 700; color: #38bdf8;">{faith.get('badge', '✅ Faithful')}</span>
    </div>
    <div style="font-size: 0.80rem; color: #cbd5e1; line-height: 1.4;">
        {faith.get('explanation', '')}
        {''.join(issues)}
        {retry_html}
    </div>
</div>
""")

    cards.append(f"""
<div style="margin-top: 8px; padding: 8px 12px; background: rgba(30,41,59,0.5); border-radius: 8px; font-size: 0.8rem; color: #94a3b8;">
    ⏱️ Total Pipeline Latency: <strong style="color: #38bdf8;">{tot_ms:.1f} ms</strong>
</div>
""")

    return "\n".join(cards)


def format_evidence_markdown(sources: list) -> str:
    """Formats retrieved sources into rich visual cards."""
    if not sources:
        return "<p style='color: #94a3b8; font-style: italic;'>No direct RAG sources retrieved for this run. Submit a query to inspect citations.</p>"

    md_blocks = []
    for i, s in enumerate(sources, 1):
        doc = s.get("source") or s.get("document") or "crystal1 (1).pdf"
        page = s.get("page", "?")
        comb = s.get("combined_score", s.get("score", 0.0))
        dense = s.get("dense_score", 0.0)
        bm25 = s.get("bm25_score", 0.0)

        raw_snippet = (
            s.get("text_snippet")
            or s.get("snippet")
            or s.get("text")
            or s.get("content")
            or s.get("passage")
            or ""
        )
        snippet = str(raw_snippet).strip()
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."

        score_detail = f"Score: {comb:.2f}"
        if dense > 0 or bm25 > 0:
            score_detail += f" (D: {dense:.2f} | B: {bm25:.2f})"

        if snippet:
            content_html = f'"{snippet}"'
            content_style = "font-size: 0.82rem; color: #cbd5e1; line-height: 1.45; font-style: italic; background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 6px;"
        else:
            content_html = '<span style="color: #94a3b8; font-style: italic;">Evidence snippet unavailable for this source chunk.</span>'
            content_style = "font-size: 0.82rem; color: #94a3b8; line-height: 1.45; background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 6px;"

        card = f"""
<div class="evidence-card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <div>
            <span style="font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 6px; background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.3);">📄 Page {page}</span>
            <span style="font-size: 0.8rem; color: #94a3b8; margin-left: 8px;">{doc}</span>
        </div>
        <div>
            <span style="font-size: 0.72rem; font-weight: 600; padding: 2px 7px; border-radius: 5px; background: rgba(99, 102, 241, 0.2); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.3);">{score_detail}</span>
        </div>
    </div>
    <div style="{content_style}">
        {content_html}
    </div>
</div>
"""
        md_blocks.append(card)

    return "\n".join(md_blocks)


def format_agent_evidence_html(result: dict) -> str:
    """Extracts retrieved source matches from agent run trajectory or cited pages."""
    trajectory = result.get("trajectory", [])
    matches = []
    seen_keys = set()

    for step in trajectory:
        for tc in step.get("tool_calls", []):
            tool_name = tc.get("tool")
            full_res = tc.get("full_result", {})

            if tool_name == "search_corpus" and isinstance(full_res, dict):
                m_list = full_res.get("matches", [])
                for m in m_list:
                    doc = m.get("source", "crystal1 (1).pdf")
                    page = m.get("page", "?")
                    snip = m.get("text_snippet") or m.get("snippet") or m.get("text") or ""
                    key = (doc, page, snip[:40])
                    if key not in seen_keys:
                        seen_keys.add(key)
                        matches.append(m)

            elif tool_name == "quote_exact" and isinstance(full_res, dict):
                if full_res.get("status") == "success" and full_res.get("text"):
                    doc = full_res.get("document", "crystal1 (1).pdf")
                    page = full_res.get("page", "?")
                    text_val = full_res.get("text", "").strip()
                    snip = text_val[:280] + ("..." if len(text_val) > 280 else "")
                    key = (doc, page, snip[:40])
                    if key not in seen_keys:
                        seen_keys.add(key)
                        matches.append({
                            "source": doc,
                            "page": page,
                            "score": 1.0,
                            "snippet": snip,
                            "text_snippet": snip,
                            "text": text_val
                        })

    if matches:
        return format_evidence_markdown(matches)

    cited_pages = result.get("cited_pages", [])
    if cited_pages:
        page_tags = " ".join([f"<span style='background: rgba(16,185,129,0.2); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.3); font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 6px; margin-right: 6px;'>Page {p}</span>" for p in cited_pages])
        return f"""
<div class="evidence-card">
    <div style="margin-bottom: 6px; color: #94a3b8; font-size: 0.82rem;">Agent cited pages verified from textbook corpus:</div>
    <div>{page_tags}</div>
</div>
"""

    return "<p style='color: #94a3b8; font-style: italic;'>Agent answered directly from reasoning or arithmetic tools without external chunk retrieval.</p>"


def format_stats_markdown(result: dict, mode: str) -> str:
    """Formats latency and cost metrics into clean summary HTML."""
    summary = cost_tracker.get_summary()
    tot_cost = summary.get("total_cost_usd", 0.0)
    budget_pct = summary.get("budget_used_percent", 0.0)
    q_cost = result.get("cost_usd", 0.0)
    tot_ms = result.get("duration_ms", result.get("total_time_ms", 0.0))
    tokens = result.get("total_tokens", 0)
    tool_count = result.get("tool_calls_count", 0)

    tool_stat = f"<strong style='color: #c084fc; font-size: 1rem;'>{tool_count} calls</strong>" if mode == "agent" else "<strong style='color: #6ee7b7; font-size: 1rem;'>Hybrid Pipeline</strong>"
    tool_label = "🛠️ Tool Actions" if mode == "agent" else "⚙️ Mode"

    return f"""
<div class="metric-grid">
    <div class="metric-pill">
        <div class="metric-label">⏱️ Total Latency</div>
        <div class="metric-value" style="color: #38bdf8;">{tot_ms:.1f} ms</div>
    </div>
    <div class="metric-pill">
        <div class="metric-label">{tool_label}</div>
        <div class="metric-value">{tool_stat}</div>
    </div>
    <div class="metric-pill">
        <div class="metric-label">🪙 Run Tokens / Cost</div>
        <div class="metric-value" style="color: #facc15;">{tokens} tok / ${q_cost:.5f}</div>
    </div>
    <div class="metric-pill">
        <div class="metric-label">💳 Total Spend ($5 Budget)</div>
        <div class="metric-value" style="color: #a78bfa;">${tot_cost:.4f} ({budget_pct:.2f}%)</div>
    </div>
</div>
"""


def format_approval_card(pending: dict) -> str:
    """Formats Human-in-the-Loop approval prompt banner."""
    channel = pending.get("channel", "#general")
    message = pending.get("message", "")
    return f"""
<div class="approval-card">
    <div style="display: flex; align-items: center; margin-bottom: 6px;">
        <span style="font-size: 1.2rem; margin-right: 8px;">⚠️</span>
        <strong style="color: #f59e0b; font-size: 0.95rem;">HUMAN-IN-THE-LOOP APPROVAL REQUIRED</strong>
    </div>
    <p style="font-size: 0.85rem; color: #cbd5e1; margin: 4px 0;">
        The agent has prepared an irreversible action to post to <strong>{channel}</strong>:
    </p>
    <div style="background: rgba(0,0,0,0.3); border-left: 3px solid #f59e0b; padding: 8px 12px; margin: 8px 0; border-radius: 4px; font-size: 0.85rem; color: #f8fafc; font-style: italic;">
        "{message}"
    </div>
    <p style="font-size: 0.8rem; color: #94a3b8; margin: 4px 0;">
        Click <strong>'Confirm &amp; Send'</strong> below to execute via MCP client / Webhook, or <strong>'Reject'</strong> to abort.
    </p>
</div>
"""


def load_audit_log_html() -> str:
    """Reads recent entries from sent_messages.log for system status audit trail."""
    log_path = ROOT / "data" / "sent_messages.log"
    if not log_path.exists():
        return "<p style='color: #94a3b8; font-style: italic;'>No external messages dispatched yet.</p>"

    entries = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except Exception as exc:
        return f"<p style='color: #fda4af;'>Failed to read audit log: {exc}</p>"

    if not entries:
        return "<p style='color: #94a3b8; font-style: italic;'>No external messages dispatched yet.</p>"

    rows = []
    for e in reversed(entries[-5:]):
        ts = e.get("timestamp", "")[:19].replace("T", " ")
        ch = e.get("channel", "")
        msg = e.get("message", "")
        if len(msg) > 50:
            msg = msg[:50] + "..."
        stat = e.get("status", "sent")
        rows.append(f"<tr><td>{ts}</td><td><code>{ch}</code></td><td>{msg}</td><td><span style='color: #6ee7b7;'>✅ {stat}</span></td></tr>")

    return f"""
<table class="glass-table">
    <thead>
        <tr><th>Timestamp (UTC)</th><th>Channel</th><th>Message</th><th>Status</th></tr>
    </thead>
    <tbody>
        {''.join(rows)}
    </tbody>
</table>
"""


def build_chapters_table_html() -> str:
    """Generates the 29-chapter table for Document / Corpus Information."""
    rows = []
    for c in CHAPTERS:
        rows.append(f"""
        <tr>
            <td style="font-weight: 700; color: #a5b4fc;">Ch {c['chapter']}</td>
            <td>{c['title']}</td>
            <td style="color: #6ee7b7;">Page {c['start_page']}</td>
            <td style="color: #94a3b8;">PDF {c['pdf_page']}</td>
        </tr>
        """)

    return f"""
<div style="max-height: 380px; overflow-y: auto; border: 1px solid var(--border-subtle); border-radius: 8px;">
    <table class="glass-table">
        <thead>
            <tr>
                <th>Chapter</th>
                <th>Title</th>
                <th>Book Page</th>
                <th>PDF Page</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
</div>
"""


def switch_section(target_sec: str, current_sec: str):
    """
    Controls expandable detail panel state.
    - If user clicks current section or 'chat', collapses panel.
    - If user clicks any other section, expands panel and shows ONLY that section.
    - Returns updates for state, detail panel, section title, 5 groups, and 6 nav buttons.
    """
    # Toggle behavior: clicking the currently active section closes it and returns to chat
    if target_sec == current_sec or target_sec == "chat":
        new_sec = "chat"
    else:
        new_sec = target_sec

    is_open = (new_sec != "chat")
    title = SECTION_TITLES.get(new_sec, "")

    def btn_update(btn_key):
        is_active = (new_sec == btn_key)
        return gr.update(
            variant="primary" if is_active else "secondary",
            elem_classes=["nav-btn", "nav-btn-active"] if is_active else ["nav-btn"]
        )

    return (
        new_sec,                                       # current_section_state
        gr.update(visible=is_open),                    # detail_panel
        f"### {title}",                                # panel_title_md
        gr.update(visible=(new_sec == "doc")),         # group_doc_info
        gr.update(visible=(new_sec == "retrieval")),   # group_retrieval
        gr.update(visible=(new_sec == "sources")),     # group_sources
        gr.update(visible=(new_sec == "eval")),        # group_eval
        gr.update(visible=(new_sec == "status")),      # group_status
        btn_update("chat"),                            # btn_nav_chat
        btn_update("doc"),                             # btn_nav_doc
        btn_update("retrieval"),                       # btn_nav_retrieval
        btn_update("sources"),                         # btn_nav_sources
        btn_update("eval"),                            # btn_nav_eval
        btn_update("status"),                          # btn_nav_status
    )


def format_chat_answer(answer_text: str, cited_pages: Optional[List[int]] = None) -> str:
    """
    Cleans and formats assistant response:
    - Preserves Markdown structure, headings, paragraphs, lists, math, and code
    - Keeps inline citations [Page X] directly beside statements when verified
    - Strips unsupported / unverified citations (such as page 303 unless verified)
    - Ensures a clearly formatted 'Sources cited' section at the end with verified pages
    """
    if not answer_text:
        return ""

    text = answer_text.strip()
    valid_pages = set(cited_pages or [])

    # If verified cited_pages are provided, validate inline citations [Page X]
    if valid_pages:
        def _filter_unverified(m):
            try:
                p_num = int(m.group(1))
                if p_num in valid_pages:
                    return f"[Page {p_num}]"
                return ""  # Omit unverified citation
            except (ValueError, TypeError):
                return m.group(0)

        text = re.sub(r"\[Page\s*(\d+)\]", _filter_unverified, text)
        text = re.sub(r"\[\s*\]", "", text)
        text = re.sub(r" +([.,;])", r"\1", text).strip()

    # Check if a sources section already exists in the text
    has_sources = bool(re.search(r"(?:sources\s*cited|sources\s*:)", text, re.IGNORECASE))

    if not has_sources and valid_pages:
        unique_pages = sorted(valid_pages)
        if unique_pages:
            sources_md = "\n".join([f"- [Page {p}]" for p in unique_pages])
            text = f"{text}\n\n---\n**Sources cited:**\n{sources_md}"

    return text


def format_agent_chat_response(result: dict) -> str:
    """
    Formats the agent response for the chat view:
    - Displays concise tool sequence badge header next to the answer
    - Embeds a collapsible <details> block containing the step-by-step execution trace
    - Formats grounded textbook answer and sources
    """
    raw_answer = result.get("answer", "")
    cited_pages = result.get("cited_pages", [])
    base_answer = format_chat_answer(raw_answer, cited_pages)

    tool_seq = result.get("tool_sequence", [])
    steps_count = result.get("steps_count", len(result.get("trajectory", [])))
    duration_ms = result.get("duration_ms", 0.0)
    stop_reason = result.get("stop_reason", "completed_task")

    tool_icons = {
        "search_corpus": "🔍 search_corpus",
        "calculator": "🧮 calculator",
        "quote_exact": "📖 quote_exact",
        "list_chapters": "📑 list_chapters",
        "send_message": "💬 send_message"
    }

    if not tool_seq:
        sequence_display = "*Direct Response (No external tools called)*"
    else:
        sequence_display = " → ".join([f"`{tool_icons.get(t, t)}`" for t in tool_seq])

    # Build collapsible trace summary with unique sequential step numbering
    trace_steps = []
    step_num = 0
    trajectory = result.get("trajectory", [])

    for step_item in trajectory:
        s_lat = step_item.get("step_latency_ms", 0.0)
        s_calls = step_item.get("tool_calls", [])
        if s_calls:
            for tc in s_calls:
                step_num += 1
                t_name = tc.get("tool", "tool")
                t_args = json.dumps(tc.get("arguments", {}), ensure_ascii=False)
                t_res = tc.get("result_summary", "")
                t_stat = tc.get("status", "success")
                tc_lat = tc.get("duration_ms", s_lat)
                icon = tool_icons.get(t_name, f"🔧 {t_name}")
                trace_steps.append(
                    f"- **Step {step_num}:** `[Tool Call]` {icon} (*{t_stat}*, {tc_lat:.1f}ms)\n"
                    f"  - **Input:** `{t_args}`\n"
                    f"  - **Observation:** {t_res}"
                )
        else:
            step_num += 1
            final_snip = step_item.get("final_content", "")[:160].strip()
            if not final_snip and raw_answer:
                final_snip = raw_answer[:160].strip()
            step_type = "[Direct Response (No Tools Needed)]" if step_num == 1 else "[Synthesized Final Answer]"
            icon = "🧠"
            trace_steps.append(
                f"- **Step {step_num}:** `{step_type}` {icon} ({s_lat:.1f}ms)\n"
                f"  - **Final Output:** {final_snip}..."
            )

    total_steps_count = len(trace_steps) if trace_steps else result.get("steps_count", len(trajectory))
    trace_content = "\n".join(trace_steps) if trace_steps else "Direct synthesis without external tool execution."

    meta_header = (
        f"> **⚡ Action Sequence:** {sequence_display}  \n"
        f"> *{total_steps_count} steps | {duration_ms:,.0f} ms | Stop reason: `{stop_reason}`*\n\n"
        f"<details style='margin-bottom: 12px; padding: 6px 10px; background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 6px;'>\n"
        f"<summary style='cursor: pointer; font-size: 0.82rem; color: #a5b4fc; font-weight: 600;'>🔍 View Full Execution Trace ({total_steps_count} steps)</summary>\n\n"
        f"{trace_content}\n\n"
        f"</details>\n\n"
    )

    return meta_header + base_answer


def process_query(user_query: str, execution_mode: str, chat_history: Optional[List[Dict[str, Any]]]):
    """
    Main query handler supporting Autonomous Agent and Fixed RAG modes.
    Appends conversation turns to Chatbot and updates all trace/evidence/stat views.
    """
    history = list(chat_history or [])

    if not user_query or not user_query.strip():
        return (
            history,
            "",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(value="", visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            {}
        )

    clean_query = user_query.strip()
    history.append({"role": "user", "content": clean_query})

    if "Agent" in execution_mode:
        # Run Autonomous Agent Loop
        result = agent.run(clean_query)
        answer_text = format_agent_chat_response(result)
        trajectory_html = format_trajectory_html(result)
        evidence_html = format_agent_evidence_html(result)
        stats_html = format_stats_markdown(result, "agent")

        history.append({"role": "assistant", "content": answer_text})

        pending = result.get("pending_approval")
        if pending:
            approval_html = format_approval_card(pending)
            return (
                history,
                "",
                trajectory_html,
                trajectory_html,
                evidence_html,
                stats_html,
                gr.update(value=approval_html, visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                pending
            )
        else:
            return (
                history,
                "",
                trajectory_html,
                trajectory_html,
                evidence_html,
                stats_html,
                gr.update(value="", visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                {}
            )
    else:
        # Run Fixed RAG Pipeline with chat memory for coreference resolution & follow-up questions
        result = pipeline.run(clean_query, chat_history=history[:-1], top_k=10)
        answer_text = format_chat_answer(result["answer"], result.get("cited_pages", []))
        evidence_html = format_evidence_markdown(result.get("sources", []))
        trajectory_html = format_pipeline_trace_html(result)
        stats_html = format_stats_markdown(result, "rag")

        history.append({"role": "assistant", "content": answer_text})
        return (
            history,
            "",
            trajectory_html,
            trajectory_html,
            evidence_html,
            stats_html,
            gr.update(value="", visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            {}
        )



def execute_approved_message(pending_state: Dict[str, Any], chat_history: Optional[List[Dict[str, Any]]]):
    """Executes message dispatch after explicit human approval and posts status to chat."""
    history = list(chat_history or [])
    if not pending_state or not pending_state.get("channel"):
        history.append({"role": "assistant", "content": "*No pending message action to approve.*"})
        return (
            history,
            gr.update(value="", visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            {}
        )

    channel = pending_state.get("channel", "general").lstrip("#")
    msg_text = pending_state.get("message", "")

    # Invoke send_message with approved=True
    res = send_message(channel=channel, message=msg_text, approved=True)

    status_msg = f"### ✅ Message Dispatched\n\n**Result:** {res.get('result')}\n\n**Details:** {res.get('details')}"
    history.append({"role": "assistant", "content": status_msg})

    return (
        history,
        gr.update(value="", visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        {}
    )


def build_app() -> gr.Blocks:
    """Constructs the optimized, clean, expandable Gradio user interface."""
    with gr.Blocks(title="The Predictors — Crystal Ball RAG") as demo:
        # State management

        current_section_state = gr.State("chat")
        pending_action_state = gr.State({})

        # Header Box
        with gr.Row(elem_id="header-bar"):
            with gr.Column(scale=4):
                gr.HTML("""
                <div>
                    <span class="title-badge">Team Predictors • Hackathon RAG &amp; Agent</span>
                    <span class="mode-badge">Autonomous Agent + Hybrid RAG</span>
                    <h1 style="margin: 4px 0 2px 0; font-size: 1.65rem; font-weight: 800; background: linear-gradient(90deg, #f8fafc, #94a3b8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                        The Crystal Ball Assistant
                    </h1>
                    <p style="margin: 0; color: #94a3b8; font-size: 0.88rem;">
                        Autonomous grounded Q&amp;A, multi-step tool reasoning, and MCP-connected workflow over <em>The Crystal Ball Instruction Manual (Volume One)</em> by Stephen Davies, Ph.D.
                    </p>
                </div>
                """)
            with gr.Column(scale=1, min_width=180):
                cost_summary = cost_tracker.get_summary()
                spend = cost_summary.get("total_cost_usd", 0.0)
                gr.HTML(f"""
                <div style="text-align: right; padding-top: 10px;">
                    <span class="budget-chip">● Spend: ${spend:.4f} / $5.00</span>
                </div>
                """)

        # Main Layout: Sidebar + Main Chatbot with Expandable Detail Sections
        with gr.Row(elem_id="main-app-row", equal_height=False):

            # -------------------------------------------------------------
            # 1. Compact Sidebar Navigation Menu (Narrow Left Sidebar)
            # -------------------------------------------------------------
            with gr.Column(scale=1, min_width=180, elem_id="sidebar-nav-col"):
                gr.HTML('<div class="nav-header-label">Views &amp; Details</div>')

                btn_nav_chat = gr.Button(
                    "💬 Chat View",
                    variant="primary",
                    elem_classes=["nav-btn", "nav-btn-active"],
                    elem_id="btn-nav-chat"
                )
                btn_nav_doc = gr.Button(
                    "📖 Document Info",
                    variant="secondary",
                    elem_classes=["nav-btn"],
                    elem_id="btn-nav-doc"
                )
                btn_nav_retrieval = gr.Button(
                    "🔍 Retrieval Details",
                    variant="secondary",
                    elem_classes=["nav-btn"],
                    elem_id="btn-nav-retrieval"
                )
                btn_nav_sources = gr.Button(
                    "📑 Sources & Citations",
                    variant="secondary",
                    elem_classes=["nav-btn"],
                    elem_id="btn-nav-sources"
                )
                btn_nav_eval = gr.Button(
                    "📊 Evaluation Results",
                    variant="secondary",
                    elem_classes=["nav-btn"],
                    elem_id="btn-nav-eval"
                )
                btn_nav_status = gr.Button(
                    "⚙️ System Status",
                    variant="secondary",
                    elem_classes=["nav-btn"],
                    elem_id="btn-nav-status"
                )

                gr.HTML('<div class="nav-header-label" style="margin-top: 16px;">Engine Mode</div>')
                engine_selector = gr.Radio(
                    choices=["🤖 Autonomous Agent (Multi-Tool Loop)", "⚡ Direct Hybrid RAG (Fixed Pipeline)"],
                    value="🤖 Autonomous Agent (Multi-Tool Loop)",
                    label="",
                    show_label=False,
                    elem_id="engine-selector"
                )

            # -------------------------------------------------------------
            # 2. Main Chatbot Interface (Large, Spacious, Always Visible)
            # -------------------------------------------------------------
            with gr.Column(scale=6, elem_id="chatbot-main-col"):
                # Clean welcome banner rendered via native Gradio HTML component
                gr.HTML("""
                <div style="background: rgba(30, 41, 59, 0.55); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 12px; padding: 12px 16px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                    <div>
                        <div style="font-weight: 700; color: #f8fafc; font-size: 0.92rem; display: flex; align-items: center; gap: 6px;">
                            <span>🔮</span> Welcome to The Predictors Assistant
                        </div>
                        <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 2px;">
                            Grounded Q&amp;A on <em>The Crystal Ball Instruction Manual (Volume One)</em>
                        </div>
                    </div>
                    <div style="display: flex; gap: 6px; flex-wrap: wrap;">
                        <span style="background: rgba(99,102,241,0.15); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.3); font-size: 0.72rem; padding: 2px 8px; border-radius: 9999px;">Autonomous ReAct</span>
                        <span style="background: rgba(16,185,129,0.15); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.3); font-size: 0.72rem; padding: 2px 8px; border-radius: 9999px;">FAISS + BM25 Hybrid</span>
                        <span style="background: rgba(245,158,11,0.15); color: #fcd34d; border: 1px solid rgba(245,158,11,0.3); font-size: 0.72rem; padding: 2px 8px; border-radius: 9999px;">Verified Citations</span>
                    </div>
                </div>
                """)

                chatbot = gr.Chatbot(
                    label="Conversation History & Grounded Citations",
                    elem_id="main-chatbot",
                    height=540,
                    autoscroll=True,
                    render_markdown=True,
                    line_breaks=True,
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "$", "right": "$", "display": False},
                        {"left": "\\(", "right": "\\)", "display": False},
                        {"left": "\\[", "right": "\\]", "display": True},
                    ],
                    placeholder="Ask a question about the textbook or select a scenario below to start chatting...",
                )

                # Human-in-the-Loop approval prompt card
                approval_card_html = gr.HTML("", visible=False)
                with gr.Row():
                    approve_btn = gr.Button("✅ Confirm & Send via MCP/Webhook", variant="primary", elem_classes=["approval-btn"], visible=False)
                    reject_btn = gr.Button("❌ Reject Action", variant="secondary", visible=False)

                # Input bar
                with gr.Row():
                    query_input = gr.Textbox(
                        label="Ask a question or issue an agent instruction:",
                        placeholder="e.g., What is the difference between the starting page of Chapter 15 and Chapter 6, and what is half that value?",
                        lines=2,
                        elem_id="query-box",
                        scale=5,
                    )

                with gr.Row():
                    submit_btn = gr.Button("🚀 Run Query", variant="primary", elem_classes=["primary-btn"], scale=2)
                    clear_btn = gr.Button("🧹 Clear Chat", variant="secondary", scale=1)

                # Feedback & Session Export Toolbar
                with gr.Row(elem_id="feedback-export-row"):
                    with gr.Column(scale=4):
                        with gr.Row():
                            btn_thumbs_up = gr.Button("👍 Helpful", variant="secondary", size="sm", elem_id="btn-thumbs-up")
                            btn_thumbs_down = gr.Button("👎 Not Helpful", variant="secondary", size="sm", elem_id="btn-thumbs-down")
                            btn_export_chat = gr.Button("📥 Export Chat (.md)", variant="secondary", size="sm", elem_id="btn-export-chat")
                            btn_ab_compare = gr.Button("⚖️ A/B Comparison", variant="secondary", size="sm", elem_id="btn-ab-compare")
                    with gr.Column(scale=2):
                        feedback_status = gr.HTML("<span style='color: #64748b; font-size: 0.8rem;'>Rate answers or compare retrieval strategies</span>", elem_id="feedback-status")

                download_export_file = gr.File(label="📄 Exported Conversation File", visible=False, elem_id="download-export-file")

                # Collapsible Quick Scenarios
                with gr.Accordion("💡 Curated Demonstration Scenarios (Click to populate)", open=False):
                    gr.Examples(
                        examples=[
                            # Multi-Step Tool Reasoning
                            ["What is the difference between the page number where Chapter 15 starts (145) and Chapter 6 starts (43), and what is half of that difference?"],
                            ["Look up the two Simpson family gender IQ medians reported on page 196 (female IQ 120.0 and male IQ 74.0), and calculate the average of these two median IQ values."],
                            ["Find the page where Chapter 15 starts (145) and the page where Chapter 6 starts (43), and calculate the difference between these two starting pages."],
                            # MCP Messaging with Human Approval
                            ["Please propose sending a message to #general summarizing the four levels of the data-to-wisdom hierarchy."],
                            # Core Corpus Grounding & Priority Questions
                            ["What are the four levels of the data-to-wisdom hierarchy described in Chapter 1?"],
                            ["How does the conceptual idea of an associative array in Chapter 7 transition into Pandas Series in Chapter 11?"],
                            ["What informal nickname does the author use for parentheses () throughout the book?"],
                            ["What is Simpson's Paradox as illustrated in Chapter 16?"],
                            # Out of scope refusal
                            ["What is the backpropagation algorithm for multi-layer neural networks and how are weight gradients computed?"],
                            # Ambiguity handling
                            ["How do you evaluate performance?"],
                        ],
                        inputs=[query_input],
                        label="Click any scenario to populate the input box:",
                    )

                # -------------------------------------------------------------
                # Expandable Full Agent Execution Trace (Main Interface)
                # -------------------------------------------------------------
                with gr.Accordion("🔍 Full Agent Execution Trace", open=False, elem_id="full-agent-trace-accordion"):
                    main_trace_box = gr.HTML(
                        value="<p style='color: #94a3b8; font-style: italic;'>No agent execution trace recorded yet. Run a query in Autonomous Agent mode to view the step-by-step trace.</p>",
                        elem_id="main-trace-box"
                    )

                # -------------------------------------------------------------
                # 3. Expandable Detail Section (HIDDEN BY DEFAULT, OPENS BELOW CHAT)
                # -------------------------------------------------------------
                with gr.Column(visible=False, elem_id="expandable-detail-section") as detail_panel:

                    # Section Header with Close / Return to Chat button
                    with gr.Row(elem_id="detail-section-header"):
                        with gr.Column(scale=4):
                            panel_title_md = gr.Markdown("### Section Title")
                        with gr.Column(scale=1, min_width=120):
                            close_panel_btn = gr.Button(
                                "✕ Close Section",
                                variant="secondary",
                                size="sm",
                                elem_classes=["close-panel-btn"],
                                elem_id="btn-close-detail-panel"
                            )

                    # Section 1: Document / Corpus Information
                    with gr.Group(visible=False, elem_id="sec-doc-info") as group_doc_info:
                        gr.HTML("""
                        <div class="info-card">
                            <div style="font-weight: 700; color: #f8fafc; font-size: 0.95rem; margin-bottom: 4px;">
                                📖 The Crystal Ball Instruction Manual
                            </div>
                            <div style="color: #94a3b8; font-size: 0.82rem; margin-bottom: 10px;">
                                Volume One: Introduction to Data Science, v1.2 • Stephen Davies, Ph.D.
                            </div>
                            <div class="metric-grid">
                                <div class="metric-pill">
                                    <div class="metric-label">Total Extracted Pages</div>
                                    <div class="metric-value" style="color: #38bdf8;">314 pages</div>
                                </div>
                                <div class="metric-pill">
                                    <div class="metric-label">Indexed Chunks</div>
                                    <div class="metric-value" style="color: #6ee7b7;">386 chunks</div>
                                </div>
                                <div class="metric-pill">
                                    <div class="metric-label">Chunk Overlap</div>
                                    <div class="metric-value" style="color: #facc15;">0 tokens (Strict)</div>
                                </div>
                                <div class="metric-pill">
                                    <div class="metric-label">Book Chapters</div>
                                    <div class="metric-value" style="color: #c084fc;">29 chapters</div>
                                </div>
                            </div>
                        </div>
                        """)

                        gr.Markdown("#### 📚 29-Chapter Table of Contents Directory")
                        gr.HTML(build_chapters_table_html())

                        gr.HTML("""
                        <div class="info-card" style="margin-top: 12px;">
                            <div style="font-weight: 700; color: #f8fafc; font-size: 0.88rem; margin-bottom: 6px;">
                                🎯 Key Topics &amp; Themes
                            </div>
                            <ul style="margin: 0; padding-left: 18px; font-size: 0.82rem; color: #cbd5e1; line-height: 1.5;">
                                <li><strong>DIKW Hierarchy:</strong> Data → Information → Knowledge → Wisdom (Chapter 1)</li>
                                <li><strong>Measurement Scales:</strong> Nominal, Ordinal, Interval, Ratio (Chapter 6)</li>
                                <li><strong>Python Data Structures:</strong> NumPy Arrays, Pandas Series, DataFrames (Ch 8-18)</li>
                                <li><strong>Exploratory Data Analysis:</strong> Univariate &amp; Bivariate EDA, Simpson's Paradox (Ch 15-20)</li>
                                <li><strong>Machine Learning:</strong> Decision Trees, Classification, Confusion Matrix (Ch 25-29)</li>
                            </ul>
                        </div>
                        """)

                    # Section 2: Retrieval Details & Real-Time Trajectory
                    with gr.Group(visible=False, elem_id="sec-retrieval") as group_retrieval:
                        gr.HTML("""
                        <div class="info-card">
                            <div style="font-weight: 700; color: #f8fafc; font-size: 0.95rem; margin-bottom: 4px;">
                                ⚙️ Hybrid Retrieval Architecture
                            </div>
                            <div class="metric-grid">
                                <div class="metric-pill">
                                    <div class="metric-label">Dense Vector Search</div>
                                    <div class="metric-value" style="color: #38bdf8;">FAISS (65%)</div>
                                </div>
                                <div class="metric-pill">
                                    <div class="metric-label">Sparse Lexical Search</div>
                                    <div class="metric-value" style="color: #6ee7b7;">Okapi BM25 (35%)</div>
                                </div>
                                <div class="metric-pill">
                                    <div class="metric-label">Embedding Model</div>
                                    <div class="metric-value" style="color: #facc15; font-size: 0.95rem;">text-embedding-3-small</div>
                                </div>
                                <div class="metric-pill">
                                    <div class="metric-label">Top-K &amp; Cutoff</div>
                                    <div class="metric-value" style="color: #c084fc;">K=7 | Cutoff=0.25</div>
                                </div>
                            </div>
                        </div>
                        """)

                        gr.Markdown("#### 🛠️ Autonomous Tool Trajectory & Observations")
                        trace_box = gr.HTML(
                            value="<p style='color: #94a3b8; font-style: italic;'>Autonomous tool sequence, arguments, and observations will render here in real time.</p>"
                        )

                        gr.Markdown("#### ⏱️ Latency & Token Performance")
                        stats_box = gr.HTML(
                            value=format_stats_markdown({}, "agent")
                        )

                    # Section 3: Sources and Citations
                    with gr.Group(visible=False, elem_id="sec-sources") as group_sources:
                        gr.HTML("""
                        <div class="info-card">
                            <div style="font-weight: 700; color: #f8fafc; font-size: 0.95rem; margin-bottom: 4px;">
                                📑 Grounding &amp; Citation Integrity
                            </div>
                            <p style="margin: 0 0 8px 0; font-size: 0.82rem; color: #cbd5e1; line-height: 1.45;">
                                Every factual assertion is grounded in the textbook with explicit <code>[Page X]</code> markers referencing the physical book page. Out-of-scope topics are strictly refused to guarantee 0% hallucination.
                            </p>
                        </div>
                        """)

                        gr.Markdown("#### 🔍 Retrieved Source Passages for Current Query")
                        evidence_box = gr.HTML(
                            value="<p style='color: #94a3b8; font-style: italic;'>No sources retrieved yet. Submit a query to inspect citations.</p>"
                        )

                    # Section 4: Evaluation Results & Benchmarks
                    with gr.Group(visible=False, elem_id="sec-eval") as group_eval:
                        with gr.Tabs():
                            with gr.Tab("📈 Retrieval Metrics Dashboard"):
                                gr.Markdown("#### 📈 Benchmark IR Evaluation Metrics (Recall@1, 3, 5 & MRR)")
                                with gr.Row():
                                    btn_run_eval = gr.Button("⚡ Run / Re-Evaluate Benchmark (16 Questions)", variant="primary", size="sm")
                                metrics_dashboard_html = gr.HTML(
                                    value=format_metrics_dashboard_html(load_or_run_benchmark(force_run=False))
                                )

                            with gr.Tab("⚖️ A/B Retrieval Comparison"):
                                gr.Markdown("#### ⚖️ Side-by-Side Retrieval Strategy Comparison")
                                gr.Markdown("Compare Vector-only, BM25-only, Hybrid, and Cross-Encoder Reranking on the same query.")
                                with gr.Row():
                                    ab_query_input = gr.Textbox(
                                        label="Comparison Question:",
                                        value="What is Simpson's Paradox as illustrated with the Simpson family example?",
                                        lines=1,
                                        scale=4,
                                    )
                                    btn_run_ab = gr.Button("🔍 Run A/B Comparison", variant="primary", scale=1)
                                ab_results_html = gr.HTML(
                                    value="<p style='color: #94a3b8; font-style: italic;'>Click 'Run A/B Comparison' to compare Vector, BM25, Hybrid, and Cross-Encoder retrieval results.</p>"
                                )

                            with gr.Tab("🛡️ Faithfulness & Verification"):
                                gr.HTML("""
                                <div class="info-card">
                                    <div style="font-weight: 700; color: #f8fafc; font-size: 0.95rem; margin-bottom: 4px;">
                                        🛡️ Faithfulness Verification & Grounding Engine
                                    </div>
                                    <p style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 8px;">
                                        Post-generation verification validates that all claims and physical book citations <code>[Page X]</code> in the generated answer are explicitly supported by retrieved excerpts.
                                    </p>
                                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin-top: 10px;">
                                        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 10px;">
                                            <div style="color: #6ee7b7; font-weight: 700; font-size: 0.85rem;">✅ Faithful</div>
                                            <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 4px;">All claims directly confirmed by retrieved textbook excerpts with valid page citations.</div>
                                        </div>
                                        <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 10px;">
                                            <div style="color: #fcd34d; font-weight: 700; font-size: 0.85rem;">⚠️ Partially Supported</div>
                                            <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 4px;">Core claims present, but minor details or specific citations lack exact excerpt grounding.</div>
                                        </div>
                                        <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 10px;">
                                            <div style="color: #fca5a5; font-weight: 700; font-size: 0.85rem;">❌ Unsupported</div>
                                            <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 4px;">Claims not substantiated by evidence; triggers automated 1-step retrieval adjustment.</div>
                                        </div>
                                        <div style="background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 8px; padding: 10px;">
                                            <div style="color: #7dd3fc; font-weight: 700; font-size: 0.85rem;">🛡️ Faithful Refusal</div>
                                            <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 4px;">Correct refusal when corpus does not contain the answer, preventing hallucination.</div>
                                        </div>
                                    </div>
                                </div>
                                """)

                            with gr.Tab("📊 Category Benchmark Summary"):
                                gr.HTML("""
                                <div class="info-card">
                                    <div style="font-weight: 700; color: #f8fafc; font-size: 0.95rem; margin-bottom: 4px;">
                                        📊 Benchmark Evaluation Summary (45 Test Queries)
                                    </div>
                                    <div class="metric-grid">
                                        <div class="metric-pill">
                                            <div class="metric-label">Recall@7 (Hit Rate)</div>
                                            <div class="metric-value" style="color: #10b981;">82.9%</div>
                                        </div>
                                        <div class="metric-pill">
                                            <div class="metric-label">Out-of-Scope Refusal</div>
                                            <div class="metric-value" style="color: #38bdf8;">100.0%</div>
                                        </div>
                                        <div class="metric-pill">
                                            <div class="metric-label">Full Context Hit Rate</div>
                                            <div class="metric-value" style="color: #a78bfa;">68.6%</div>
                                        </div>
                                        <div class="metric-pill">
                                            <div class="metric-label">Average Latency</div>
                                            <div class="metric-value" style="color: #facc15;">2,201.9 ms</div>
                                        </div>
                                    </div>
                                </div>

                                <div style="margin-top: 10px;">
                                    <h4 style="margin: 0 0 6px 0; color: #f8fafc; font-size: 0.9rem;">Category Performance Breakdown</h4>
                                    <table class="glass-table">
                                        <thead>
                                            <tr>
                                                <th>Category</th>
                                                <th>Queries</th>
                                                <th>Hit Rate</th>
                                                <th>Refusal / Clarify</th>
                                                <th>Avg Latency</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td><strong style="color: #38bdf8;">factual</strong></td>
                                                <td>12</td>
                                                <td>91.7%</td>
                                                <td>100.0%</td>
                                                <td>2,357 ms</td>
                                            </tr>
                                            <tr>
                                                <td><strong style="color: #818cf8;">cross-reference</strong></td>
                                                <td>9</td>
                                                <td>88.9%</td>
                                                <td>100.0%</td>
                                                <td>2,526 ms</td>
                                            </tr>
                                            <tr>
                                                <td><strong style="color: #fb923c;">stress-test</strong></td>
                                                <td>14</td>
                                                <td>71.4%</td>
                                                <td>100.0%</td>
                                                <td>2,070 ms</td>
                                            </tr>
                                            <tr>
                                                <td><strong style="color: #f43f5e;">out-of-scope</strong></td>
                                                <td>5</td>
                                                <td>100.0%</td>
                                                <td>100.0% (Refused)</td>
                                                <td>981 ms</td>
                                            </tr>
                                            <tr>
                                                <td><strong style="color: #e879f9;">ambiguous</strong></td>
                                                <td>5</td>
                                                <td>100.0%</td>
                                                <td>40.0% (Clarified)</td>
                                                <td>2,836 ms</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                                """)

                    # Section 5: System Status / Settings
                    with gr.Group(visible=False, elem_id="sec-status") as group_status:
                        cost_info = cost_tracker.get_summary()
                        gr.HTML(f"""
                        <div class="info-card">
                            <div style="font-weight: 700; color: #f8fafc; font-size: 0.95rem; margin-bottom: 4px;">
                                💳 Real-Time Budget &amp; Spend Tracker
                            </div>
                            <div class="metric-grid">
                                <div class="metric-pill">
                                    <div class="metric-label">Total Spend ($5 Hard Cap)</div>
                                    <div class="metric-value" style="color: #10b981;">${cost_info.get('total_cost_usd', 0.0):.4f}</div>
                                </div>
                                <div class="metric-pill">
                                    <div class="metric-label">Budget Used</div>
                                    <div class="metric-value" style="color: #38bdf8;">{cost_info.get('budget_used_percent', 0.0):.2f}%</div>
                                </div>
                                <div class="metric-pill">
                                    <div class="metric-label">Embedding Tokens</div>
                                    <div class="metric-value" style="color: #facc15;">{cost_info.get('embedding_tokens', 0):,}</div>
                                </div>
                                <div class="metric-pill">
                                    <div class="metric-label">Chat Tokens</div>
                                    <div class="metric-value" style="color: #c084fc;">{cost_info.get('chat_input_tokens', 0) + cost_info.get('chat_output_tokens', 0):,}</div>
                                </div>
                            </div>
                        </div>

                        <div class="info-card">
                            <div style="font-weight: 700; color: #f8fafc; font-size: 0.88rem; margin-bottom: 6px;">
                                ⚙️ Infrastructure &amp; Tool Health
                            </div>
                            <ul style="margin: 0; padding-left: 18px; font-size: 0.82rem; color: #cbd5e1; line-height: 1.5;">
                                <li><strong>FAISS Vector Store:</strong> <span style="color: #6ee7b7;">● Active</span> (386 vectors, 1536 dimensions)</li>
                                <li><strong>BM25 Lexical Index:</strong> <span style="color: #6ee7b7;">● Active</span> (386 chunks, Okapi BM25)</li>
                                <li><strong>Embedding Model:</strong> <code>text-embedding-3-small</code> ($0.02 / 1M tokens)</li>
                                <li><strong>LLM Generator:</strong> <code>gpt-4o-mini</code> ($0.15 in / $0.60 out per 1M tokens)</li>
                                <li><strong>Human-in-the-Loop Gateway:</strong> Active for irreversible actions (Slack/Teams)</li>
                            </ul>
                        </div>

                        <div style="margin-top: 12px;">
                            <h4 style="margin: 0 0 6px 0; color: #f8fafc; font-size: 0.9rem;">Recent MCP / Human-in-the-Loop Message Dispatches</h4>
                            {load_audit_log_html()}
                        </div>
                        """)

        # -------------------------------------------------------------
        # Navigation Wire-Up: Toggle / Replace Section Logic
        # -------------------------------------------------------------
        nav_outputs = [
            current_section_state,
            detail_panel,
            panel_title_md,
            group_doc_info,
            group_retrieval,
            group_sources,
            group_eval,
            group_status,
            btn_nav_chat,
            btn_nav_doc,
            btn_nav_retrieval,
            btn_nav_sources,
            btn_nav_eval,
            btn_nav_status,
        ]

        btn_nav_chat.click(
            fn=lambda cur: switch_section("chat", cur),
            inputs=[current_section_state],
            outputs=nav_outputs,
        )
        btn_nav_doc.click(
            fn=lambda cur: switch_section("doc", cur),
            inputs=[current_section_state],
            outputs=nav_outputs,
        )
        btn_nav_retrieval.click(
            fn=lambda cur: switch_section("retrieval", cur),
            inputs=[current_section_state],
            outputs=nav_outputs,
        )
        btn_nav_sources.click(
            fn=lambda cur: switch_section("sources", cur),
            inputs=[current_section_state],
            outputs=nav_outputs,
        )
        btn_nav_eval.click(
            fn=lambda cur: switch_section("eval", cur),
            inputs=[current_section_state],
            outputs=nav_outputs,
        )
        btn_nav_status.click(
            fn=lambda cur: switch_section("status", cur),
            inputs=[current_section_state],
            outputs=nav_outputs,
        )
        close_panel_btn.click(
            fn=lambda cur: switch_section("chat", cur),
            inputs=[current_section_state],
            outputs=nav_outputs,
        )

        # -------------------------------------------------------------
        # Query Submission & Chatbot Wire-Up
        # -------------------------------------------------------------
        query_outputs = [
            chatbot,
            query_input,
            main_trace_box,
            trace_box,
            evidence_box,
            stats_box,
            approval_card_html,
            approve_btn,
            reject_btn,
            pending_action_state,
        ]

        submit_btn.click(
            fn=process_query,
            inputs=[query_input, engine_selector, chatbot],
            outputs=query_outputs,
        )
        query_input.submit(
            fn=process_query,
            inputs=[query_input, engine_selector, chatbot],
            outputs=query_outputs,
        )

        # -------------------------------------------------------------
        # Human Approval Wire-Up
        # -------------------------------------------------------------
        approve_btn.click(
            fn=execute_approved_message,
            inputs=[pending_action_state, chatbot],
            outputs=[chatbot, approval_card_html, approve_btn, reject_btn, pending_action_state],
        )

        def reject_action(history):
            h = list(history or [])
            h.append({"role": "assistant", "content": "*Action was cancelled by human user. No external message was sent.*"})
            return (h, gr.update(value="", visible=False), gr.update(visible=False), gr.update(visible=False), {})

        reject_btn.click(
            fn=reject_action,
            inputs=[chatbot],
            outputs=[chatbot, approval_card_html, approve_btn, reject_btn, pending_action_state],
        )

        # -------------------------------------------------------------
        # Clear Chat Wire-Up
        # -------------------------------------------------------------
        def clear_all():
            return (
                [],
                "",
                "<p style='color: #94a3b8; font-style: italic;'>No agent execution trace recorded yet. Run a query in Autonomous Agent mode to view the step-by-step trace.</p>",
                "<p style='color: #94a3b8; font-style: italic;'>Autonomous tool sequence, arguments, and observations will render here in real time.</p>",
                "<p style='color: #94a3b8; font-style: italic;'>No sources retrieved yet. Submit a query to inspect citations.</p>",
                format_stats_markdown({}, "agent"),
                gr.update(value="", visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                {},
            )


        # -------------------------------------------------------------
        # User Feedback & Session Export Wire-Up
        # -------------------------------------------------------------
        def record_thumbs_up(history):
            if history:
                last_q = next((m.get("content", "") for m in reversed(history) if m.get("role") == "user"), "")
                last_a = next((m.get("content", "") for m in reversed(history) if m.get("role") == "assistant"), "")
                save_feedback(last_q, last_a, "thumbs_up")
                return "<span style='color: #10b981; font-weight: 600; font-size: 0.82rem;'>👍 Feedback recorded: Helpful! Thank you.</span>"
            return "<span style='color: #94a3b8; font-size: 0.82rem;'>No message to rate yet.</span>"

        def record_thumbs_down(history):
            if history:
                last_q = next((m.get("content", "") for m in reversed(history) if m.get("role") == "user"), "")
                last_a = next((m.get("content", "") for m in reversed(history) if m.get("role") == "assistant"), "")
                save_feedback(last_q, last_a, "thumbs_down")
                return "<span style='color: #f59e0b; font-weight: 600; font-size: 0.82rem;'>👎 Feedback recorded: Needs Improvement. Saved to audit log.</span>"
            return "<span style='color: #94a3b8; font-size: 0.82rem;'>No message to rate yet.</span>"

        def export_chat_session(history):
            if not history:
                return gr.update(visible=False)
            _, file_path = chat_memory.export_to_markdown()
            return gr.update(value=str(file_path), visible=True)

        btn_thumbs_up.click(
            fn=record_thumbs_up,
            inputs=[chatbot],
            outputs=[feedback_status],
        )
        btn_thumbs_down.click(
            fn=record_thumbs_down,
            inputs=[chatbot],
            outputs=[feedback_status],
        )
        btn_export_chat.click(
            fn=export_chat_session,
            inputs=[chatbot],
            outputs=[download_export_file],
        )

        # Wire up A/B Comparison and Benchmark Evaluation Handlers
        def trigger_ab_from_chat(cur_sec, q_text):
            nav_tuple = switch_section("eval", cur_sec)
            target_q = q_text.strip() if q_text and q_text.strip() else "What is Simpson's Paradox as illustrated with the Simpson family example?"
            return nav_tuple + (target_q,)

        btn_ab_compare.click(
            fn=trigger_ab_from_chat,
            inputs=[current_section_state, query_input],
            outputs=nav_outputs + [ab_query_input],
        )

        btn_run_ab.click(
            fn=lambda q: format_ab_comparison_html(run_retrieval_ab_comparison(q)),
            inputs=[ab_query_input],
            outputs=[ab_results_html],
        )
        ab_query_input.submit(
            fn=lambda q: format_ab_comparison_html(run_retrieval_ab_comparison(q)),
            inputs=[ab_query_input],
            outputs=[ab_results_html],
        )

        btn_run_eval.click(
            fn=lambda: format_metrics_dashboard_html(run_benchmark_evaluation()),
            outputs=[metrics_dashboard_html],
        )

        clear_btn.click(
            fn=clear_all,
            outputs=query_outputs,
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    try:
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False, css=CUSTOM_CSS)
    except OSError:
        demo.launch(server_name="127.0.0.1", share=False, css=CUSTOM_CSS)
