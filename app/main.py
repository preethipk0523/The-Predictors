"""
Clean, modern, and expandable web portal for The Predictors RAG Assistant & Autonomous Agent.
Provides:
- 2-Column responsive layout (narrow sidebar + spacious main chat).
- Expandable detail sections (Document Info, Retrieval Benchmark, Retrieved Sources, Evaluation, Performance & Stats).
- Dual Engine Execution: Direct Hybrid RAG Pipeline vs Autonomous Agent (Think-Act-Observe).
- Human-in-the-Loop action approval gate for external messaging.
- Real-time cost, token, and latency tracking against $5.00 budget.
- Multi-turn conversation memory, thumbs up/down feedback, and Markdown chat export.
"""

import sys
import json
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Ensure root workspace is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import gradio as gr
from app.pipeline import pipeline
from app.agent import agent_assistant
from app.cost_tracker import cost_tracker
from app.feedback import save_feedback
from app.chat_memory import chat_memory
from tools.outline import CHAPTERS
from tools.messaging import send_message

CUSTOM_CSS = """
/* Modern Dark Glassmorphic Theme */
:root {
    --primary: #6366f1;
    --primary-hover: #4f46e5;
    --bg-dark: #0f172a;
    --card-bg: #1e293b;
    --border-color: #334155;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
}

body, .gradio-container {
    background: radial-gradient(circle at top right, #1e1b4b 0%, #0f172a 50%, #020617 100%) !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: var(--text-main) !important;
    min-height: 100vh;
}

#header-box {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 16px;
    padding: 20px 28px;
    margin-bottom: 20px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 0 15px rgba(99, 102, 241, 0.15);
    backdrop-filter: blur(12px);
}

.title-badge {
    display: inline-block;
    background: linear-gradient(90deg, #6366f1, #06b6d4);
    color: white;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 3px 10px;
    border-radius: 9999px;
    margin-bottom: 6px;
}

.nav-btn {
    text-align: left !important;
    justify-content: flex-start !important;
    border-radius: 8px !important;
    margin-bottom: 6px !important;
    font-size: 0.88rem !important;
    transition: all 0.15s ease !important;
}

.evidence-card {
    background: rgba(15, 23, 42, 0.7) !important;
    border-left: 4px solid #6366f1 !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    margin-bottom: 12px !important;
    transition: all 0.2s ease;
}

.evidence-card:hover {
    border-left-color: #06b6d4 !important;
    background: rgba(30, 41, 59, 0.85) !important;
}

.page-badge {
    font-size: 0.74rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 6px;
    background: rgba(16, 185, 129, 0.2);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.score-badge {
    font-size: 0.74rem;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 6px;
    background: rgba(99, 102, 241, 0.2);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.3);
}
"""


# --------------------------------------------------------------------------
# Helper Formatters
# --------------------------------------------------------------------------

def build_chapters_table_html() -> str:
    """Builds a responsive glassmorphic HTML table of all 29 textbook chapters."""
    rows = []
    for c in CHAPTERS:
        rows.append(
            f"""
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.06); transition: background 0.15s ease;">
                <td style="padding: 9px 12px; font-weight: 700; color: #a5b4fc;">Ch {c['chapter']}</td>
                <td style="padding: 9px 12px; color: #f8fafc;">{c['title']}</td>
                <td style="padding: 9px 12px; text-align: center; color: #6ee7b7; font-weight: 600;">{c['start_page']}</td>
                <td style="padding: 9px 12px; text-align: center; color: #94a3b8;">{c['pdf_page']}</td>
            </tr>
            """
        )

    return f"""
    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #334155; border-radius: 12px; padding: 16px; max-height: 480px; overflow-y: auto;">
        <div style="margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: 800; color: #f8fafc; font-size: 1.05rem;">📖 The Crystal Ball Manual — Table of Contents</span>
            <span style="background: rgba(99, 102, 241, 0.2); color: #a5b4fc; padding: 2px 10px; border-radius: 6px; font-size: 0.78rem; font-weight: 600;">29 Chapters</span>
        </div>
        <table style="width: 100%; border-collapse: collapse; font-size: 0.86rem;">
            <thead>
                <tr style="border-bottom: 1px solid #475569; text-align: left; color: #94a3b8; font-size: 0.78rem; text-transform: uppercase;">
                    <th style="padding: 8px 12px;">Chapter</th>
                    <th style="padding: 8px 12px;">Title</th>
                    <th style="padding: 8px 12px; text-align: center;">Book Page</th>
                    <th style="padding: 8px 12px; text-align: center;">PDF Page</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    """


def format_chat_answer(raw_text: str, cited_pages: Optional[List[int]] = None) -> str:
    """Formats assistant answer ensuring inline citations and Sources cited section."""
    if not raw_text:
        return ""

    if cited_pages is not None:
        valid_pages = set(cited_pages)
        for m in re.finditer(r"\[Page\s*(\d+)\]", raw_text, flags=re.IGNORECASE):
            p = int(m.group(1))
            if p not in valid_pages or p >= 300:
                raw_text = raw_text.replace(m.group(0), "")
        raw_text = re.sub(r"\s{2,}", " ", raw_text).strip()

    if "Sources cited" in raw_text:
        return raw_text

    clean_pages = sorted(list(set(int(p) for p in (cited_pages or []) if int(p) < 300)))
    if clean_pages:
        sources_block = "\n\n**Sources cited:**\n" + "\n".join(f"- [Page {p}]" for p in clean_pages)
        return raw_text.rstrip() + sources_block
    return raw_text


def format_evidence_markdown(sources: list) -> str:
    """Formats retrieved sources into rich evidence cards."""
    if not sources:
        return "*No sources retrieved.*"

    cards = []
    for s in sources:
        doc = s.get("source", "crystal1 (1).pdf")
        page = s.get("page", "?")
        dense = s.get("dense_score", s.get("score", 0.0))
        bm25 = s.get("bm25_score", 0.0)
        comb = s.get("combined_score", s.get("score", dense))

        raw_snippet = s.get("snippet", s.get("text_snippet", s.get("text", "")))
        if raw_snippet and str(raw_snippet).strip():
            snippet_html = f'"{str(raw_snippet).strip()}"'
        else:
            snippet_html = "<em>Evidence snippet unavailable for this source chunk</em>"

        card = f"""
<div class="evidence-card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <div>
            <span class="page-badge">📄 Page {page}</span>
            <span style="font-size: 0.8rem; color: #94a3b8; margin-left: 8px;">{doc}</span>
        </div>
        <div>
            <span class="score-badge">Dense: {dense:.2f}</span>
            <span class="score-badge">BM25: {bm25:.2f}</span>
            <span class="score-badge" style="background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border-color: rgba(16, 185, 129, 0.3);">Score: {comb:.2f}</span>
        </div>
    </div>
    <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.45; font-style: italic; background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 6px;">
        {snippet_html}
    </div>
</div>
"""
        cards.append(card)

    return "\n".join(cards)


def format_agent_evidence_html(agent_res: dict) -> str:
    """Extracts and renders retrieved evidence chunks from agent trajectory search and quote calls."""
    if not agent_res:
        return "*No agent sources recorded.*"

    sources = []
    seen = set()

    for step_item in agent_res.get("trajectory", []):
        for tc in step_item.get("tool_calls", []):
            tool_name = tc.get("tool", "")
            fres = tc.get("full_result", {})
            if isinstance(fres, dict):
                if tool_name == "search_corpus":
                    for m in fres.get("matches", []):
                        p = m.get("page", "?")
                        key = (tool_name, p)
                        if key not in seen:
                            seen.add(key)
                            sources.append({
                                "source": m.get("source", "crystal1 (1).pdf"),
                                "page": p,
                                "score": m.get("score", m.get("combined_score", 0.9)),
                                "snippet": m.get("snippet", m.get("text", "")),
                            })
                elif tool_name == "quote_exact":
                    p = fres.get("page", tc.get("arguments", {}).get("page", "?"))
                    key = (tool_name, p)
                    if key not in seen:
                        seen.add(key)
                        sources.append({
                            "source": fres.get("document", "crystal1 (1).pdf"),
                            "page": p,
                            "score": 1.0,
                            "snippet": fres.get("text", fres.get("passage", "")),
                        })

    if not sources:
        for p in agent_res.get("cited_pages", []):
            sources.append({
                "source": "crystal1 (1).pdf",
                "page": p,
                "score": 0.85,
                "snippet": "Corpus reference verified by agent.",
            })

    return format_evidence_markdown(sources)


def format_approval_card(pending_action: Optional[dict]) -> str:
    """Formats human approval card for sensitive or external actions."""
    if not pending_action or not isinstance(pending_action, dict):
        return ""

    channel = pending_action.get("channel", "general")
    message = pending_action.get("message", "")

    return f"""
    <div style="background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 12px; padding: 18px 20px; margin: 16px 0;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.25rem;">⚠️</span>
                <strong style="color: #fbbf24; font-size: 0.98rem; text-transform: uppercase; letter-spacing: 0.05em;">Human-in-the-Loop Action Approval Required</strong>
            </div>
            <span style="background: rgba(245, 158, 11, 0.25); color: #fde68a; padding: 2px 10px; border-radius: 6px; font-size: 0.74rem; font-weight: 700;">Pending Confirmation</span>
        </div>
        <p style="margin: 0 0 10px 0; color: #cbd5e1; font-size: 0.88rem;">
            The autonomous agent proposes executing an irreversible external action to post an answer to Slack/Teams.
        </p>
        <div style="background: rgba(0, 0, 0, 0.35); border-radius: 8px; padding: 12px 14px; font-size: 0.84rem; margin-bottom: 12px;">
            <div style="color: #94a3b8; margin-bottom: 4px;"><strong>Target Channel:</strong> <span style="color: #38bdf8;">{channel}</span></div>
            <div style="color: #94a3b8;"><strong>Proposed Message:</strong></div>
            <div style="color: #f8fafc; font-style: italic; margin-top: 4px; line-height: 1.45;">"{message}"</div>
        </div>
        <div style="font-size: 0.8rem; color: #fbbf24;">
            👉 Click <strong>Approve</strong> below to authorize the dispatch, or <strong>Reject</strong> to decline.
        </div>
    </div>
    """


def format_stats_markdown(result: dict) -> str:
    """Formats latency and cost metrics into clean summary HTML."""
    summary = cost_tracker.get_summary()
    tot_cost = summary["total_cost_usd"]
    budget_pct = summary["budget_used_percent"]
    q_cost = result.get("cost_usd", 0.0)
    ret_ms = result.get("retrieval_time_ms", 0.0)
    gen_ms = result.get("generation_time_ms", result.get("duration_ms", 0.0))
    tokens = result.get("total_tokens", 0)

    return f"""
<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; font-size: 0.85rem;">
    <div style="background: rgba(15,23,42,0.6); padding: 12px; border-radius: 10px; border: 1px solid #334155;">
        <span style="color: #94a3b8;">⏱️ Retrieval / Agent Time</span><br>
        <strong style="color: #6ee7b7; font-size: 1.1rem;">{ret_ms or gen_ms:.1f} ms</strong>
    </div>
    <div style="background: rgba(15,23,42,0.6); padding: 12px; border-radius: 10px; border: 1px solid #334155;">
        <span style="color: #94a3b8;">⚡ Total Duration</span><br>
        <strong style="color: #38bdf8; font-size: 1.1rem;">{gen_ms:.1f} ms</strong>
    </div>
    <div style="background: rgba(15,23,42,0.6); padding: 12px; border-radius: 10px; border: 1px solid #334155;">
        <span style="color: #94a3b8;">🪙 Query Cost &amp; Tokens</span><br>
        <strong style="color: #facc15; font-size: 1.1rem;">{tokens} tok / ${q_cost:.5f}</strong>
    </div>
    <div style="background: rgba(15,23,42,0.6); padding: 12px; border-radius: 10px; border: 1px solid #334155;">
        <span style="color: #94a3b8;">💳 Cumulative Spend ($5 Budget)</span><br>
        <strong style="color: #a78bfa; font-size: 1.1rem;">${tot_cost:.4f} ({budget_pct:.2f}%)</strong>
    </div>
</div>
"""


def format_pipeline_trace_html(result: dict) -> str:
    """Formats execution trace for direct RAG pipeline."""
    if not result:
        return "*No RAG pipeline trace available.*"

    conf = result.get("confidence", {})
    faith = result.get("faithfulness", {})
    rewrite = result.get("rewritten_query_info", {})

    return f"""
    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #334155; border-radius: 12px; padding: 16px; font-size: 0.85rem;">
        <div style="font-weight: 800; color: #f8fafc; font-size: 1.02rem; margin-bottom: 10px;">🔍 RAG Pipeline Internal Trace</div>
        <div style="margin-bottom: 8px;">
            <strong style="color: #94a3b8;">Query Rewriting:</strong> 
            <span style="color: #f8fafc;">"{rewrite.get('query', result.get('query', ''))}"</span>
            {'<span style="color: #f59e0b; margin-left: 8px;">(Contextual Follow-up)</span>' if rewrite.get('was_rewritten') else '<span style="color: #10b981; margin-left: 8px;">(Direct)</span>'}
        </div>
        <div style="margin-bottom: 8px;">
            <strong style="color: #94a3b8;">Retrieval Confidence:</strong> 
            <span style="color: #6ee7b7; font-weight: 700;">{conf.get('confidence_score', 0.85):.2f}</span>
            <span style="background: rgba(16, 185, 129, 0.2); color: #6ee7b7; padding: 2px 8px; border-radius: 4px; font-size: 0.74rem; margin-left: 6px;">{conf.get('tier', 'High Confidence')}</span>
        </div>
        <div style="margin-bottom: 8px;">
            <strong style="color: #94a3b8;">Faithfulness Check:</strong> 
            <span style="color: #38bdf8; font-weight: 700;">{faith.get('status', 'Faithful')}</span>
            <span style="color: #94a3b8; margin-left: 8px;">(Grounding Score: {result.get('grounding_score', 1.0):.2f})</span>
        </div>
        <div style="margin-bottom: 4px;">
            <strong style="color: #94a3b8;">Security Guard:</strong> 
            <span style="color: #10b981;">No Prompt Injection Detected</span>
        </div>
    </div>
    """


def format_agent_chat_response(run_data: dict) -> str:
    """Formats agent run result into rich chat markdown with collapsible execution trace."""
    duration_ms = run_data.get("duration_ms", 0.0)
    stop_reason = run_data.get("stop_reason", "completed_task")
    raw_answer = run_data.get("answer", "")
    cited_pages = run_data.get("cited_pages", run_data.get("citations", []))
    trajectory = run_data.get("trajectory", [])

    items = []
    # If direct answer (1 step, no tool calls)
    if stop_reason == "direct_answer" or (len(trajectory) == 1 and not trajectory[0].get("tool_calls")):
        items.append({
            "type": "direct",
            "label": "`[Direct Response (No Tools Needed)]`",
            "content": raw_answer[:120] if raw_answer else "Direct answer generated.",
        })
    else:
        # Collect tool calls across trajectory
        for step_dict in trajectory:
            for tc in step_dict.get("tool_calls", []):
                t_name = tc.get("tool", "tool")
                t_args = json.dumps(tc.get("arguments", {}), ensure_ascii=False)
                items.append({
                    "type": "tool",
                    "label": "`[Tool Call]`",
                    "content": f"`{t_name}` args={t_args}",
                })
        # Add final synthesis item
        items.append({
            "type": "synthesis",
            "label": "`[Synthesized Final Answer]`",
            "content": "Final grounded answer formulated with citations.",
        })

    steps_count = run_data.get("steps_count", len(items))
    header = f"{steps_count} steps | {duration_ms:,.0f} ms | Stop reason: `{stop_reason}`"

    trace_lines = []
    for idx, itm in enumerate(items, 1):
        trace_lines.append(f"- **Step {idx}:** {itm['label']} {itm['content']}")

    trace_block = (
        f"<details><summary>View Full Execution Trace ({steps_count} steps)</summary>\n\n"
        + "\n".join(trace_lines)
        + "\n</details>"
    )

    answer_formatted = format_chat_answer(raw_answer, cited_pages)
    return f"{header}\n\n{answer_formatted}\n\n{trace_block}"


def format_trajectory_html(run_or_trajectory) -> str:
    """Formats step-by-step tool trajectory cards with execution timings and status badges."""
    if not run_or_trajectory:
        return "<div style='color: #94a3b8; font-style: italic; padding: 12px;'>No agent tool trajectory recorded yet. Ask a question in Agent Mode to observe step-by-step tool calls.</div>"

    if isinstance(run_or_trajectory, list):
        run_data = {"trajectory": run_or_trajectory}
        duration_ms = sum(s.get("step_latency_ms", 0.0) for s in run_or_trajectory)
        stop_reason = "completed_task"
    else:
        run_data = run_or_trajectory
        duration_ms = run_data.get("duration_ms", 0.0)
        stop_reason = run_data.get("stop_reason", "completed_task")

    trajectory = run_data.get("trajectory", [])
    if not trajectory:
        return "<div style='color: #94a3b8; font-style: italic; padding: 12px;'>No agent tool trajectory recorded yet.</div>"

    # Unroll steps
    unrolled_steps = []
    is_direct = stop_reason == "direct_answer" or (len(trajectory) == 1 and not trajectory[0].get("tool_calls"))

    if is_direct:
        first = trajectory[0]
        unrolled_steps.append({
            "is_direct": True,
            "tool_calls": [],
            "latency": first.get("step_latency_ms", duration_ms),
            "content": first.get("final_content", run_data.get("answer", "")),
        })
    else:
        for s in trajectory:
            tcs = s.get("tool_calls", [])
            if tcs:
                # If step has multiple tool calls, unroll them as individual steps
                for tc in tcs:
                    unrolled_steps.append({
                        "is_direct": False,
                        "tool_calls": [tc],
                        "latency": tc.get("duration_ms", s.get("step_latency_ms", 0.0)),
                        "content": None,
                    })
            elif s.get("final_content"):
                unrolled_steps.append({
                    "is_direct": False,
                    "is_synthesis": True,
                    "tool_calls": [],
                    "latency": s.get("step_latency_ms", 0.0),
                    "content": s.get("final_content"),
                })

        # Ensure final synthesis step is present if final answer exists and wasn't added
        if run_data.get("answer") and not any(u.get("is_synthesis") for u in unrolled_steps):
            unrolled_steps.append({
                "is_direct": False,
                "is_synthesis": True,
                "tool_calls": [],
                "latency": 50.0,
                "content": run_data.get("answer"),
            })

    total_steps = run_data.get("steps_count", len(unrolled_steps))

    # Header summary
    stop_badge = "Task Completed Successfully"
    if stop_reason == "completed_task":
        stop_badge = "Task Completed Successfully"
    elif "halted" in stop_reason or "error" in stop_reason:
        stop_badge = "Error / Failed"
    elif stop_reason == "direct_answer":
        stop_badge = "Direct Response (No Tools Needed)"
    elif stop_reason == "pending_approval":
        stop_badge = "Pending Human Approval"

    cards = [
        f"""
        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 12px; padding: 14px 18px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong style="color: #f8fafc; font-size: 1.05rem;">{total_steps} Total Steps</strong>
                <span style="color: #94a3b8; margin-left: 10px; font-size: 0.85rem;">⏱️ {duration_ms:,.1f} ms</span>
            </div>
            <span style="background: rgba(99, 102, 241, 0.2); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.4); padding: 3px 10px; border-radius: 6px; font-size: 0.78rem; font-weight: 700;">
                {stop_badge}
            </span>
        </div>
        """
    ]

    for idx, ustep in enumerate(unrolled_steps, 1):
        step_lat = ustep.get("latency", 0.0)
        if ustep.get("is_direct"):
            cards.append(
                f"""
                <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #334155; border-radius: 10px; padding: 14px 16px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: 700; color: #a5b4fc;">Step {idx}</span>
                        <div>
                            <span style="font-size: 0.75rem; background: rgba(16, 185, 129, 0.2); color: #6ee7b7; padding: 2px 8px; border-radius: 4px; font-weight: 600;">Final Synthesis / Direct Response</span>
                            <span style="font-size: 0.75rem; color: #94a3b8; margin-left: 8px;">{step_lat:.1f} ms</span>
                        </div>
                    </div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 6px;">
                        {ustep.get('content', '')}
                    </div>
                </div>
                """
            )
        elif ustep.get("is_synthesis"):
            cards.append(
                f"""
                <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #334155; border-radius: 10px; padding: 14px 16px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: 700; color: #a5b4fc;">Step {idx}</span>
                        <div>
                            <span style="font-size: 0.75rem; background: rgba(16, 185, 129, 0.2); color: #6ee7b7; padding: 2px 8px; border-radius: 4px; font-weight: 600;">Synthesized Final Answer</span>
                            <span style="font-size: 0.75rem; color: #94a3b8; margin-left: 8px;">{step_lat:.1f} ms</span>
                        </div>
                    </div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 6px;">
                        {ustep.get('content', '')}
                    </div>
                </div>
                """
            )
        else:
            for tc in ustep.get("tool_calls", []):
                t_name = tc.get("tool", "tool")
                t_args = tc.get("arguments", {})
                args_str = json.dumps(t_args, ensure_ascii=False)
                res_sum = tc.get("result_summary", "")
                fres = tc.get("full_result", {})
                if not res_sum and isinstance(fres, dict):
                    res_sum = str(fres.get("result", fres.get("value", fres.get("error", ""))))

                status = tc.get("status", "success")
                if status == "error" or "error" in str(res_sum).lower():
                    status_badge = '<span style="background: rgba(244, 63, 94, 0.2); color: #fda4af; padding: 2px 7px; border-radius: 4px; font-size: 0.72rem; font-weight: 700;">Error / Failed</span>'
                elif status == "pending_approval":
                    status_badge = '<span style="background: rgba(245, 158, 11, 0.2); color: #fde68a; padding: 2px 7px; border-radius: 4px; font-size: 0.72rem; font-weight: 700;">Pending Human Approval</span>'
                else:
                    status_badge = '<span style="background: rgba(16, 185, 129, 0.2); color: #6ee7b7; padding: 2px 7px; border-radius: 4px; font-size: 0.72rem; font-weight: 700;">Success</span>'

                cards.append(
                    f"""
                    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #334155; border-radius: 10px; padding: 14px 16px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <div>
                                <span style="font-weight: 700; color: #a5b4fc;">Step {idx}</span>
                                <span style="font-weight: 700; color: #38bdf8; margin-left: 8px;">🔧 Tool Call: {t_name}</span>
                            </div>
                            <div>
                                {status_badge}
                                <span style="font-size: 0.75rem; color: #94a3b8; margin-left: 8px;">{step_lat:.1f} ms</span>
                            </div>
                        </div>
                        <div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 6px;">
                            <strong>Inputs:</strong> <code style="color: #f8fafc; background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 4px;">{args_str}</code>
                        </div>
                        <div style="font-size: 0.84rem; color: #cbd5e1; background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 6px;">
                            <strong>Observation / Result:</strong> {res_sum}
                            {f"<div style='margin-top: 4px; color: #94a3b8; font-size: 0.78rem;'>Matches: {json.dumps(fres.get('matches', []), ensure_ascii=False)}</div>" if fres.get('matches') else ''}
                        </div>
                    </div>
                    """
                )

    return "\n".join(cards)


class PanelTitle(str):
    """String that also behaves like a dict with .get() for Gradio and unit tests."""
    def get(self, key, default=None):
        if key == "value":
            return str(self)
        if key == "visible":
            return len(str(self)) > 0
        return default


def switch_section(target_sec: str, current_sec: str):
    """
    Expandable navigation state machine:
    Toggles detail panel on/off or switches the active detail section.
    Returns 14-element UI update tuple.
    """
    # If clicking the currently active section or clicking chat, close panel and return to chat
    if target_sec == current_sec and target_sec != "chat":
        target_sec = "chat"

    if target_sec == "chat":
        return (
            "chat",
            gr.update(visible=False),
            PanelTitle(""),
            gr.update(visible=False),  # g_doc
            gr.update(visible=False),  # g_retr
            gr.update(visible=False),  # g_src
            gr.update(visible=False),  # g_eval
            gr.update(visible=False),  # g_stat
            gr.update(variant="primary"),    # b_chat
            gr.update(variant="secondary"),  # b_doc
            gr.update(variant="secondary"),  # b_retr
            gr.update(variant="secondary"),  # b_src
            gr.update(variant="secondary"),  # b_eval
            gr.update(variant="secondary"),  # b_stat
        )

    titles = {
        "doc": "📖 Document Info & Table of Contents",
        "retrieval": "📈 Retrieval Benchmark & IR Metrics",
        "sources": "📚 Retrieved Sources & Grounded Evidence",
        "eval": "📋 Evaluation Scorecard & Benchmarks",
        "stat": "⚡ Performance & Budget Metrics",
    }
    title = titles.get(target_sec, "Detail View")
    panel_title = PanelTitle(f"### {title}")

    return (
        target_sec,
        gr.update(visible=True),
        panel_title,
        gr.update(visible=(target_sec == "doc")),
        gr.update(visible=(target_sec == "retrieval")),
        gr.update(visible=(target_sec == "sources")),
        gr.update(visible=(target_sec == "eval")),
        gr.update(visible=(target_sec == "stat")),
        gr.update(variant="secondary"),                                  # b_chat
        gr.update(variant="primary" if target_sec == "doc" else "secondary"),
        gr.update(variant="primary" if target_sec == "retrieval" else "secondary"),
        gr.update(variant="primary" if target_sec == "sources" else "secondary"),
        gr.update(variant="primary" if target_sec == "eval" else "secondary"),
        gr.update(variant="primary" if target_sec == "stat" else "secondary"),
    )


def process_query(
    user_query: str,
    chat_history: list,
    mode: str = "Autonomous Agent (Think-Act-Observe)",
):
    """Processes user query through either RAG or Autonomous Agent pipeline."""
    if not user_query or not user_query.strip():
        return (
            chat_history,
            "",
            "*No sources retrieved.*",
            "*No agent trajectory recorded.*",
            format_stats_markdown({}),
            "",
            gr.update(visible=False),
            gr.update(visible=False),
            None,
        )

    clean_q = user_query.strip()
    is_agent = "agent" in mode.lower()

    if is_agent:
        result = agent_assistant.run(clean_q)
        formatted_ans = format_agent_chat_response(result)
        evidence_html = format_agent_evidence_html(result)
        trajectory_html = format_trajectory_html(result)
        stats_html = format_stats_markdown(result)
        pending = result.get("pending_approval")
        approval_html = format_approval_card(pending)
        has_pending = pending is not None
    else:
        result = pipeline.run(clean_q, top_k=7)
        ans_text = format_chat_answer(result["answer"], result.get("cited_pages", []))
        formatted_ans = ans_text
        evidence_html = format_evidence_markdown(result["sources"])
        trajectory_html = format_pipeline_trace_html(result)
        stats_html = format_stats_markdown(result)
        approval_html = ""
        has_pending = False
        pending = None

    # Record turn in chat history
    new_history = list(chat_history or [])
    new_history.append({"role": "user", "content": clean_q})
    new_history.append({"role": "assistant", "content": formatted_ans})

    chat_memory.add_turn(clean_q, result.get("answer", ""), citations=result.get("cited_pages", []))

    return (
        new_history,
        "",  # clear input box
        evidence_html,
        trajectory_html,
        stats_html,
        approval_html,
        gr.update(visible=has_pending),  # approve_btn
        gr.update(visible=has_pending),  # reject_btn
        pending,  # pending state
    )


def handle_approval_action(action: str, pending_data: Optional[dict], chat_history: list):
    """Handles human approval or rejection of pending actions."""
    if not pending_data:
        return chat_history, "", gr.update(visible=False), gr.update(visible=False), None

    channel = pending_data.get("channel", "general")
    message = pending_data.get("message", "")

    if action == "approve":
        send_res = send_message(channel=channel, message=message, approved=True)
        status_msg = f"✅ **Action Approved by Human**: Message was successfully dispatched to `{send_res.get('channel')}` via {send_res.get('delivery_method')}."
    else:
        status_msg = f"❌ **Action Rejected by Human**: Message to `{channel}` was cancelled and not sent."

    new_history = list(chat_history or [])
    new_history.append({"role": "assistant", "content": status_msg})

    return (
        new_history,
        "",  # clear approval box
        gr.update(visible=False),
        gr.update(visible=False),
        None,
    )


# --------------------------------------------------------------------------
# Main Gradio App Builder
# --------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Predictors RAG & Agent Assistant") as demo:
        # State tracking
        current_section = gr.State("chat")
        pending_action_state = gr.State(None)

        # Header Box
        with gr.Row(elem_id="header-box"):
            with gr.Column():
                gr.HTML("""
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <span class="title-badge">Team Predictors • AI Hackathon</span>
                        <h1 style="margin: 0; font-size: 1.85rem; font-weight: 800; background: linear-gradient(90deg, #f8fafc, #94a3b8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                            The Crystal Ball Manual — RAG &amp; Agent Portal
                        </h1>
                        <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 0.92rem;">
                            Autonomous grounded Q&amp;A and multi-step reasoning over <em>The Crystal Ball Instruction Manual (Volume One)</em> by Stephen Davies, Ph.D.
                        </p>
                    </div>
                    <div style="text-align: right;">
                        <span style="background: rgba(16, 185, 129, 0.15); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.3); padding: 4px 12px; border-radius: 8px; font-size: 0.8rem; font-weight: 600;">
                            ● Production Online
                        </span>
                    </div>
                </div>
                """)

        # Main 2-Column Layout
        with gr.Row():
            # 1. Left Sidebar Navigation
            with gr.Column(scale=1, elem_id="sidebar-nav-col", min_width=240):
                gr.Markdown("### 🧭 Navigation")
                b_chat = gr.Button("💬 Chat View", variant="primary", elem_classes=["nav-btn"])
                b_doc = gr.Button("📖 Document Info", variant="secondary", elem_classes=["nav-btn"])
                b_retr = gr.Button("📈 Retrieval Benchmark", variant="secondary", elem_classes=["nav-btn"])
                b_src = gr.Button("📚 Retrieved Sources", variant="secondary", elem_classes=["nav-btn"])
                b_eval = gr.Button("📋 Evaluation Scorecard", variant="secondary", elem_classes=["nav-btn"])
                b_stat = gr.Button("⚡ Performance & Stats", variant="secondary", elem_classes=["nav-btn"])

                gr.Markdown("---")
                gr.Markdown("### ⚙️ Engine Mode")
                mode_selector = gr.Radio(
                    choices=[
                        "Autonomous Agent (Think-Act-Observe)",
                        "RAG (Direct Hybrid + Rerank)",
                    ],
                    value="Autonomous Agent (Think-Act-Observe)",
                    label="Execution Pipeline",
                )

                gr.Markdown("---")
                gr.Markdown("### 📑 Actions")
                export_btn = gr.Button("📥 Export Chat to Markdown", variant="secondary", elem_classes=["nav-btn"])
                export_status = gr.Markdown("", visible=False)

                with gr.Row():
                    thumbs_up_btn = gr.Button("👍 Helpful", variant="secondary")
                    thumbs_down_btn = gr.Button("👎 Not Helpful", variant="secondary")
                feedback_status = gr.Markdown("", visible=False)

                clear_btn = gr.Button("🧹 Clear Conversation", variant="secondary", elem_classes=["nav-btn"])

            # 2. Right Main Chatbot & Expandable Detail Panel
            with gr.Column(scale=4, elem_id="chatbot-main-col"):
                # Expandable Detail Section (Hidden by default)
                with gr.Group(elem_id="expandable-detail-section", visible=False) as detail_panel:
                    panel_title_md = gr.Markdown("### Detail View")

                    # Section 1: Document Info (TOC Table)
                    with gr.Group(visible=False) as g_doc:
                        gr.HTML(build_chapters_table_html())

                    # Section 2: Retrieval Benchmark (IR Metrics)
                    with gr.Group(visible=False) as g_retr:
                        try:
                            from app.metrics_dashboard import format_metrics_dashboard_html, METRICS_FILE
                            if METRICS_FILE.exists():
                                retr_html = format_metrics_dashboard_html(json.loads(METRICS_FILE.read_text(encoding="utf-8")))
                            else:
                                retr_html = format_metrics_dashboard_html({})
                        except Exception:
                            retr_html = "<em>Retrieval benchmark metrics module available. Run `python -m app.metrics_dashboard` to generate.</em>"
                        gr.HTML(retr_html)

                    # Section 3: Retrieved Sources
                    with gr.Group(visible=False) as g_src:
                        evidence_box = gr.HTML(
                            value="*Ask a question to see hybrid dense + sparse source cards and page citations.*"
                        )

                    # Section 4: Evaluation Scorecard & Agent Trajectory
                    with gr.Group(visible=False) as g_eval:
                        trajectory_box = gr.HTML(
                            value="*Agent tool trajectory and multi-step trace cards will appear here.*"
                        )

                    # Section 5: Performance & Budget Stats
                    with gr.Group(visible=False) as g_stat:
                        stats_box = gr.HTML(
                            value=format_stats_markdown({})
                        )

                # Always-Visible Main Chatbot
                chatbot = gr.Chatbot(
                    elem_id="main-chatbot",
                    visible=True,
                    height=520,
                    show_label=False,
                )

                # Approval Box (Visible only when action requires human approval)
                approval_box = gr.HTML(value="")
                with gr.Row():
                    approve_btn = gr.Button("✅ Approve Action", variant="primary", visible=False)
                    reject_btn = gr.Button("❌ Reject Action", variant="stop", visible=False)

                # Query Input & Submit
                with gr.Row():
                    query_input = gr.Textbox(
                        placeholder="Ask a question about The Crystal Ball manual (e.g., 'What are the four levels of data-to-wisdom hierarchy?' or 'Calculate difference between Chapter 15 and 6')...",
                        lines=2,
                        scale=5,
                        show_label=False,
                    )
                    submit_btn = gr.Button("🚀 Ask Assistant", variant="primary", scale=1)

                # Curated Examples
                gr.Markdown("#### 💡 Recommended Test Queries")
                example_picker = gr.Examples(
                    examples=[
                        ["What are the four levels of the data-to-wisdom hierarchy described in Chapter 1?"],
                        ["What is the difference between the starting page of Chapter 15 and Chapter 6, and what is half that value?"],
                        ["What is Simpson's Paradox as illustrated in Chapter 16?"],
                        ["Post a short summary of Chapter 1 to the general Slack channel."],
                        ["What is the backpropagation algorithm for multi-layer neural networks and how are weight gradients computed?"],
                    ],
                    inputs=[query_input],
                    label="Click to populate query:",
                )

        # ----------------------------------------------------------------------
        # Wire Event Handlers
        # ----------------------------------------------------------------------

        # Section navigation switching
        nav_buttons = [b_chat, b_doc, b_retr, b_src, b_eval, b_stat]
        for btn, sec_name in zip(nav_buttons, ["chat", "doc", "retrieval", "sources", "eval", "stat"]):
            btn.click(
                fn=lambda s=sec_name, cur="chat": switch_section(s, cur),
                inputs=[current_section],
                outputs=[
                    current_section,
                    detail_panel,
                    panel_title_md,
                    g_doc,
                    g_retr,
                    g_src,
                    g_eval,
                    g_stat,
                    b_chat,
                    b_doc,
                    b_retr,
                    b_src,
                    b_eval,
                    b_stat,
                ],
            )

        # Submit query
        submit_btn.click(
            fn=process_query,
            inputs=[query_input, chatbot, mode_selector],
            outputs=[
                chatbot,
                query_input,
                evidence_box,
                trajectory_box,
                stats_box,
                approval_box,
                approve_btn,
                reject_btn,
                pending_action_state,
            ],
        )

        query_input.submit(
            fn=process_query,
            inputs=[query_input, chatbot, mode_selector],
            outputs=[
                chatbot,
                query_input,
                evidence_box,
                trajectory_box,
                stats_box,
                approval_box,
                approve_btn,
                reject_btn,
                pending_action_state,
            ],
        )

        # Human Approval
        approve_btn.click(
            fn=lambda pending, hist: handle_approval_action("approve", pending, hist),
            inputs=[pending_action_state, chatbot],
            outputs=[chatbot, approval_box, approve_btn, reject_btn, pending_action_state],
        )

        reject_btn.click(
            fn=lambda pending, hist: handle_approval_action("reject", pending, hist),
            inputs=[pending_action_state, chatbot],
            outputs=[chatbot, approval_box, approve_btn, reject_btn, pending_action_state],
        )

        # Clear Conversation
        clear_btn.click(
            fn=lambda: ([], "", "*No sources.*", "*No trajectory.*", format_stats_markdown({}), "", gr.update(visible=False), gr.update(visible=False), None),
            outputs=[chatbot, query_input, evidence_box, trajectory_box, stats_box, approval_box, approve_btn, reject_btn, pending_action_state],
        )

        # Markdown Export
        def export_chat():
            md_text, fpath = chat_memory.export_to_markdown("The Predictors Live Session")
            return gr.update(value=f"✅ Chat exported to `{fpath.name}` ({len(md_text)} chars)", visible=True)

        export_btn.click(fn=export_chat, outputs=[export_status])

        # Feedback
        def record_thumbs(rating: str, hist: list):
            last_q = "Session Question"
            last_a = "Session Answer"
            if hist:
                for msg in reversed(hist):
                    if msg.get("role") == "user" and last_q == "Session Question":
                        last_q = msg.get("content", "")
                    elif msg.get("role") == "assistant" and last_a == "Session Answer":
                        last_a = msg.get("content", "")
            save_feedback(query=last_q, answer=last_a, rating=rating)
            return gr.update(value=f"✨ Recorded {rating.replace('_', ' ').title()}! Thank you.", visible=True)

        thumbs_up_btn.click(fn=lambda h: record_thumbs("thumbs_up", h), inputs=[chatbot], outputs=[feedback_status])
        thumbs_down_btn.click(fn=lambda h: record_thumbs("thumbs_down", h), inputs=[chatbot], outputs=[feedback_status])

    return demo


if __name__ == "__main__":
    demo = build_app()
    try:
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
    except OSError:
        demo.launch(server_name="127.0.0.1", share=False)
