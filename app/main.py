"""
Gradio Web Interface for Predictors RAG Assistant.
Provides a modern, responsive UI with source citations, retrieved evidence inspection,
and real-time cost and performance metrics.
"""

import sys
from pathlib import Path

# Ensure root workspace is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
from app.pipeline import pipeline
from app.cost_tracker import cost_tracker

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
    padding: 24px 32px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 0 15px rgba(99, 102, 241, 0.15);
    backdrop-filter: blur(12px);
}

.title-badge {
    display: inline-block;
    background: linear-gradient(90deg, #6366f1, #06b6d4);
    color: white;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 4px 12px;
    border-radius: 9999px;
    margin-bottom: 8px;
}

.stat-card {
    background: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    backdrop-filter: blur(8px);
}

.evidence-card {
    background: rgba(15, 23, 42, 0.6) !important;
    border-left: 4px solid #6366f1 !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    margin-bottom: 12px !important;
    transition: all 0.2s ease;
}

.evidence-card:hover {
    border-left-color: #06b6d4 !important;
    background: rgba(30, 41, 59, 0.8) !important;
}

.score-badge {
    font-size: 0.75rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 6px;
    background: rgba(99, 102, 241, 0.2);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.3);
}

.page-badge {
    font-size: 0.75rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 6px;
    background: rgba(16, 185, 129, 0.2);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

button.primary-btn {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4) !important;
}

button.primary-btn:hover {
    background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
    transform: translateY(-1px);
}
"""


def format_evidence_markdown(sources: list) -> str:
    """Formats retrieved sources into rich Markdown cards."""
    if not sources:
        return "*No sources retrieved.*"

    md_blocks = []
    for i, s in enumerate(sources, 1):
        doc = s.get("source", "crystal1 (1).pdf")
        page = s.get("page", "?")
        dense = s.get("dense_score", 0.0)
        bm25 = s.get("bm25_score", 0.0)
        comb = s.get("combined_score", 0.0)
        snippet = s.get("text_snippet", "")

        card = f"""
<div class="evidence-card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <div>
            <span class="page-badge">📄 Page {page}</span>
            <span style="font-size: 0.8rem; color: #94a3b8; margin-left: 8px;">{doc}</span>
        </div>
        <div>
            <span class="score-badge" title="Cosine Similarity">Dense: {dense:.2f}</span>
            <span class="score-badge" title="BM25 Relevance">BM25: {bm25:.2f}</span>
            <span class="score-badge" style="background: rgba(16, 185, 129, 0.15); color: #6ee7b7;" title="Combined Hybrid Score">Score: {comb:.2f}</span>
        </div>
    </div>
    <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.45; font-style: italic; background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 6px;">
        "{snippet}"
    </div>
</div>
"""
        md_blocks.append(card)

    return "\n".join(md_blocks)


def format_stats_markdown(result: dict) -> str:
    """Formats latency and cost metrics into clean summary HTML."""
    summary = cost_tracker.get_summary()
    tot_cost = summary["total_cost_usd"]
    budget_pct = summary["budget_used_percent"]
    q_cost = result.get("cost_usd", 0.0)
    ret_ms = result.get("retrieval_time_ms", 0)
    gen_ms = result.get("generation_time_ms", 0)
    tot_ms = result.get("total_time_ms", 0)
    tokens = result.get("total_tokens", 0)

    return f"""
<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; font-size: 0.85rem;">
    <div style="background: rgba(15,23,42,0.5); padding: 10px; border-radius: 8px; border: 1px solid #334155;">
        <span style="color: #94a3b8;">⏱️ Retrieval Latency</span><br>
        <strong style="color: #6ee7b7; font-size: 1.05rem;">{ret_ms:.1f} ms</strong>
    </div>
    <div style="background: rgba(15,23,42,0.5); padding: 10px; border-radius: 8px; border: 1px solid #334155;">
        <span style="color: #94a3b8;">⚡ Generation Latency</span><br>
        <strong style="color: #38bdf8; font-size: 1.05rem;">{gen_ms:.1f} ms</strong>
    </div>
    <div style="background: rgba(15,23,42,0.5); padding: 10px; border-radius: 8px; border: 1px solid #334155;">
        <span style="color: #94a3b8;">🪙 Query Tokens & Cost</span><br>
        <strong style="color: #facc15; font-size: 1.05rem;">{tokens} tok / ${q_cost:.5f}</strong>
    </div>
    <div style="background: rgba(15,23,42,0.5); padding: 10px; border-radius: 8px; border: 1px solid #334155;">
        <span style="color: #94a3b8;">💳 Total Spend ($5 Budget)</span><br>
        <strong style="color: #a78bfa; font-size: 1.05rem;">${tot_cost:.4f} ({budget_pct:.2f}%)</strong>
    </div>
</div>
"""


def process_query(user_query: str):
    if not user_query or not user_query.strip():
        return (
            "Please enter a question about *The Crystal Ball Instruction Manual*.",
            "*No sources retrieved.*",
            "*No metrics available.*",
        )

    result = pipeline.run(user_query.strip(), top_k=7)
    answer_text = result["answer"]
    evidence_html = format_evidence_markdown(result["sources"])
    stats_html = format_stats_markdown(result)

    return answer_text, evidence_html, stats_html


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Predictors RAG | Crystal Ball") as demo:
        with gr.Row(elem_id="header-box"):
            with gr.Column():
                gr.HTML("""
                <div>
                    <span class="title-badge">Team Predictors • Hackathon RAG Application</span>
                    <h1 style="margin: 0; font-size: 1.85rem; font-weight: 800; background: linear-gradient(90deg, #f8fafc, #94a3b8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                        The Crystal Ball Data Science Assistant
                    </h1>
                    <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 0.95rem;">
                        Autonomous grounded Q&amp;A over <em>The Crystal Ball Instruction Manual (Volume One: Introduction to Data Science)</em> by Stephen Davies, Ph.D.
                    </p>
                </div>
                """)

        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Ask a question about the corpus:",
                    placeholder="e.g., What are the four levels of the data-to-wisdom hierarchy?",
                    lines=2,
                    elem_id="query-box",
                )

                with gr.Row():
                    submit_btn = gr.Button("🔍 Search & Generate", variant="primary", elem_classes=["primary-btn"])
                    clear_btn = gr.Button("Clear", variant="secondary")

                answer_output = gr.Markdown(
                    label="Assistant Response",
                    value="*Ask a question above or choose from the curated examples below to test the RAG pipeline.*",
                )

                gr.Markdown("### 💡 Recommended Test Queries")
                example_picker = gr.Examples(
                    examples=[
                        # Priority Questions
                        ["What are the four levels of the data-to-wisdom hierarchy described in Chapter 1?"],
                        ["How does the conceptual idea of an associative array in Chapter 7 transition into Pandas Series in Chapter 11?"],
                        ["How does the author explain the difference between a predictor feature and a target variable in classification?"],
                        ["What is the backpropagation algorithm for multi-layer neural networks and how are weight gradients computed?"],  # Out-of-scope priority
                        ["How do you evaluate performance?"],  # Ambiguous priority
                        # Stress Tests & Factual
                        ["What informal nickname does the author use for parentheses () throughout the book?"],
                        ["What is Simpson's Paradox as illustrated in Chapter 16?"],
                        ["Does the Celsius temperature scale represent a ratio scale of measurement?"],
                        ["How do consecutive sections in Chapter 15 contrast the mean and median regarding their sensitivity to outliers?"],
                        ["Who won the 2024 ICC Men's T20 Cricket World Cup?"],  # General out-of-scope
                    ],
                    inputs=[query_input],
                    label="Click any example to populate:",
                )

            with gr.Column(scale=2):
                gr.Markdown("### 📚 Retrieved Corpus Evidence")
                evidence_box = gr.HTML(
                    value="*Evidence cards with document names, page citations, and hybrid relevance scores will appear here.*"
                )

                gr.Markdown("### 📊 Performance & Budget Tracker")
                stats_box = gr.HTML(
                    value=format_stats_markdown({})
                )

        submit_btn.click(
            fn=process_query,
            inputs=[query_input],
            outputs=[answer_output, evidence_box, stats_box],
        )
        query_input.submit(
            fn=process_query,
            inputs=[query_input],
            outputs=[answer_output, evidence_box, stats_box],
        )
        clear_btn.click(
            fn=lambda: ("", "*Enter a question to start.*", "*No sources.*", format_stats_markdown({})),
            outputs=[query_input, answer_output, evidence_box, stats_box],
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    try:
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False, css=CUSTOM_CSS)
    except OSError:
        demo.launch(server_name="127.0.0.1", share=False, css=CUSTOM_CSS)
