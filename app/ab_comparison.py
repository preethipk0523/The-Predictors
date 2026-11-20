"""
A/B Retrieval Comparison Module.
Enables side-by-side comparative evaluation of four distinct retrieval strategies
on the exact same user query:
A. FAISS / Vector-only retrieval
B. BM25 / Keyword-only retrieval
C. Existing Hybrid FAISS + BM25 retrieval
D. Hybrid + Cross-Encoder Reranking
"""

import time
import numpy as np
import faiss
from typing import List, Dict, Any, Set, Tuple
from app.config import (
    EMBEDDING_MODEL,
    RERANKER_TOP_INITIAL,
    RERANKER_TOP_FINAL,
    ENABLE_RERANKER,
)
from app.retriever import retriever, tokenize_for_bm25
from app.reranker import reranker
from app.cost_tracker import cost_tracker


def retrieve_vector_only(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Executes dense semantic vector-only search via FAISS."""
    embed_resp = retriever.client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
    cost_tracker.log_embedding(embed_resp.usage.total_tokens)
    query_vec = np.array([embed_resp.data[0].embedding], dtype=np.float32)
    faiss.normalize_L2(query_vec)

    sims, indices = retriever.index.search(query_vec, top_k)
    results = []
    for rank, (sim, idx) in enumerate(zip(sims[0], indices[0]), 1):
        if 0 <= idx < len(retriever.chunks):
            chunk = dict(retriever.chunks[idx])
            results.append({
                "rank": rank,
                "source": chunk.get("source", "crystal1 (1).pdf"),
                "page": chunk.get("page", 1),
                "score": round(float(sim), 4),
                "snippet": chunk.get("text", "")[:280] + ("..." if len(chunk.get("text", "")) > 280 else ""),
                "full_text": chunk.get("text", ""),
                "method": "Vector (FAISS)",
            })
    return results


def retrieve_bm25_only(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Executes sparse keyword-only search via Okapi BM25."""
    tokens = tokenize_for_bm25(query)
    scores = retriever.bm25.get_scores(tokens)
    max_score = max(scores) if max(scores) > 0 else 1.0
    norm_scores = scores / max_score
    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []
    for rank, idx in enumerate(top_indices, 1):
        if 0 <= idx < len(retriever.chunks):
            chunk = dict(retriever.chunks[idx])
            results.append({
                "rank": rank,
                "source": chunk.get("source", "crystal1 (1).pdf"),
                "page": chunk.get("page", 1),
                "score": round(float(norm_scores[idx]), 4),
                "raw_bm25_score": round(float(scores[idx]), 4),
                "snippet": chunk.get("text", "")[:280] + ("..." if len(chunk.get("text", "")) > 280 else ""),
                "full_text": chunk.get("text", ""),
                "method": "BM25 (Keyword)",
            })
    return results


def retrieve_hybrid(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Executes the standard production Hybrid (FAISS + BM25 + RRF) retrieval."""
    chunks = retriever.retrieve(query, top_k=top_k)
    results = []
    for rank, c in enumerate(chunks, 1):
        results.append({
            "rank": rank,
            "source": c.get("source", "crystal1 (1).pdf"),
            "page": c.get("page", 1),
            "score": round(c.get("combined_score", 0.0), 4),
            "dense_score": c.get("dense_score", 0.0),
            "bm25_score": c.get("bm25_score", 0.0),
            "snippet": c.get("text", "")[:280] + ("..." if len(c.get("text", "")) > 280 else ""),
            "full_text": c.get("text", ""),
            "method": "Hybrid (FAISS + BM25)",
        })
    return results


def retrieve_hybrid_reranked(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Executes Hybrid retrieval followed by Cross-Encoder reranking."""
    initial_chunks = retriever.retrieve(query, top_k=max(top_k * 2, 10))
    reranked_chunks = reranker.rerank(query, initial_chunks, top_n=top_k)
    results = []
    for rank, c in enumerate(reranked_chunks, 1):
        results.append({
            "rank": rank,
            "initial_rank": c.get("initial_rank", rank),
            "source": c.get("source", "crystal1 (1).pdf"),
            "page": c.get("page", 1),
            "score": round(c.get("rerank_score", c.get("combined_score", 0.0)), 4),
            "initial_combined_score": round(c.get("combined_score", 0.0), 4),
            "snippet": c.get("text", "")[:280] + ("..." if len(c.get("text", "")) > 280 else ""),
            "full_text": c.get("text", ""),
            "method": "Hybrid + Cross-Encoder",
        })
    return results


def run_retrieval_ab_comparison(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Runs all four retrieval strategies on the same query and compiles comparative statistics.
    """
    clean_q = query.strip()
    if not clean_q:
        return {"query": "", "methods": {}, "statistics": {}, "timing_ms": {}}

    t0 = time.time()
    vector_res = retrieve_vector_only(clean_q, top_k=top_k)
    t_v = round((time.time() - t0) * 1000, 1)

    t1 = time.time()
    bm25_res = retrieve_bm25_only(clean_q, top_k=top_k)
    t_b = round((time.time() - t1) * 1000, 1)

    t2 = time.time()
    hybrid_res = retrieve_hybrid(clean_q, top_k=top_k)
    t_h = round((time.time() - t2) * 1000, 1)

    t3 = time.time()
    reranked_res = retrieve_hybrid_reranked(clean_q, top_k=top_k)
    t_r = round((time.time() - t3) * 1000, 1)

    # Collect retrieved page sets
    v_pages: Set[int] = {c["page"] for c in vector_res}
    b_pages: Set[int] = {c["page"] for c in bm25_res}
    h_pages: Set[int] = {c["page"] for c in hybrid_res}
    r_pages: Set[int] = {c["page"] for c in reranked_res}

    all_unique_pages = v_pages | b_pages | h_pages | r_pages

    # Compute overlaps
    v_b_overlap = len(v_pages & b_pages)
    v_h_overlap = len(v_pages & h_pages)
    b_h_overlap = len(b_pages & h_pages)
    h_r_overlap = len(h_pages & r_pages)

    # Calculate average scores
    avg_v = round(float(np.mean([c["score"] for c in vector_res])), 3) if vector_res else 0.0
    avg_b = round(float(np.mean([c["score"] for c in bm25_res])), 3) if bm25_res else 0.0
    avg_h = round(float(np.mean([c["score"] for c in hybrid_res])), 3) if hybrid_res else 0.0
    avg_r = round(float(np.mean([c["score"] for c in reranked_res])), 3) if reranked_res else 0.0

    statistics = {
        "vector_avg_score": avg_v,
        "bm25_avg_score": avg_b,
        "hybrid_avg_score": avg_h,
        "reranked_avg_score": avg_r,
        "vector_top_page": vector_res[0]["page"] if vector_res else None,
        "bm25_top_page": bm25_res[0]["page"] if bm25_res else None,
        "hybrid_top_page": hybrid_res[0]["page"] if hybrid_res else None,
        "reranked_top_page": reranked_res[0]["page"] if reranked_res else None,
        "vector_bm25_overlap_count": v_b_overlap,
        "vector_hybrid_overlap_count": v_h_overlap,
        "bm25_hybrid_overlap_count": b_h_overlap,
        "hybrid_reranked_overlap_count": h_r_overlap,
        "total_unique_pages": len(all_unique_pages),
        "total_time_ms": round((time.time() - t0) * 1000, 1),
    }

    return {
        "query": clean_q,
        "top_k": top_k,
        "methods": {
            "vector": vector_res,
            "bm25": bm25_res,
            "hybrid": hybrid_res,
            "reranked": reranked_res,
        },
        "pages": {
            "vector": sorted(list(v_pages)),
            "bm25": sorted(list(b_pages)),
            "hybrid": sorted(list(h_pages)),
            "reranked": sorted(list(r_pages)),
        },
        "statistics": statistics,
        "timing_ms": {
            "vector": t_v,
            "bm25": t_b,
            "hybrid": t_h,
            "reranked": t_r,
        },
    }


def format_ab_comparison_html(ab_data: Dict[str, Any]) -> str:
    """Renders the A/B retrieval results and statistics into a clean, modern HTML layout."""
    if not ab_data or not ab_data.get("methods"):
        return "<p style='color: #94a3b8; font-style: italic;'>Click 'Run Retrieval A/B Comparison' to compare retrieval methods for this question.</p>"

    query = ab_data.get("query", "")
    stats = ab_data.get("statistics", {})
    methods = ab_data.get("methods", {})
    timings = ab_data.get("timing_ms", {})

    # Overview Metrics Table
    summary_card = f"""
    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 12px; padding: 14px 18px; margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 10px;">
            <div style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">
                ⚖️ Comparative Retrieval Statistics for: <span style="color: #38bdf8;">"{query}"</span>
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8;">
                Total Query Latency: <strong style="color: #6ee7b7;">{stats.get('total_time_ms', 0):.1f} ms</strong>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
            <div style="background: rgba(30, 41, 59, 0.6); padding: 8px 12px; border-radius: 8px; border-left: 3px solid #38bdf8;">
                <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 600;">VECTOR (FAISS)</div>
                <div style="font-size: 0.9rem; font-weight: 700; color: #e2e8f0;">Top: Page {stats.get('vector_top_page', '?')}</div>
                <div style="font-size: 0.75rem; color: #94a3b8;">Avg Score: {stats.get('vector_avg_score', 0.0):.2f} • {timings.get('vector', 0):.0f}ms</div>
            </div>

            <div style="background: rgba(30, 41, 59, 0.6); padding: 8px 12px; border-radius: 8px; border-left: 3px solid #f59e0b;">
                <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 600;">KEYWORD (BM25)</div>
                <div style="font-size: 0.9rem; font-weight: 700; color: #e2e8f0;">Top: Page {stats.get('bm25_top_page', '?')}</div>
                <div style="font-size: 0.75rem; color: #94a3b8;">Avg Score: {stats.get('bm25_avg_score', 0.0):.2f} • {timings.get('bm25', 0):.0f}ms</div>
            </div>

            <div style="background: rgba(30, 41, 59, 0.6); padding: 8px 12px; border-radius: 8px; border-left: 3px solid #10b981;">
                <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 600;">HYBRID (FAISS+BM25)</div>
                <div style="font-size: 0.9rem; font-weight: 700; color: #e2e8f0;">Top: Page {stats.get('hybrid_top_page', '?')}</div>
                <div style="font-size: 0.75rem; color: #94a3b8;">Avg Score: {stats.get('hybrid_avg_score', 0.0):.2f} • {timings.get('hybrid', 0):.0f}ms</div>
            </div>

            <div style="background: rgba(30, 41, 59, 0.6); padding: 8px 12px; border-radius: 8px; border-left: 3px solid #a855f7;">
                <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 600;">HYBRID + RERANKER</div>
                <div style="font-size: 0.9rem; font-weight: 700; color: #e2e8f0;">Top: Page {stats.get('reranked_top_page', '?')}</div>
                <div style="font-size: 0.75rem; color: #94a3b8;">Avg Score: {stats.get('reranked_avg_score', 0.0):.2f} • {timings.get('reranked', 0):.0f}ms</div>
            </div>
        </div>

        <div style="margin-top: 10px; font-size: 0.78rem; color: #cbd5e1; display: flex; gap: 14px; flex-wrap: wrap;">
            <span>🔄 <strong>Vector &amp; BM25 Overlap:</strong> {stats.get('vector_bm25_overlap_count', 0)} pages</span>
            <span>🤝 <strong>Vector &amp; Hybrid Overlap:</strong> {stats.get('vector_hybrid_overlap_count', 0)} pages</span>
            <span>🎯 <strong>BM25 &amp; Hybrid Overlap:</strong> {stats.get('bm25_hybrid_overlap_count', 0)} pages</span>
            <span>⚡ <strong>Hybrid &amp; Rerank Overlap:</strong> {stats.get('hybrid_reranked_overlap_count', 0)} pages</span>
            <span>🌐 <strong>Unique Pages Discovered:</strong> {stats.get('total_unique_pages', 0)}</span>
        </div>
    </div>
    """

    def render_column(title: str, color_hex: str, chunks: List[Dict[str, Any]]) -> str:
        cards_html = []
        for c in chunks:
            rank = c.get("rank", 1)
            page = c.get("page", "?")
            score = c.get("score", 0.0)
            snip = c.get("snippet", "")
            init_rank = c.get("initial_rank")
            rank_shift_badge = f'<span style="background: rgba(168,85,247,0.2); color: #c084fc; font-size: 0.68rem; padding: 1px 5px; border-radius: 4px;">Initial #{init_rank}</span>' if init_rank and init_rank != rank else ""

            card = f"""
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 8px; padding: 8px 10px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <span style="font-weight: 800; color: {color_hex}; font-size: 0.78rem;">#{rank}</span>
                        <span style="background: rgba(16, 185, 129, 0.15); color: #6ee7b7; font-size: 0.72rem; padding: 1px 6px; border-radius: 4px; font-weight: 600;">Page {page}</span>
                        {rank_shift_badge}
                    </div>
                    <span style="font-size: 0.72rem; color: #94a3b8; font-family: monospace;">Score: {score:.3f}</span>
                </div>
                <div style="font-size: 0.76rem; color: #cbd5e1; line-height: 1.35; font-style: italic; background: rgba(0,0,0,0.2); padding: 6px 8px; border-radius: 4px;">
                    "{snip}"
                </div>
            </div>
            """
            cards_html.append(card)

        return f"""
        <div style="flex: 1; min-width: 240px; background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 10px;">
            <div style="font-weight: 700; color: {color_hex}; font-size: 0.85rem; margin-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 4px;">
                {title}
            </div>
            {''.join(cards_html) if cards_html else '<p style="color: #64748b; font-size: 0.75rem;">No results</p>'}
        </div>
        """

    col_vector = render_column("A. Vector-only (FAISS)", "#38bdf8", methods.get("vector", []))
    col_bm25 = render_column("B. Keyword-only (BM25)", "#f59e0b", methods.get("bm25", []))
    col_hybrid = render_column("C. Hybrid (FAISS + BM25)", "#10b981", methods.get("hybrid", []))
    col_rerank = render_column("D. Hybrid + Reranker", "#a855f7", methods.get("reranked", []))

    columns_grid = f"""
    <div style="display: flex; gap: 12px; flex-wrap: wrap;">
        {col_vector}
        {col_bm25}
        {col_hybrid}
        {col_rerank}
    </div>
    """

    return summary_card + columns_grid
