"""
RAG Metrics Dashboard Module.
Computes standard Information Retrieval (IR) evaluation metrics over a curated textbook benchmark:
- Recall@1
- Recall@3
- Recall@5
- MRR (Mean Reciprocal Rank)

Evaluates four distinct retrieval strategies:
1. FAISS / Vector-only
2. BM25 / Keyword-only
3. Hybrid (FAISS + BM25 + RRF)
4. Hybrid + Cross-Encoder Reranking
Saves and reproduces benchmark results in data/retrieval_benchmark_metrics.json.
"""

import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.config import DATA_DIR
from app.ab_comparison import (
    retrieve_vector_only,
    retrieve_bm25_only,
    retrieve_hybrid,
    retrieve_hybrid_reranked,
)

METRICS_FILE = Path(DATA_DIR) / "retrieval_benchmark_metrics.json"

# Curated, representative 16-question evaluation dataset from "The Crystal Ball Instruction Manual"
EVALUATION_DATASET: List[Dict[str, Any]] = [
    {
        "id": 1,
        "question": "What are the four levels of the data-to-wisdom hierarchy described in Chapter 1?",
        "expected_pages": [8, 9, 10],
        "category": "factual",
        "concept": "DIKW Hierarchy",
    },
    {
        "id": 2,
        "question": "What are the three kinds of atomic data described in Chapter 3?",
        "expected_pages": [19, 20, 21],
        "category": "factual",
        "concept": "Atomic Data Types",
    },
    {
        "id": 3,
        "question": "What is the fundamental difference between interval and ratio scales of measurement according to Chapter 6?",
        "expected_pages": [44, 54, 55],
        "category": "factual",
        "concept": "Stevens Scales of Measurement",
    },
    {
        "id": 4,
        "question": "What are the three aggregate data types introduced in Chapter 7?",
        "expected_pages": [59, 60],
        "category": "factual",
        "concept": "Aggregate Data Types",
    },
    {
        "id": 5,
        "question": "How does an associative array relate to a Pandas Series in Python?",
        "expected_pages": [59, 109, 110],
        "category": "cross-reference",
        "concept": "Series Data Structure",
    },
    {
        "id": 6,
        "question": "What informal nickname does the author use for parentheses () throughout the book?",
        "expected_pages": [34, 62, 63],
        "category": "stress-test",
        "concept": "Syntax Nicknames (Bananas)",
    },
    {
        "id": 7,
        "question": "What is Simpson's Paradox as illustrated with the Simpson family example?",
        "expected_pages": [194, 195, 196],
        "category": "factual",
        "concept": "Simpson's Paradox",
    },
    {
        "id": 8,
        "question": "What is the difference between univariate EDA in Chapter 15 and bivariate EDA in Chapter 19?",
        "expected_pages": [145, 146, 193, 194],
        "category": "cross-reference",
        "concept": "Univariate vs Bivariate EDA",
    },
    {
        "id": 9,
        "question": "How is a confusion matrix structured and what are false positives and false negatives?",
        "expected_pages": [287, 288, 289],
        "category": "factual",
        "concept": "Evaluation Metrics",
    },
    {
        "id": 10,
        "question": "How does a decision tree determine the best split using Gini impurity or purity?",
        "expected_pages": [270, 271, 272],
        "category": "factual",
        "concept": "Decision Tree Splitting",
    },
    {
        "id": 11,
        "question": "What is the difference between supervised and unsupervised learning according to Chapter 25?",
        "expected_pages": [256, 257],
        "category": "factual",
        "concept": "Machine Learning Paradigms",
    },
    {
        "id": 12,
        "question": "What is the difference between a predictor feature variable and a target variable?",
        "expected_pages": [262, 263],
        "category": "factual",
        "concept": "Classification Concepts",
    },
    {
        "id": 13,
        "question": "How do nominal and ordinal measurement scales differ in Stevens hierarchy?",
        "expected_pages": [44, 46, 47],
        "category": "factual",
        "concept": "Nominal vs Ordinal Scales",
    },
    {
        "id": 14,
        "question": "Where is the Table of Contents located in the instruction manual?",
        "expected_pages": [5, 6],
        "category": "stress-test",
        "concept": "Document Outline",
    },
    {
        "id": 15,
        "question": "What are leaf or terminal nodes in a classification decision tree?",
        "expected_pages": [272, 273],
        "category": "factual",
        "concept": "Decision Tree Terminology",
    },
    {
        "id": 16,
        "question": "What does a ratio scale possess that an interval scale does not?",
        "expected_pages": [54, 55],
        "category": "factual",
        "concept": "True Zero Point",
    },
]


def calculate_query_metrics(
    retrieved_pages: List[int],
    expected_pages: List[int],
    max_k: int = 5,
) -> Dict[str, Any]:
    """
    Computes Hit@1, Hit@3, Hit@5 and Reciprocal Rank for a single query.

    Relevance Definition:
    A retrieved chunk is relevant if its physical textbook page is one of the expected
    ground-truth pages designated for the question.
    """
    exp_set = set(expected_pages)
    top1 = retrieved_pages[:1]
    top3 = retrieved_pages[:3]
    top5 = retrieved_pages[:5]

    hit_1 = 1.0 if any(p in exp_set for p in top1) else 0.0
    hit_3 = 1.0 if any(p in exp_set for p in top3) else 0.0
    hit_5 = 1.0 if any(p in exp_set for p in top5) else 0.0

    # Reciprocal Rank: 1 / rank of first relevant page in top 5, or 0.0
    reciprocal_rank = 0.0
    for rank, p in enumerate(top5, 1):
        if p in exp_set:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "recall_at_1": hit_1,
        "recall_at_3": hit_3,
        "recall_at_5": hit_5,
        "reciprocal_rank": reciprocal_rank,
    }


def run_benchmark_evaluation(dataset: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Runs the full evaluation benchmark across all 4 retrieval strategies.
    Computes Recall@1, Recall@3, Recall@5, and MRR.
    Saves results to data/retrieval_benchmark_metrics.json.
    """
    eval_set = dataset or EVALUATION_DATASET
    n_queries = len(eval_set)

    methods = ["Vector (FAISS)", "BM25 (Keyword)", "Hybrid (FAISS+BM25)", "Hybrid + Reranker"]
    raw_scores: Dict[str, Dict[str, List[float]]] = {
        m: {"r1": [], "r3": [], "r5": [], "rr": []} for m in methods
    }
    per_query_details: List[Dict[str, Any]] = []

    start_time = time.time()

    for item in eval_set:
        q = item["question"]
        exp_pages = item["expected_pages"]

        # Run 4 strategies (top 5 chunks each)
        v_res = retrieve_vector_only(q, top_k=5)
        b_res = retrieve_bm25_only(q, top_k=5)
        h_res = retrieve_hybrid(q, top_k=5)
        r_res = retrieve_hybrid_reranked(q, top_k=5)

        v_pages = [c["page"] for c in v_res]
        b_pages = [c["page"] for c in b_res]
        h_pages = [c["page"] for c in h_res]
        r_pages = [c["page"] for c in r_res]

        m_v = calculate_query_metrics(v_pages, exp_pages, max_k=5)
        m_b = calculate_query_metrics(b_pages, exp_pages, max_k=5)
        m_h = calculate_query_metrics(h_pages, exp_pages, max_k=5)
        m_r = calculate_query_metrics(r_pages, exp_pages, max_k=5)

        # Accumulate
        mapping = [
            ("Vector (FAISS)", m_v),
            ("BM25 (Keyword)", m_b),
            ("Hybrid (FAISS+BM25)", m_h),
            ("Hybrid + Reranker", m_r),
        ]
        for m_name, m_dict in mapping:
            raw_scores[m_name]["r1"].append(m_dict["recall_at_1"])
            raw_scores[m_name]["r3"].append(m_dict["recall_at_3"])
            raw_scores[m_name]["r5"].append(m_dict["recall_at_5"])
            raw_scores[m_name]["rr"].append(m_dict["reciprocal_rank"])

        per_query_details.append({
            "id": item["id"],
            "question": q,
            "expected_pages": exp_pages,
            "category": item.get("category", "general"),
            "concept": item.get("concept", ""),
            "retrieved_pages": {
                "vector": v_pages,
                "bm25": b_pages,
                "hybrid": h_pages,
                "reranked": r_pages,
            },
            "metrics": {
                "vector": m_v,
                "bm25": m_b,
                "hybrid": m_h,
                "reranked": m_r,
            }
        })

    total_duration_sec = round(time.time() - start_time, 2)

    # Calculate aggregate means
    summary_metrics: Dict[str, Dict[str, float]] = {}
    for m in methods:
        summary_metrics[m] = {
            "recall_at_1": round(sum(raw_scores[m]["r1"]) / n_queries, 4),
            "recall_at_3": round(sum(raw_scores[m]["r3"]) / n_queries, 4),
            "recall_at_5": round(sum(raw_scores[m]["r5"]) / n_queries, 4),
            "mrr": round(sum(raw_scores[m]["rr"]) / n_queries, 4),
        }

    benchmark_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_questions": n_queries,
        "evaluation_duration_sec": total_duration_sec,
        "relevance_definition": (
            "A retrieved chunk is relevant if its physical book page number matches "
            "one of the ground-truth textbook pages documented for that question."
        ),
        "methods": summary_metrics,
        "per_query_results": per_query_details,
    }

    # Save to disk for reproducibility
    try:
        METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(benchmark_record, f, indent=2)
    except Exception as exc:
        print(f"Warning: Failed to save benchmark metrics file: {exc}")

    return benchmark_record


def load_or_run_benchmark(force_run: bool = False) -> Dict[str, Any]:
    """
    Loads saved benchmark results from JSON file if available; otherwise runs the evaluation.
    """
    if not force_run and METRICS_FILE.exists():
        try:
            with open(METRICS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("methods") and data.get("num_questions"):
                    return data
        except Exception:
            pass

    return run_benchmark_evaluation()


def format_metrics_dashboard_html(data: Dict[str, Any]) -> str:
    """Formats benchmark evaluation results into a clean, modern table with visual bars."""
    if not data or not data.get("methods"):
        return "<p style='color: #94a3b8;'>No evaluation metrics available. Click 'Run Evaluation' to compute.</p>"

    methods = data.get("methods", {})
    num_q = data.get("num_questions", len(EVALUATION_DATASET))
    ts = data.get("timestamp", "")[:19].replace("T", " ")
    dur = data.get("evaluation_duration_sec", 0.0)

    rows = []
    color_map = {
        "Vector (FAISS)": "#38bdf8",
        "BM25 (Keyword)": "#f59e0b",
        "Hybrid (FAISS+BM25)": "#10b981",
        "Hybrid + Reranker": "#a855f7",
    }

    for method, vals in methods.items():
        color = color_map.get(method, "#6366f1")
        r1 = vals.get("recall_at_1", 0.0) * 100
        r3 = vals.get("recall_at_3", 0.0) * 100
        r5 = vals.get("recall_at_5", 0.0) * 100
        mrr = vals.get("mrr", 0.0)

        # Visual mini-progress bar for MRR
        mrr_bar_width = int(mrr * 100)

        rows.append(f"""
        <tr>
            <td style="font-weight: 700; color: {color};">
                <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: {color}; margin-right: 6px;"></span>
                {method}
            </td>
            <td style="font-weight: 600; color: #f8fafc;">{r1:.1f}%</td>
            <td style="font-weight: 600; color: #f8fafc;">{r3:.1f}%</td>
            <td style="font-weight: 600; color: #f8fafc;">{r5:.1f}%</td>
            <td>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="flex: 1; height: 6px; background: rgba(255,255,255,0.08); border-radius: 9999px; overflow: hidden; min-width: 60px;">
                        <div style="width: {mrr_bar_width}%; height: 100%; background: {color}; border-radius: 9999px;"></div>
                    </div>
                    <strong style="color: {color}; font-family: monospace;">{mrr:.3f}</strong>
                </div>
            </td>
        </tr>
        """)

    html = f"""
    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 12px; padding: 16px; margin-bottom: 14px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 12px;">
            <div>
                <span style="font-weight: 800; color: #f8fafc; font-size: 1.02rem;">📈 RAG Retrieval Evaluation Dashboard</span>
                <span style="background: rgba(99, 102, 241, 0.2); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.4); padding: 2px 8px; border-radius: 6px; font-size: 0.74rem; font-weight: 600; margin-left: 8px;">
                    {num_q} Questions Benchmark
                </span>
            </div>
            <div style="font-size: 0.76rem; color: #94a3b8;">
                Last Evaluated: <strong style="color: #cbd5e1;">{ts} UTC</strong> {f'({dur}s)' if dur else ''}
            </div>
        </div>

        <table class="glass-table" style="width: 100%; margin-bottom: 12px;">
            <thead>
                <tr>
                    <th>Retrieval Strategy</th>
                    <th>Recall@1</th>
                    <th>Recall@3</th>
                    <th>Recall@5</th>
                    <th>MRR (Mean Reciprocal Rank)</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>

        <div style="background: rgba(0, 0, 0, 0.25); border-radius: 8px; padding: 10px 14px; font-size: 0.78rem; color: #94a3b8; line-height: 1.5;">
            <div><strong>ℹ️ Evaluation Methodology &amp; Relevance Definition:</strong></div>
            <div style="color: #cbd5e1; margin-top: 2px;">
                Relevance is strictly verified against documented physical book pages across 16 textbook questions.
                <strong>Recall@K</strong> measures the proportion of questions where at least one correct textbook page appears within the top-K chunks.
                <strong>MRR</strong> measures the reciprocal rank of the first relevant retrieved page across the corpus.
                Results are saved and reproducible in <code>data/retrieval_benchmark_metrics.json</code>.
            </div>
        </div>
    </div>
    """
    return html
