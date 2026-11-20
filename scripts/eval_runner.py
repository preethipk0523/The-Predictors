#!/usr/bin/env python3
"""
Comprehensive Evaluation Benchmark Runner
Assignment 2 / Toolkit 6 requirement:
Runs questions through the pipeline, verifies retrieval hits against expected chunks,
evaluates answer generation, computes category/difficulty breakdowns, and logs scorecard.
Supports both Direct RAG and Autonomous Agent modes.
"""

import sys
import json
import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.retriever import retriever
from app.generator import generator
from app.agent import run_agent

QUESTIONS_PATH = ROOT / "data" / "questions.json"
DEFAULT_OUTPUT_PATH = ROOT / "data" / "eval_results.json"
TOP_K = 7


def get_pairs(items):
    pairs = set()
    for item in items or []:
        source = item.get("source")
        pages = item.get("page")
        if not isinstance(pages, list):
            pages = [pages] if pages is not None else []
        for page in pages:
            pairs.add((str(source), str(page)))
    return pairs


def compute_token_f1(prediction: str, ground_truth: str) -> dict:
    """Computes token precision, recall, and F1 between prediction and expected answer."""
    if not prediction or not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_tokens = re.findall(r"\w+", prediction.lower())
    gt_tokens = re.findall(r"\w+", ground_truth.lower())

    if not pred_tokens or not gt_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compute_citation_metrics(cited_pages: list, expected_chunks: list) -> dict:
    """Computes citation precision, recall, and F1 against expected ground-truth chunks."""
    expected_pages = set()
    for c in expected_chunks or []:
        p = c.get("page")
        if isinstance(p, list):
            expected_pages.update(p)
        elif p is not None:
            expected_pages.add(p)

    if not expected_pages:
        return {"precision": 1.0 if not cited_pages else 0.0, "recall": 1.0, "f1": 1.0}

    cited_set = set(cited_pages or [])
    if not cited_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    intersection = cited_set & expected_pages
    precision = len(intersection) / len(cited_set)
    recall = len(intersection) / len(expected_pages)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def rate(hits, total):
    return round(hits * 100 / total, 2) if total else 0.0


def run_evaluation(
    mode: str = "rag",
    limit: int = None,
    category_filter: str = None,
    question_id: int = None,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    run_name: str = "eval_run"
):
    raw_questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions = []
    for idx, q in enumerate(raw_questions, 1):
        if "id" not in q:
            q["id"] = idx
        questions.append(q)

    if question_id is not None:
        questions = [q for q in questions if q.get("id") == question_id]
    elif category_filter:
        questions = [q for q in questions if q.get("category", "").lower() == category_filter.lower()]

    if limit and limit > 0:
        questions = questions[:limit]

    results = []
    print(f"==================================================")
    print(f"Starting {run_name} Evaluation | Mode: {mode.upper()} | Questions: {len(questions)}")
    print(f"==================================================")

    for i, item in enumerate(questions, start=1):
        q_id = item.get("id", i)
        question = item["question"]
        expected = item.get("expected_chunks", [])
        cat = item.get("category", "uncategorized")
        diff = item.get("difficulty", "unknown")

        print(f"[{i}/{len(questions)}] Q#{q_id} [{cat}]: {question[:65]}...")

        retrieved = []
        retrieval_error = None
        generation_error = None
        answer = None
        metrics = {}

        if mode == "agent":
            # Run Autonomous Agent
            try:
                agent_res = run_agent(question)
                answer = agent_res.get("answer")
                cited_pages = agent_res.get("cited_pages", [])
                retrieved_pairs = {(str(item.get("source", "crystal1 (1).pdf")), str(p)) for p in cited_pages}

                expected_pairs = get_pairs(expected)
                matched = expected_pairs & retrieved_pairs

                retrieval_hit = bool(matched) if expected_pairs else None
                retrieval_full_hit = expected_pairs.issubset(retrieved_pairs) if expected_pairs else None

                metrics = {
                    "tool_calls_count": agent_res.get("tool_calls_count", 0),
                    "steps_count": agent_res.get("steps_count", 0),
                    "trajectory": agent_res.get("trajectory", []),
                    "cited_pages": cited_pages,
                    "generation_time_ms": agent_res.get("duration_ms"),
                    "prompt_tokens": agent_res.get("prompt_tokens"),
                    "completion_tokens": agent_res.get("completion_tokens"),
                    "cost_usd": agent_res.get("cost_usd"),
                    "is_refusal": "don't have information" in (answer or "").lower(),
                }
            except Exception as exc:
                generation_error = f"Agent error: {exc}"
                retrieval_hit = False
                retrieval_full_hit = False

        else:
            # Direct RAG Pipeline
            try:
                retrieved = retriever.retrieve(question, top_k=TOP_K)
            except Exception as exc:
                retrieval_error = f"{type(exc).__name__}: {exc}"

            chunks = [
                {
                    "source": chunk.get("source"),
                    "page": chunk.get("page"),
                    "score": chunk.get("combined_score"),
                    "dense_score": chunk.get("dense_score"),
                    "bm25_score": chunk.get("bm25_score"),
                    "chunk_id": chunk.get("chunk_id"),
                }
                for chunk in retrieved
            ]

            expected_pairs = get_pairs(expected)
            retrieved_pairs = get_pairs(retrieved)
            matched = expected_pairs & retrieved_pairs

            retrieval_hit = bool(matched) if expected_pairs else None
            retrieval_full_hit = expected_pairs.issubset(retrieved_pairs) if expected_pairs else None

            if retrieval_error is None:
                try:
                    generated = generator.generate(question, retrieved)
                    answer = generated.get("answer")
                    exp_ans = item.get("expected_answer", "")
                    tok_f1 = compute_token_f1(answer or "", exp_ans)
                    cit_m = compute_citation_metrics(generated.get("cited_pages", []), expected)
                    metrics = {
                        "cited_pages": generated.get("cited_pages", []),
                        "verified_citations": generated.get("verified_citations", []),
                        "hallucinated_citations": generated.get("hallucinated_citations", []),
                        "grounding_score": generated.get("grounding_score", 1.0),
                        "token_precision": tok_f1["precision"],
                        "token_recall": tok_f1["recall"],
                        "token_f1": tok_f1["f1"],
                        "citation_precision": cit_m["precision"],
                        "citation_recall": cit_m["recall"],
                        "citation_f1": cit_m["f1"],
                        "generation_time_ms": generated.get("generation_time_ms"),
                        "prompt_tokens": generated.get("prompt_tokens"),
                        "completion_tokens": generated.get("completion_tokens"),
                        "cost_usd": generated.get("cost_usd"),
                        "is_refusal": generated.get("is_refusal"),
                    }
                except Exception as exc:
                    generation_error = f"{type(exc).__name__}: {exc}"

        if retrieval_error:
            status = "retrieval error"
        elif generation_error:
            status = "generation error"
        elif retrieval_hit is False:
            status = "retrieval miss"
        else:
            status = "completed"

        results.append({
            "id": q_id,
            "question": question,
            "category": cat,
            "difficulty": diff,
            "expected_answer": item.get("expected_answer"),
            "expected_chunks": expected,
            "retrieved_chunks": [
                {"source": c.get("source"), "page": c.get("page"), "score": c.get("combined_score")}
                for c in retrieved
            ] if retrieved else [],
            "retrieval_hit": retrieval_hit,
            "retrieval_full_hit": retrieval_full_hit,
            "answer": answer,
            "status": status,
            "retrieval_error": retrieval_error,
            "generation_error": generation_error,
            **metrics,
        })

    scored = [r for r in results if r["retrieval_hit"] is not None]
    hits = sum(r["retrieval_hit"] is True for r in scored)
    full_hits = sum(r["retrieval_full_hit"] is True for r in scored)

    def breakdown(field):
        groups = defaultdict(list)
        for result in results:
            groups[str(result[field])].append(result)

        summary = {}
        for name, group in sorted(groups.items()):
            group_scored = [r for r in group if r["retrieval_hit"] is not None]
            group_hits = sum(r["retrieval_hit"] is True for r in group_scored)
            summary[name] = {
                "total_questions": len(group),
                "retrieval_hits": group_hits,
                "retrieval_hit_rate_percent": rate(group_hits, len(group_scored)),
            }
        return summary

    failures = []
    for r in results:
        reasons = []
        if r["retrieval_hit"] is False:
            reasons.append("retrieval miss")
        if r.get("retrieval_error"):
            reasons.append("retrieval error")
        if r.get("generation_error"):
            reasons.append("generation error")
        if reasons:
            failures.append({
                "id": r["id"],
                "question": r["question"],
                "category": r["category"],
                "difficulty": r["difficulty"],
                "failure_reasons": reasons,
            })

    scored_f1 = [r["token_f1"] for r in results if r.get("token_f1") is not None and r.get("expected_answer")]
    avg_f1 = round(sum(scored_f1) / len(scored_f1), 4) if scored_f1 else 0.0

    scored_cit_p = [r["citation_precision"] for r in results if r.get("citation_precision") is not None and r.get("expected_chunks")]
    avg_cit_p = round(sum(scored_cit_p) / len(scored_cit_p), 4) if scored_cit_p else 0.0

    scored_cit_r = [r["citation_recall"] for r in results if r.get("citation_recall") is not None and r.get("expected_chunks")]
    avg_cit_r = round(sum(scored_cit_r) / len(scored_cit_r), 4) if scored_cit_r else 0.0

    grounding_scores = [r["grounding_score"] for r in results if r.get("grounding_score") is not None]
    avg_grounding = round(sum(grounding_scores) / len(grounding_scores), 4) if grounding_scores else 0.0

    total_hallucinated_cit = sum(len(r.get("hallucinated_citations", [])) for r in results)

    output = {
        "run_name": run_name,
        "mode": mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_questions": len(results),
        "summary": {
            "retrieval_scored_questions": len(scored),
            "retrieval_hits": hits,
            "retrieval_hit_rate_percent": rate(hits, len(scored)),
            "retrieval_full_hits": full_hits,
            "retrieval_full_hit_rate_percent": rate(full_hits, len(scored)),
            "average_grounding_score": avg_grounding,
            "average_answer_token_f1": avg_f1,
            "average_citation_precision": avg_cit_p,
            "average_citation_recall": avg_cit_r,
            "total_hallucinated_citations": total_hallucinated_cit,
            "retrieval_errors": sum(r.get("retrieval_error") is not None for r in results),
            "generation_errors": sum(r.get("generation_error") is not None for r in results),
            "category_breakdown": breakdown("category"),
            "difficulty_breakdown": breakdown("difficulty"),
        },
        "failed_questions": failures,
        "results": results,
    }

    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== EVALUATION SCORECARD ==========")
    print(f"Mode: {mode.upper()} | Evaluated: {len(results)} questions")
    print(f"Overall retrieval hit rate: {hits}/{len(scored)} ({rate(hits, len(scored))}%)")
    print(f"Full retrieval hit rate: {full_hits}/{len(scored)} ({rate(full_hits, len(scored))}%)")
    print(f"Average grounding score: {avg_grounding:.3f}")
    print(f"Average answer token F1: {avg_f1:.4f}")
    print(f"Citation precision: {avg_cit_p:.4f} | Citation recall: {avg_cit_r:.4f}")
    print(f"Hallucinated citations: {total_hallucinated_cit}")

    print("\nBy category:")
    for name, stats in output["summary"]["category_breakdown"].items():
        print(f"  {name}: {stats}")

    print("\nBy difficulty:")
    for name, stats in output["summary"]["difficulty_breakdown"].items():
        print(f"  {name}: {stats}")

    print(f"\nFailed questions: {len(failures)}")
    for failure in failures:
        print(f"  Q#{failure['id']} [{failure['category']}]: {', '.join(failure['failure_reasons'])} - {failure['question'][:75]}")

    print(f"\nResults saved to: {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Run evaluation benchmark over questions.json")
    parser.add_argument("--mode", choices=["rag", "agent"], default="rag", help="Pipeline engine to test (default: rag)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to evaluate")
    parser.add_argument("--category", type=str, default=None, help="Filter by category (factual, multi-step, stress-test, etc.)")
    parser.add_argument("--id", type=int, default=None, help="Evaluate a single question by ID")
    parser.add_argument("--run-name", type=str, default="benchmark", help="Run identifier name")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Output json path")
    args = parser.parse_args()

    run_evaluation(
        mode=args.mode,
        limit=args.limit,
        category_filter=args.category,
        question_id=args.id,
        output_path=Path(args.output),
        run_name=args.run_name
    )


if __name__ == "__main__":
    main()