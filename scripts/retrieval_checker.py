#!/usr/bin/env python3
"""
Retrieval Checker - Inspection and Debugging Tool for Corpus Retrieval
Assignment 2 / Toolkit 6 requirement:
Takes a question, runs retrieval, and shows top-k chunks with scores and expected chunk verification.
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.retriever import retriever
from app.config import TOP_K, SIMILARITY_THRESHOLD

QUESTIONS_PATH = ROOT / "data" / "questions.json"


def load_questions():
    if QUESTIONS_PATH.exists():
        try:
            items = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
            for idx, item in enumerate(items, 1):
                if "id" not in item:
                    item["id"] = idx
            return items
        except Exception:
            return []
    return []


def find_expected_chunks(question_text, questions):
    q_norm = question_text.strip().lower()
    for item in questions:
        if item.get("question", "").strip().lower() == q_norm:
            return item.get("expected_chunks", []), item
    return None, None


def get_page_pairs(items):
    pairs = set()
    for it in items or []:
        src = str(it.get("source", ""))
        pages = it.get("page")
        if not isinstance(pages, list):
            pages = [pages] if pages is not None else []
        for p in pages:
            pairs.add((src, str(p)))
    return pairs


def inspect_retrieval(query: str, top_k: int = TOP_K, question_id: int = None):
    questions = load_questions()
    matched_item = None
    expected_chunks = None

    if question_id is not None and 1 <= question_id <= len(questions):
        matched_item = questions[question_id - 1]
        query = matched_item.get("question", query)
        expected_chunks = matched_item.get("expected_chunks", [])
    else:
        expected_chunks, matched_item = find_expected_chunks(query, questions)

    print("=" * 80)
    print("🔍 RETRIEVAL CHECKER")
    print("=" * 80)
    print(f"Query: \"{query}\"")
    if matched_item:
        print(f"Dataset Match: Q#{matched_item.get('id', '?')} [{matched_item.get('category', 'uncat')}] (Difficulty: {matched_item.get('difficulty', 'unknown')})")
    print(f"Retrieving top {top_k} chunks with Hybrid Search (FAISS + BM25)...\n")

    results = retriever.retrieve(query, top_k=top_k)

    print(f"Retrieved {len(results)} chunks:")
    print("-" * 80)

    retrieved_pairs = get_page_pairs(results)
    expected_pairs = get_page_pairs(expected_chunks) if expected_chunks else set()

    for idx, chunk in enumerate(results, 1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page", "?")
        dense = chunk.get("dense_score", 0.0)
        bm25 = chunk.get("bm25_score", 0.0)
        combined = chunk.get("combined_score", 0.0)
        snippet = chunk.get("text", "").replace("\n", " ").strip()
        if len(snippet) > 220:
            snippet = snippet[:220] + "..."

        is_match = (str(source), str(page)) in expected_pairs if expected_pairs else None
        match_tag = " [TARGET HIT 🎯]" if is_match else ""

        print(f"[{idx}] Source: {source} | Page: {page} {match_tag}")
        print(f"    Scores: Combined={combined:.4f} | Dense={dense:.4f} | BM25={bm25:.4f}")
        print(f"    Excerpt: \"{snippet}\"")
        print()

    print("=" * 80)
    print("VERIFICATION SUMMARY:")
    if expected_chunks:
        print(f"Expected targets: {sorted(list(expected_pairs))}")
        print(f"Retrieved pages:  {sorted(list(retrieved_pairs))}")
        hits = expected_pairs & retrieved_pairs
        misses = expected_pairs - retrieved_pairs

        if hits:
            print(f"✅ Target Chunks Retrieved: {sorted(list(hits))}")
        if misses:
            print(f"❌ Target Chunks Missed:    {sorted(list(misses))}")

        if not misses:
            print("Status: ✅ ALL EXPECTED CHUNKS RETRIEVED (100% Hit)")
        elif hits:
            print("Status: ⚠️ PARTIAL HIT (At least one expected chunk found)")
        else:
            print("Status: ❌ RETRIEVAL MISS (Zero expected chunks in top-k)")
    else:
        print("ℹ️ No ground-truth expected chunks found for this query in questions.json.")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description="Inspect and debug retrieval for any question.")
    parser.add_argument("query", nargs="*", help="Question text to test")
    parser.add_argument("--id", type=int, default=None, help="Question ID from questions.json (1-based)")
    parser.add_argument("--k", type=int, default=TOP_K, help="Number of chunks to retrieve (default: 7)")
    args = parser.parse_args()

    query_str = " ".join(args.query).strip() if args.query else ""

    if args.id is not None:
        inspect_retrieval("", top_k=args.k, question_id=args.id)
    elif query_str:
        inspect_retrieval(query_str, top_k=args.k)
    else:
        print("Interactive Mode. Type your question or question ID (or 'exit'):")
        while True:
            try:
                line = input("\nQuery or Q-ID> ").strip()
                if not line or line.lower() in ("exit", "quit", "q"):
                    break
                if line.isdigit():
                    inspect_retrieval("", top_k=args.k, question_id=int(line))
                else:
                    inspect_retrieval(line, top_k=args.k)
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
