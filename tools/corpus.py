"""
Corpus Search Tool - Wraps the Hybrid Retriever (FAISS + BM25)
Assignment 7 requirement 1: Wrap Tuesday retrieve() function as search_corpus tool.
"""

from typing import Dict, Any, List
from pathlib import Path
import sys

# Ensure root workspace is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.retriever import retriever


def search_corpus(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Search The Crystal Ball Instruction Manual using hybrid vector + BM25 retrieval.

    Args:
        query: Natural language question or keywords to search for.
        top_k: Number of relevant chunks to retrieve (default: 5, max: 10).

    Returns:
        Dict containing retrieved chunks, page numbers, scores, and formatted text.
    """
    if not query or not query.strip():
        return {
            "status": "error",
            "error": "Query cannot be empty.",
            "count": 0,
            "matches": [],
            "formatted": "No query provided."
        }

    try:
        k = max(1, min(int(top_k), 10))
    except (ValueError, TypeError):
        k = 5

    try:
        chunks = retriever.retrieve(query.strip(), top_k=k)
    except Exception as exc:
        return {
            "status": "error",
            "error": f"Retrieval failed: {exc}",
            "count": 0,
            "matches": [],
            "formatted": f"Error during retrieval: {exc}"
        }

    if not chunks:
        return {
            "status": "no_matches",
            "query": query,
            "count": 0,
            "matches": [],
            "formatted": f"No matching passages found in the corpus for query '{query}'."
        }

    formatted_entries = []
    compact_matches = []

    for idx, c in enumerate(chunks, 1):
        doc = c.get("source", "crystal1 (1).pdf")
        page = c.get("page", "?")
        score = c.get("combined_score", 0.0)
        text = c.get("text", "").strip()

        formatted_entries.append(
            f"--- [Match {idx} | Document: {doc} | Page: {page} | Relevance: {score:.3f}] ---\n{text}"
        )
        compact_matches.append({
            "source": doc,
            "page": page,
            "score": score,
            "snippet": text[:280] + ("..." if len(text) > 280 else ""),
            "text_snippet": text[:280] + ("..." if len(text) > 280 else ""),
            "text": text
        })

    return {
        "status": "success",
        "query": query,
        "count": len(chunks),
        "matches": compact_matches,
        "formatted": "\n\n".join(formatted_entries)
    }

