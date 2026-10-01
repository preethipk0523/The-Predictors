"""
Exact Quote & Page Inspection Tool
Assignment 7 requirement 3: Add a quote_exact tool that returns the exact text of
a given page from a named document (verify or quote a passage a search pointed to).
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "corpus.json"
_CORPUS_CACHE: Optional[list] = None


def _get_corpus() -> list:
    global _CORPUS_CACHE
    if _CORPUS_CACHE is None:
        if CORPUS_PATH.exists():
            try:
                _CORPUS_CACHE = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
            except Exception:
                _CORPUS_CACHE = []
        else:
            _CORPUS_CACHE = []
    return _CORPUS_CACHE


def quote_exact(
    page: int,
    document: str = "crystal1 (1).pdf",
    passage: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieve the exact text of a given page from a named document to verify or inspect passages.

    Args:
        page: The 1-based page number in the document.
        document: The filename of the document (default: 'crystal1 (1).pdf').
        passage: Optional specific passage or keywords to verify/locate within that page.

    Returns:
        Dict containing full page text, char count, page number, and passage verification status.
    """
    corpus = _get_corpus()
    if not corpus:
        return {
            "status": "error",
            "error": "Corpus dataset not found at data/corpus.json.",
            "found": False,
            "text": ""
        }

    try:
        page_num = int(page)
        if page_num <= 0:
            return {
                "status": "error",
                "error": f"Invalid page number '{page}'. Page must be greater than 0.",
                "found": False,
                "text": ""
            }
    except (ValueError, TypeError):
        return {
            "status": "error",
            "error": f"Invalid page number '{page}'. Page must be an integer.",
            "found": False,
            "text": ""
        }

    doc_norm = document.lower().strip() if document else ""

    matched_page = None
    for entry in corpus:
        entry_doc = str(entry.get("source", "")).lower().strip()
        entry_page = entry.get("page")

        if entry_page == page_num:
            if not doc_norm or doc_norm in entry_doc or entry_doc in doc_norm:
                matched_page = entry
                break

    if not matched_page:
        return {
            "status": "error",
            "error": f"Page {page_num} not found in document '{document}'.",
            "found": False,
            "page": page_num,
            "document": document,
            "text": ""
        }

    page_text = matched_page.get("text", "")
    response = {
        "status": "success",
        "found": True,
        "document": matched_page.get("source", document),
        "page": page_num,
        "char_count": len(page_text),
        "text": page_text,
    }

    if passage:
        p_clean = passage.strip()
        if p_clean.lower() in page_text.lower():
            response["passage_verified"] = True
            response["passage_status"] = f"Passage '{p_clean}' is verified present on Page {page_num}."
        else:
            response["passage_verified"] = False
            response["passage_status"] = f"Passage '{p_clean}' was NOT found on Page {page_num}."

    return response

