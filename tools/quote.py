"""Exact-quote lookup: returns a quote only when the exact text exists in corpus."""
from tools.corpus import _load_corpus

def quote_exact(text: str) -> dict:
    if not text:
        return {"result": "Please provide the exact text to search for.", "found": False}
    for item in _load_corpus():
        corpus_text = str(item.get("text", ""))
        position = corpus_text.find(text)
        if position >= 0:
            return {
                "result": text,
                "found": True,
                "source": item.get("source"),
                "page": item.get("page"),
            }
    return {"result": "That exact wording was not found in the corpus.", "found": False}
