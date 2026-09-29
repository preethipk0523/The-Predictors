"""Simple JSON corpus search. Replace/extend with vector search if required."""
import json
from pathlib import Path
from typing import Any

CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "corpus.json"

def _load_corpus() -> list[dict[str, Any]]:
    if not CORPUS_PATH.exists():
        return []
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))

def search_corpus(query: str, top_k: int = 5) -> dict[str, Any]:
    """Return corpus entries ranked by simple token overlap."""
    query_tokens = set((query or "").lower().split())
    if not query_tokens:
        return {"result": "Please provide a search query.", "matches": []}
    ranked = []
    for item in _load_corpus():
        text = str(item.get("text", ""))
        tokens = set(text.lower().split())
        score = len(query_tokens & tokens) / max(1, len(query_tokens))
        if score > 0:
            ranked.append({**item, "score": round(score, 3)})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    matches = ranked[:max(1, min(int(top_k), 20))]
    if not matches:
        return {"result": "No matching corpus entries were found.", "matches": []}
    formatted = "\n".join(
        f"[{m.get('source', 'corpus')}:{m.get('page', '?')}] {m.get('text', '')}"
        for m in matches
    )
    return {"result": formatted, "matches": matches}
