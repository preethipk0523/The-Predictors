"""
Retrieval Confidence Indicator Module.
Computes a heuristic High / Medium / Low confidence indicator from retrieval similarity scores.
Explicitly labeled as a retrieval-based heuristic rather than a guaranteed probability.
"""

from typing import List, Dict, Any


def calculate_retrieval_confidence(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes High / Medium / Low retrieval confidence based on top chunk similarity scores.

    Args:
        chunks: List of retrieved/reranked chunk dictionaries.

    Returns:
        Dict with keys:
            - level: 'High' | 'Medium' | 'Low'
            - score: float (0.0 to 1.0 heuristic score)
            - badge: str (visual indicator for UI)
            - label: str
            - explanation: str
            - disclaimer: str
    """
    if not chunks:
        return {
            "level": "Low",
            "score": 0.0,
            "badge": "🔴 Low",
            "label": "Low (Retrieval Confidence)",
            "explanation": "No relevant textbook chunks were retrieved.",
            "disclaimer": "Retrieval-based confidence indicator (heuristic metric based on candidate relevance scores, not a guaranteed probability).",
        }

    # Extract top and average scores
    dense_scores = [c.get("dense_score", 0.0) for c in chunks]
    combined_scores = [c.get("combined_score", 0.0) for c in chunks]

    top_dense = max(dense_scores) if dense_scores else 0.0
    avg_combined = sum(combined_scores) / len(combined_scores) if combined_scores else 0.0

    # Determine confidence tier
    if top_dense >= 0.50 or avg_combined >= 0.60:
        level = "High"
        badge = "🟢 High"
        explanation = "Strong semantic and lexical alignment found in textbook passages."
        norm_score = min(1.0, max(0.75, avg_combined))
    elif top_dense >= 0.25 or avg_combined >= 0.35:
        level = "Medium"
        badge = "🟡 Medium"
        explanation = "Moderate relevance found; answer is grounded but passage coverage is partial."
        norm_score = round(max(0.40, min(0.74, avg_combined)), 2)
    else:
        level = "Low"
        badge = "🔴 Low"
        explanation = "Low similarity score; query may be outside the corpus or sparsely covered."
        norm_score = round(min(0.39, max(0.10, avg_combined)), 2)

    return {
        "level": level,
        "score": norm_score,
        "badge": badge,
        "label": f"{level} (Retrieval Confidence)",
        "explanation": explanation,
        "top_dense_score": round(top_dense, 4),
        "avg_combined_score": round(avg_combined, 4),
        "disclaimer": "Retrieval-based confidence indicator (heuristic metric based on candidate relevance scores, not a guaranteed probability).",
    }
