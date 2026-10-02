"""
Feedback Collection Module.
Persists thumbs up / thumbs down ratings, questions, answers, and timestamps to a local JSONL file.
Provides summary metrics for evaluation and system status display.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from app.config import FEEDBACK_FILE


def save_feedback(
    query: str,
    answer: str,
    rating: str,
    comment: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Saves user feedback to a local JSONL file.

    Args:
        query: The user question.
        answer: The generated answer.
        rating: 'thumbs_up' or 'thumbs_down' (or 'up'/'down').
        comment: Optional text comment.
        metadata: Optional dictionary with engine mode, confidence, etc.

    Returns:
        Dict confirming the saved record status.
    """
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    norm_rating = "thumbs_up" if "up" in rating.lower() else "thumbs_down"

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rating": norm_rating,
        "query": query,
        "answer": answer,
        "comment": comment,
        "metadata": metadata or {},
    }

    try:
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"status": "success", "record": record}
    except Exception as exc:
        print(f"[Feedback] Failed to save feedback: {exc}")
        return {"status": "error", "error": str(exc)}


def get_feedback_records() -> List[Dict[str, Any]]:
    """Reads all feedback records from disk."""
    if not FEEDBACK_FILE.exists():
        return []
    records = []
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    records.append(json.loads(line_str))
    except Exception as exc:
        print(f"[Feedback] Failed to load feedback: {exc}")
    return records


def get_feedback_stats() -> Dict[str, Any]:
    """Computes summary statistics of user ratings."""
    records = get_feedback_records()
    total = len(records)
    up_count = sum(1 for r in records if r.get("rating") == "thumbs_up")
    down_count = sum(1 for r in records if r.get("rating") == "thumbs_down")
    satisfaction_rate = round((up_count / total * 100), 1) if total > 0 else 100.0

    return {
        "total_feedback": total,
        "thumbs_up": up_count,
        "thumbs_down": down_count,
        "satisfaction_rate_pct": satisfaction_rate,
    }
