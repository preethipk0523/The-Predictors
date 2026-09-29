"""
Cost tracking module for RAG application and evaluation.
Tracks OpenAI token consumption and calculates cumulative USD costs against the $5.00 team budget.
"""

import json
from pathlib import Path
from typing import Dict, Any
from app.config import (
    COST_LOG_PATH,
    EMBEDDING_PRICE_PER_M,
    CHAT_INPUT_PRICE_PER_M,
    CHAT_OUTPUT_PRICE_PER_M,
)


class CostTracker:
    def __init__(self, log_path: Path = COST_LOG_PATH):
        self.log_path = log_path
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.log_path.exists():
            try:
                return json.loads(self.log_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "total_cost_usd": 0.0,
            "embedding_tokens": 0,
            "embedding_cost_usd": 0.0,
            "chat_input_tokens": 0,
            "chat_output_tokens": 0,
            "chat_cost_usd": 0.0,
            "query_count": 0,
        }

    def _save(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(
            json.dumps(self.data, indent=2), encoding="utf-8"
        )

    def log_embedding(self, token_count: int) -> float:
        cost = (token_count / 1_000_000) * EMBEDDING_PRICE_PER_M
        self.data["embedding_tokens"] += token_count
        self.data["embedding_cost_usd"] += cost
        self.data["total_cost_usd"] += cost
        self._save()
        return cost

    def log_chat(self, prompt_tokens: int, completion_tokens: int) -> float:
        input_cost = (prompt_tokens / 1_000_000) * CHAT_INPUT_PRICE_PER_M
        output_cost = (completion_tokens / 1_000_000) * CHAT_OUTPUT_PRICE_PER_M
        cost = input_cost + output_cost
        self.data["chat_input_tokens"] += prompt_tokens
        self.data["chat_output_tokens"] += completion_tokens
        self.data["chat_cost_usd"] += cost
        self.data["total_cost_usd"] += cost
        self.data["query_count"] += 1
        self._save()
        return cost

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_cost_usd": round(self.data["total_cost_usd"], 6),
            "budget_remaining_usd": round(5.00 - self.data["total_cost_usd"], 6),
            "budget_used_percent": round((self.data["total_cost_usd"] / 5.00) * 100, 4),
            "embedding_tokens": self.data["embedding_tokens"],
            "embedding_cost_usd": round(self.data["embedding_cost_usd"], 6),
            "chat_input_tokens": self.data["chat_input_tokens"],
            "chat_output_tokens": self.data["chat_output_tokens"],
            "chat_cost_usd": round(self.data["chat_cost_usd"], 6),
            "query_count": self.data["query_count"],
        }


# Global singleton instance
cost_tracker = CostTracker()
