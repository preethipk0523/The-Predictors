"""
Reranker Module using Sentence-Transformers Cross-Encoder.
Performs second-stage neural reranking over candidates retrieved from first-stage FAISS + BM25 hybrid search.
Configurable with graceful fallback if the neural model is unavailable or disabled.
"""

import re
from typing import List, Dict, Any, Optional
from app.config import (
    ENABLE_RERANKER,
    RERANKER_MODEL,
    RERANKER_TOP_INITIAL,
    RERANKER_TOP_FINAL,
)


class CrossEncoderReranker:
    """Neural cross-encoder reranker with configurable fallback."""

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        enabled: bool = ENABLE_RERANKER,
        top_n: int = RERANKER_TOP_FINAL,
    ):
        self.model_name = model_name
        self.enabled = enabled
        self.top_n = top_n
        self._model = None
        self._load_failed = False

    def _get_model(self):
        """Lazy load the CrossEncoder model."""
        if not self.enabled or self._load_failed:
            return None

        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                print(f"[Reranker] Could not load CrossEncoder ('{self.model_name}'): {exc}. Using calibrated fallback.")
                self._load_failed = True
                self._model = None

        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Reranks candidates and retains the top_n (default 4).
        Records before and after ranking and scores in each chunk dictionary.
        """
        if not candidates:
            return []

        limit = top_n or self.top_n
        model = self._get_model()

        # Step 1: Assign initial rank and record initial score
        for idx, chunk in enumerate(candidates, 1):
            chunk["initial_rank"] = idx
            chunk["initial_score"] = chunk.get("combined_score", 0.0)

        # Step 2: Compute reranking scores
        if model is not None:
            try:
                pairs = [[query, c.get("text", "")] for c in candidates]
                scores = model.predict(pairs)
                for c, s in zip(candidates, scores):
                    c["rerank_score"] = round(float(s), 4)
                    c["reranker_engine"] = self.model_name
            except Exception as exc:
                print(f"[Reranker] Prediction failed: {exc}. Falling back.")
                self._apply_fallback_scoring(query, candidates)
        else:
            self._apply_fallback_scoring(query, candidates)

        # Step 3: Sort by rerank score descending
        reranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)

        # Step 4: Retain top_n and assign final rank
        final_results = reranked[:limit]
        for idx, chunk in enumerate(final_results, 1):
            chunk["final_rank"] = idx

        return final_results

    def _apply_fallback_scoring(self, query: str, candidates: List[Dict[str, Any]]):
        """
        Calibrated fallback reranking when CrossEncoder is disabled or unavailable.
        Uses exact token alignment and RRF signal reinforcement.
        """
        q_tokens = set(re.findall(r"\b\w{3,}\b", query.lower()))
        for c in candidates:
            c_text_lower = c.get("text", "").lower()
            overlap_count = sum(1 for tok in q_tokens if tok in c_text_lower)
            overlap_ratio = overlap_count / len(q_tokens) if q_tokens else 0.0

            # Calibrate hybrid score
            combined = c.get("combined_score", 0.0)
            dense = c.get("dense_score", 0.0)
            calibrated_score = (0.55 * combined) + (0.25 * dense) + (0.20 * overlap_ratio)

            c["rerank_score"] = round(float(calibrated_score), 4)
            c["reranker_engine"] = "calibrated_hybrid_fallback"


# Global singleton instance
reranker = CrossEncoderReranker()
