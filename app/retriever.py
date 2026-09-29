"""
Hybrid Retriever module combining FAISS dense vector search with BM25 keyword search.
"""

from typing import List, Dict, Any
import numpy as np
import faiss
from openai import OpenAI

from app.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    TOP_K,
    DENSE_WEIGHT,
    BM25_WEIGHT,
    SIMILARITY_THRESHOLD,
)
from app.indexer import load_index, tokenize_for_bm25
from app.cost_tracker import cost_tracker


class HybridRetriever:
    def __init__(self, top_k: int = TOP_K):
        self.top_k = top_k
        self.index, self.chunks, self.bm25 = load_index()
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieves top_k relevant chunks using hybrid scoring (Dense + BM25).
        """
        k = top_k or self.top_k
        k_search = min(k * 3, len(self.chunks))

        # 1. Dense Embedding Search
        embed_resp = self.client.embeddings.create(
            input=[query], model=EMBEDDING_MODEL
        )
        cost_tracker.log_embedding(embed_resp.usage.total_tokens)

        query_vec = np.array([embed_resp.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        # Search FAISS index
        dense_sims, dense_indices = self.index.search(query_vec, k_search)
        dense_sims = dense_sims[0]
        dense_indices = dense_indices[0]

        # 2. BM25 Keyword Search
        query_tokens = tokenize_for_bm25(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        norm_bm25 = bm25_scores / max_bm25

        # 3. Hybrid Combination
        scored_candidates = {}

        # Add all dense retrieved candidates
        for rank, (idx, sim) in enumerate(zip(dense_indices, dense_sims)):
            if idx < 0 or idx >= len(self.chunks):
                continue
            bm_score = float(norm_bm25[idx]) if idx < len(norm_bm25) else 0.0
            combined = DENSE_WEIGHT * float(sim) + BM25_WEIGHT * bm_score
            scored_candidates[int(idx)] = {
                "chunk": self.chunks[idx],
                "dense_score": float(sim),
                "bm25_score": bm_score,
                "combined_score": combined,
            }

        # Also add top BM25 candidates in case they weren't in dense top
        top_bm25_indices = np.argsort(bm25_scores)[-k_search:]
        for idx in top_bm25_indices:
            idx = int(idx)
            if idx not in scored_candidates and idx < len(self.chunks):
                # Calculate dense similarity for this candidate
                cand_vec = self.index.reconstruct(idx).reshape(1, -1)
                sim = float(np.dot(query_vec, cand_vec.T)[0][0])
                bm_score = float(norm_bm25[idx])
                combined = DENSE_WEIGHT * sim + BM25_WEIGHT * bm_score
                scored_candidates[idx] = {
                    "chunk": self.chunks[idx],
                    "dense_score": sim,
                    "bm25_score": bm_score,
                    "combined_score": combined,
                }

        # Sort candidates by combined score descending
        sorted_candidates = sorted(
            scored_candidates.values(),
            key=lambda x: x["combined_score"],
            reverse=True,
        )

        results = []
        for item in sorted_candidates[:k]:
            chunk_data = dict(item["chunk"])
            chunk_data["dense_score"] = round(item["dense_score"], 4)
            chunk_data["bm25_score"] = round(item["bm25_score"], 4)
            chunk_data["combined_score"] = round(item["combined_score"], 4)
            chunk_data["is_confident"] = item["dense_score"] >= SIMILARITY_THRESHOLD
            results.append(chunk_data)

        return results


# Global singleton instance
retriever = HybridRetriever()
