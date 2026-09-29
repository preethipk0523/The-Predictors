"""
End-to-End RAG Pipeline orchestrating Retrieval and Generation.
"""

import time
from typing import Dict, Any, List
from app.retriever import retriever
from app.generator import generator


class RAGPipeline:
    def __init__(self):
        self.retriever = retriever
        self.generator = generator

    def run(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Executes end-to-end RAG pipeline for a given user query.
        """
        start_time = time.time()

        # Step 1: Retrieval
        retrieval_start = time.time()
        retrieved_chunks = self.retriever.retrieve(query, top_k=top_k)
        retrieval_ms = round((time.time() - retrieval_start) * 1000, 2)

        # Step 2: Generation
        gen_result = self.generator.generate(query, retrieved_chunks)

        total_ms = round((time.time() - start_time) * 1000, 2)

        # Assemble sources metadata
        sources_summary = []
        for chunk in retrieved_chunks:
            sources_summary.append({
                "source": chunk.get("source", "crystal1 (1).pdf"),
                "page": chunk.get("page", 1),
                "dense_score": chunk.get("dense_score", 0.0),
                "bm25_score": chunk.get("bm25_score", 0.0),
                "combined_score": chunk.get("combined_score", 0.0),
                "text_snippet": chunk.get("text", "")[:280] + ("..." if len(chunk.get("text", "")) > 280 else ""),
            })

        return {
            "query": query,
            "answer": gen_result["answer"],
            "cited_pages": gen_result["cited_pages"],
            "is_out_of_scope": gen_result["is_refusal"],
            "sources": sources_summary,
            "retrieval_time_ms": retrieval_ms,
            "generation_time_ms": gen_result["generation_time_ms"],
            "total_time_ms": total_ms,
            "prompt_tokens": gen_result["prompt_tokens"],
            "completion_tokens": gen_result["completion_tokens"],
            "total_tokens": gen_result["prompt_tokens"] + gen_result["completion_tokens"],
            "cost_usd": gen_result["cost_usd"],
        }


# Global singleton instance
pipeline = RAGPipeline()
