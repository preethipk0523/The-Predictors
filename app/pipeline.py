"""
End-to-End RAG Pipeline orchestrating:
1. Chat Memory & Conversational Context
2. Query Rewriting (coreference resolution)
3. First-Stage FAISS + BM25 Hybrid Retrieval
4. Prompt Injection Scanning & Isolation
5. Second-Stage Cross-Encoder Reranking
6. Heuristic Retrieval Confidence Calculation
7. Context-Grounded Answer Generation
8. Post-Generation Faithfulness Verification
"""

import time
from typing import Dict, Any, List, Optional
from app.config import (
    RERANKER_TOP_INITIAL,
    RERANKER_TOP_FINAL,
    ENABLE_RERANKER,
)
from app.retriever import retriever
from app.generator import generator
from app.chat_memory import chat_memory
from app.query_rewriter import rewrite_query
from app.reranker import reranker
from app.confidence import calculate_retrieval_confidence
from app.faithfulness import verify_faithfulness
from app.security_guard import scan_for_prompt_injection


class RAGPipeline:
    def __init__(self):
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker

    def run(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = RERANKER_TOP_INITIAL,
        use_reranker: bool = ENABLE_RERANKER,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end RAG pipeline for a given user query with all extensions.
        """
        start_time = time.time()

        # Step 1: Chat Memory & Query Rewriting
        active_history = chat_history if chat_history is not None else chat_memory.get_recent_messages()
        rewrite_result = rewrite_query(query, active_history)
        search_query = rewrite_result["query"]

        # Step 2: First-Stage Hybrid Retrieval (FAISS + BM25)
        retrieval_start = time.time()
        initial_k = max(top_k, RERANKER_TOP_INITIAL)
        retrieved_chunks = self.retriever.retrieve(search_query, top_k=initial_k)
        retrieval_ms = round((time.time() - retrieval_start) * 1000, 2)

        # Step 3: Security & Prompt Injection Inspection
        injection_detected = False
        neutralized_chunks = []
        for chunk in retrieved_chunks:
            chunk_copy = dict(chunk)
            scan_res = scan_for_prompt_injection(chunk_copy.get("text", ""))
            if scan_res["is_suspicious"]:
                injection_detected = True
                chunk_copy["text"] = scan_res["sanitized_text"]
                chunk_copy["injection_flagged"] = True
                chunk_copy["injection_patterns"] = scan_res["matched_patterns"]
            neutralized_chunks.append(chunk_copy)

        # Step 4: Second-Stage Cross-Encoder Reranking
        rerank_start = time.time()
        if use_reranker and neutralized_chunks:
            final_chunks = self.reranker.rerank(search_query, neutralized_chunks, top_n=RERANKER_TOP_FINAL)
        else:
            final_chunks = neutralized_chunks[:RERANKER_TOP_FINAL]
            for idx, c in enumerate(final_chunks, 1):
                c["initial_rank"] = idx
                c["final_rank"] = idx
                c["rerank_score"] = c.get("combined_score", 0.0)
        rerank_ms = round((time.time() - rerank_start) * 1000, 2)

        # Step 5: Heuristic Retrieval Confidence Calculation
        confidence_info = calculate_retrieval_confidence(final_chunks)

        # Step 6: Grounded Generation (using top reranked chunks)
        gen_result = self.generator.generate(query, final_chunks)
        answer_text = gen_result["answer"]

        # Step 7: Post-Generation Faithfulness Verification
        faithfulness_info = verify_faithfulness(query, answer_text, final_chunks)
        retry_occurred = False
        initial_faithfulness = None

        # Step 8: Controlled 1-step retrieval retry if unsupported and not an intentional refusal
        if faithfulness_info["status"] == "Unsupported" and not gen_result["is_refusal"]:
            initial_faithfulness = dict(faithfulness_info)
            # Expand retrieval candidate pool to top 15 and rerank to top 6
            retry_raw_chunks = self.retriever.retrieve(search_query, top_k=15)
            if retry_raw_chunks:
                retry_final = (
                    self.reranker.rerank(search_query, retry_raw_chunks, top_n=6)
                    if use_reranker
                    else retry_raw_chunks[:6]
                )
                retry_gen = self.generator.generate(query, retry_final)
                retry_answer = retry_gen["answer"]
                retry_faith = verify_faithfulness(query, retry_answer, retry_final)

                retry_occurred = True
                if retry_faith["score"] > faithfulness_info["score"] or retry_faith["status"] != "Unsupported":
                    answer_text = retry_answer
                    final_chunks = retry_final
                    gen_result = retry_gen
                    faithfulness_info = retry_faith
                    faithfulness_info["explanation"] = (
                        "🔄 Verified after 1-step retrieval adjustment: " + faithfulness_info.get("explanation", "")
                    )
                else:
                    answer_text += (
                        "\n\n> ⚠️ **Verification Notice:** The retrieved textbook excerpts may not sufficiently "
                        "substantiate all claims in this answer, even after an automated retrieval retry. "
                        "Please verify against the source text."
                    )
                    gen_result["answer"] = answer_text
            else:
                retry_occurred = True
                answer_text += (
                    "\n\n> ⚠️ **Verification Notice:** The retrieved textbook excerpts may not sufficiently "
                    "substantiate all claims in this answer. Please verify against the source text."
                )
                gen_result["answer"] = answer_text

        # Record turn in Chat Memory
        chat_memory.add_turn(query, answer_text, citations=gen_result.get("cited_pages", []))

        total_ms = round((time.time() - start_time) * 1000, 2)

        # Assemble sources summary with ranking progression metadata
        sources_summary = []
        for chunk in final_chunks:
            sources_summary.append({
                "source": chunk.get("source", "crystal1 (1).pdf"),
                "page": chunk.get("page", 1),
                "dense_score": chunk.get("dense_score", 0.0),
                "bm25_score": chunk.get("bm25_score", 0.0),
                "combined_score": chunk.get("combined_score", 0.0),
                "initial_rank": chunk.get("initial_rank", 1),
                "final_rank": chunk.get("final_rank", 1),
                "rerank_score": chunk.get("rerank_score", chunk.get("combined_score", 0.0)),
                "reranker_engine": chunk.get("reranker_engine", "none"),
                "injection_flagged": chunk.get("injection_flagged", False),
                "text_snippet": chunk.get("text", "")[:280] + ("..." if len(chunk.get("text", "")) > 280 else ""),
            })

        return {
            "query": query,
            "answer": answer_text,
            "cited_pages": gen_result["cited_pages"],
            "verified_citations": gen_result.get("verified_citations", gen_result["cited_pages"]),
            "hallucinated_citations": gen_result.get("hallucinated_citations", []),
            "grounding_score": gen_result.get("grounding_score", 1.0),
            "is_out_of_scope": gen_result["is_refusal"],
            "sources": sources_summary,
            "retrieval_time_ms": retrieval_ms,
            "rerank_time_ms": rerank_ms,
            "generation_time_ms": gen_result["generation_time_ms"],
            "total_time_ms": total_ms,
            "prompt_tokens": gen_result["prompt_tokens"],
            "completion_tokens": gen_result["completion_tokens"],
            "total_tokens": gen_result["prompt_tokens"] + gen_result["completion_tokens"],
            "cost_usd": gen_result["cost_usd"],
            # New extension fields
            "rewritten_query_info": rewrite_result,
            "confidence": confidence_info,
            "faithfulness": faithfulness_info,
            "prompt_injection_detected": injection_detected,
            "retry_occurred": retry_occurred,
            "initial_faithfulness": initial_faithfulness,
            "retry_successful": (retry_occurred and faithfulness_info["status"] in ("Faithful", "Partially Supported")),
        }


# Global singleton instance
pipeline = RAGPipeline()
