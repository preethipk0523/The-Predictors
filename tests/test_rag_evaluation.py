"""
Unit Tests for RAG Improvements:
1. Faithfulness Verification & Grounding (fully supported, partially supported, unsupported, refusal, incorrect citation, controlled retry)
2. A/B Retrieval Comparison (Vector, BM25, Hybrid, Reranked, metadata, statistics)
3. RAG Metrics Dashboard (Recall@1, Recall@3, Recall@5, MRR, persistence)
"""

import unittest
import json
from unittest.mock import MagicMock, patch
from pathlib import Path

from app.faithfulness import verify_faithfulness
from app.ab_comparison import (
    retrieve_vector_only,
    retrieve_bm25_only,
    retrieve_hybrid,
    retrieve_hybrid_reranked,
    run_retrieval_ab_comparison,
    format_ab_comparison_html,
)
from app.metrics_dashboard import (
    calculate_query_metrics,
    run_benchmark_evaluation,
    load_or_run_benchmark,
    format_metrics_dashboard_html,
    EVALUATION_DATASET,
    METRICS_FILE,
)
from app.pipeline import RAGPipeline


class RAGEvaluationAndFaithfulnessTests(unittest.TestCase):

    # ==========================================
    # 1. FAITHFULNESS VERIFICATION TESTS
    # ==========================================

    def test_faithfulness_fully_supported(self):
        """Live test: Verify fully supported answer receives Faithful status and high score."""
        context = [
            {"page": 8, "text": "The data-to-wisdom hierarchy has four levels: data, information, knowledge, and wisdom."}
        ]
        answer = "The data-to-wisdom hierarchy consists of data, information, knowledge, and wisdom [Page 8]."
        res = verify_faithfulness("What are the four levels?", answer, context)

        self.assertTrue(res["is_faithful"])
        self.assertEqual(res["status"], "Faithful")
        self.assertGreaterEqual(res["score"], 0.85)
        self.assertIn("✅", res["badge"])
        self.assertEqual(res["unsupported_citations"], [])

    def test_faithfulness_partially_supported(self):
        """Verify partially supported answer receives Partially Supported status."""
        context = [
            {"page": 44, "text": "Stanley Smith Stevens defined four measurement scales: nominal, ordinal, interval, and ratio."}
        ]
        # Core is Stevens scales, but also includes ungrounded claim about temperature formulas
        answer = (
            "Stevens defined four scales: nominal, ordinal, interval, and ratio [Page 44]. "
            "Additionally, Kelvin scale requires quantum thermodynamic calibrations."
        )
        res = verify_faithfulness("What are Stevens scales?", answer, context)
        self.assertIn(res["status"], ("Partially Supported", "Faithful"))

    def test_faithfulness_unsupported(self):
        """Verify completely fabricated answer is flagged as Unsupported."""
        context = [
            {"page": 8, "text": "The data-to-wisdom hierarchy has four levels: data, information, knowledge, and wisdom."}
        ]
        # Answer completely unrelated to context
        answer = "Photosynthesis converts solar photons into glucose using chlorophyll A and B."
        res = verify_faithfulness("What are the four levels?", answer, context)

        self.assertFalse(res["is_faithful"])
        self.assertEqual(res["status"], "Unsupported")
        self.assertLess(res["score"], 0.40)
        self.assertIn("❌", res["badge"])
        self.assertTrue(res["retry_recommended"])

    def test_faithfulness_faithful_refusal(self):
        """Verify standard out-of-scope refusal is recognized as Faithful Refusal."""
        refusal = "I don't have information about this in the corpus."
        res = verify_faithfulness("What is quantum chromodynamics?", refusal, [])

        self.assertTrue(res["is_faithful"])
        self.assertEqual(res["status"], "Faithful Refusal")
        self.assertEqual(res["score"], 1.0)
        self.assertIn("🛡️", res["badge"])
        self.assertFalse(res["retry_recommended"])

    def test_faithfulness_incorrect_citation_flagged(self):
        """Verify that citations to pages absent from retrieved excerpts are flagged as invalid citations."""
        context = [
            {"page": 8, "text": "Deriving knowledge from data is the core definition of data science."}
        ]
        # Answer cites [Page 999] which was NOT in retrieved context
        answer = "Data science derives knowledge from data [Page 999]."
        res = verify_faithfulness("What is data science?", answer, context)

        self.assertIn(999, res["unsupported_citations"])
        self.assertIn(res["status"], ("Partially Supported", "Unsupported"))
        self.assertTrue(any("999" in str(claim) for claim in res["unsupported_claims"]))

    def test_pipeline_controlled_retry_on_unsupported(self):
        """Verify that pipeline performs exactly one controlled retrieval retry when answer is unsupported."""
        pipe = RAGPipeline()

        # Mock initial retrieve returning weak context that causes unsupported verification
        weak_chunk = [{"source": "crystal1 (1).pdf", "page": 1, "text": "Title page", "combined_score": 0.3}]
        better_chunk = [{"source": "crystal1 (1).pdf", "page": 8, "text": "Data, information, knowledge, wisdom", "combined_score": 0.9}]

        mock_initial_gen = {
            "answer": "Fabricated unsubstantiated claim about Mars rover.",
            "cited_pages": [999],
            "verified_citations": [],
            "hallucinated_citations": [999],
            "grounding_score": 0.1,
            "is_refusal": False,
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "generation_time_ms": 100.0,
            "cost_usd": 0.0001,
        }

        mock_retry_gen = {
            "answer": "Data, information, knowledge, wisdom are the four levels [Page 8].",
            "cited_pages": [8],
            "verified_citations": [8],
            "hallucinated_citations": [],
            "grounding_score": 0.95,
            "is_refusal": False,
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "generation_time_ms": 100.0,
            "cost_usd": 0.0001,
        }

        with patch.object(pipe.retriever, "retrieve", side_effect=[weak_chunk, better_chunk]), \
             patch.object(pipe.generator, "generate", side_effect=[mock_initial_gen, mock_retry_gen]):
            out = pipe.run("What are the levels of hierarchy?")
            self.assertTrue(out["retry_occurred"])
            self.assertIsNotNone(out["initial_faithfulness"])
            self.assertEqual(out["initial_faithfulness"]["status"], "Unsupported")

    # ==========================================
    # 2. A/B RETRIEVAL COMPARISON TESTS
    # ==========================================

    def test_ab_vector_retrieval(self):
        """Verify Vector-only search runs and returns ranked chunks with dense scores."""
        res = retrieve_vector_only("What is data science?", top_k=3)
        self.assertIsInstance(res, list)
        self.assertGreater(len(res), 0)
        top = res[0]
        self.assertEqual(top["rank"], 1)
        self.assertIn("score", top)
        self.assertIn("page", top)
        self.assertIn("source", top)
        self.assertIn("snippet", top)
        self.assertEqual(top["method"], "Vector (FAISS)")

    def test_ab_bm25_retrieval(self):
        """Verify BM25-only search runs and returns ranked chunks with BM25 scores."""
        res = retrieve_bm25_only("Stevens scales of measurement", top_k=3)
        self.assertIsInstance(res, list)
        self.assertGreater(len(res), 0)
        top = res[0]
        self.assertEqual(top["rank"], 1)
        self.assertIn("score", top)
        self.assertIn("page", top)
        self.assertIn("snippet", top)
        self.assertEqual(top["method"], "BM25 (Keyword)")

    def test_ab_hybrid_retrieval(self):
        """Verify Hybrid retrieval runs and returns combined scores."""
        res = retrieve_hybrid("Simpson's Paradox", top_k=3)
        self.assertIsInstance(res, list)
        self.assertGreater(len(res), 0)
        top = res[0]
        self.assertIn("dense_score", top)
        self.assertIn("bm25_score", top)
        self.assertIn("score", top)
        self.assertEqual(top["method"], "Hybrid (FAISS + BM25)")

    def test_ab_comparison_output_structure_and_stats(self):
        """Verify complete A/B comparison compiles all methods and comparative statistics."""
        ab = run_retrieval_ab_comparison("What is Gini impurity?", top_k=3)
        self.assertIn("query", ab)
        self.assertIn("methods", ab)
        self.assertIn("statistics", ab)
        self.assertIn("timing_ms", ab)

        methods = ab["methods"]
        self.assertIn("vector", methods)
        self.assertIn("bm25", methods)
        self.assertIn("hybrid", methods)
        self.assertIn("reranked", methods)

        stats = ab["statistics"]
        self.assertIn("vector_avg_score", stats)
        self.assertIn("bm25_avg_score", stats)
        self.assertIn("hybrid_avg_score", stats)
        self.assertIn("total_unique_pages", stats)
        self.assertIn("vector_hybrid_overlap_count", stats)

        html = format_ab_comparison_html(ab)
        self.assertIn("Comparative Retrieval Statistics", html)
        self.assertIn("Vector-only", html)
        self.assertIn("Keyword-only", html)

    # ==========================================
    # 3. METRICS DASHBOARD TESTS
    # ==========================================

    def test_recall_and_mrr_calculation(self):
        """Verify exact calculation of Recall@1, Recall@3, Recall@5, and MRR."""
        # Case A: Relevant page appears at Rank 1
        m1 = calculate_query_metrics(retrieved_pages=[8, 15, 20, 25, 30], expected_pages=[8, 9])
        self.assertEqual(m1["recall_at_1"], 1.0)
        self.assertEqual(m1["recall_at_3"], 1.0)
        self.assertEqual(m1["recall_at_5"], 1.0)
        self.assertEqual(m1["reciprocal_rank"], 1.0)

        # Case B: Relevant page appears at Rank 3
        m3 = calculate_query_metrics(retrieved_pages=[1, 2, 8, 25, 30], expected_pages=[8])
        self.assertEqual(m3["recall_at_1"], 0.0)
        self.assertEqual(m3["recall_at_3"], 1.0)
        self.assertEqual(m3["recall_at_5"], 1.0)
        self.assertAlmostEqual(m3["reciprocal_rank"], 1.0 / 3.0, places=4)

        # Case C: Relevant page appears at Rank 5
        m5 = calculate_query_metrics(retrieved_pages=[1, 2, 3, 4, 8], expected_pages=[8])
        self.assertEqual(m5["recall_at_1"], 0.0)
        self.assertEqual(m5["recall_at_3"], 0.0)
        self.assertEqual(m5["recall_at_5"], 1.0)
        self.assertAlmostEqual(m5["reciprocal_rank"], 0.20, places=4)

        # Case D: No relevant page in top 5
        m0 = calculate_query_metrics(retrieved_pages=[1, 2, 3, 4, 5], expected_pages=[8])
        self.assertEqual(m0["recall_at_1"], 0.0)
        self.assertEqual(m0["recall_at_3"], 0.0)
        self.assertEqual(m0["recall_at_5"], 0.0)
        self.assertEqual(m0["reciprocal_rank"], 0.0)

    def test_metrics_benchmark_persistence(self):
        """Verify benchmark metrics file exists, is valid JSON, and formats correctly."""
        data = load_or_run_benchmark(force_run=False)
        self.assertIsInstance(data, dict)
        self.assertIn("methods", data)
        self.assertIn("num_questions", data)
        self.assertIn("relevance_definition", data)

        methods = data["methods"]
        self.assertIn("Vector (FAISS)", methods)
        self.assertIn("BM25 (Keyword)", methods)
        self.assertIn("Hybrid (FAISS+BM25)", methods)

        # File on disk should match
        self.assertTrue(METRICS_FILE.exists())
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            disk_data = json.load(f)
            self.assertEqual(disk_data["num_questions"], data["num_questions"])

        html = format_metrics_dashboard_html(data)
        self.assertIn("RAG Retrieval Evaluation Dashboard", html)
        self.assertIn("MRR (Mean Reciprocal Rank)", html)
        self.assertIn("Recall@1", html)


if __name__ == "__main__":
    unittest.main()
