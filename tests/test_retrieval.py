"""
Retrieval Evaluation: FAISS, BM25, and Hybrid Ranking Tests.
Verifies:
- Code-aware BM25 tokenization preserving syntax, code methods, and aliases
- Query signal extraction (page numbers, chapter spans, method calls)
- HybridRetriever output format, key consistency, and sorting
- Regression verification on key failure cases (Q1, Q6, Q23, Q30)
"""

import unittest
from app.indexer import tokenize_for_bm25
from app.retriever import retriever, extract_query_signals


class RetrievalEvaluationTests(unittest.TestCase):

    def test_code_aware_bm25_tokenizer(self):
        """Verify that tokenizer preserves code methods, parentheses, and brackets."""
        text = "Call .dropna() and check the bananas () or boxies [] carefully."
        tokens = tokenize_for_bm25(text)

        self.assertIn(".dropna", tokens)
        self.assertIn("()", tokens)
        self.assertIn("[]", tokens)
        self.assertIn("parentheses", tokens)
        self.assertIn("brackets", tokens)

    def test_query_signals_extraction(self):
        """Verify extraction of target page, chapter, and code methods from natural queries."""
        q1 = "Which string methods alter the casing of characters as demonstrated on page 43?"
        s1 = extract_query_signals(q1)
        self.assertEqual(s1["target_page"], 43)
        self.assertNotIn("on page 43", s1["clean_embed"].lower())

        q2 = "What is Simpson's Paradox as illustrated in Chapter 16?"
        s2 = extract_query_signals(q2)
        self.assertEqual(s2["target_chapter"], 16)

        q3 = "How does the method .strip() operate in Python?"
        s3 = extract_query_signals(q3)
        self.assertIn(".strip()", s3["method_calls"])

    def test_retriever_output_schema_and_ordering(self):
        """Verify retriever return types, required dictionary keys, and score sorting."""
        results = retriever.retrieve("What is data science?", top_k=5)

        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 5)
        self.assertGreater(len(results), 0)

        required_keys = {
            "chunk_id",
            "source",
            "page",
            "text",
            "dense_score",
            "bm25_score",
            "combined_score",
            "is_confident",
        }

        prev_score = float("inf")
        for chunk in results:
            for k in required_keys:
                self.assertIn(k, chunk, f"Missing key '{k}' in retrieved chunk")

            score = chunk["combined_score"]
            self.assertLessEqual(
                score,
                prev_score + 1e-6,
                "Retrieved chunks are not sorted descending by combined_score",
            )
            prev_score = score

    def test_verified_hits_on_benchmark_targets(self):
        """Verify targeted benchmark questions achieve expected retrieval hits."""
        # Q1: Data-to-wisdom hierarchy -> Page 8
        r1 = retriever.retrieve("What are the four levels of the data-to-wisdom hierarchy described in Chapter 1?", top_k=7)
        pages1 = [c["page"] for c in r1]
        self.assertIn(8, pages1, "Failed to retrieve Page 8 for Q1")

        # Q6: Simpson's Paradox -> Page 177
        r6 = retriever.retrieve("What is Simpson's Paradox as illustrated in Chapter 16?", top_k=7)
        pages6 = [c["page"] for c in r6]
        self.assertIn(177, pages6, "Failed to retrieve Page 177 for Q6")

        # Q23: Informal nickname for parentheses () -> Page 43
        r23 = retriever.retrieve("What informal nickname does the author use for parentheses () throughout the book?", top_k=7)
        pages23 = [c["page"] for c in r23]
        self.assertIn(43, pages23, "Failed to retrieve Page 43 for Q23")

        # Q30: String methods on page 43 -> Page 43
        r30 = retriever.retrieve("Which string methods alter the casing of characters as demonstrated on page 43?", top_k=7)
        pages30 = [c["page"] for c in r30]
        self.assertIn(43, pages30, "Failed to retrieve Page 43 for Q30")


if __name__ == "__main__":
    unittest.main()
