"""
Unit tests for RAG Extensions:
1. Chat Memory (multi-turn retention)
2. Query Rewriting (coreference resolution)
3. Reranking (Cross-Encoder & configurable fallback)
4. Confidence Score (High / Medium / Low indicator)
5. Faithfulness Verification (context support check)
6. Feedback Saving (thumbs up / down local persistence)
7. Markdown Export (chat download formatting)
"""

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

from app.chat_memory import ChatMemory
from app.query_rewriter import is_context_dependent, rewrite_query
from app.reranker import CrossEncoderReranker
from app.confidence import calculate_retrieval_confidence
from app.faithfulness import verify_faithfulness
from app.feedback import save_feedback, get_feedback_records
from app.config import FEEDBACK_FILE


class RAGExtensionTests(unittest.TestCase):

    def test_chat_memory_retention_and_trimming(self):
        """Verify that chat memory retains the last 3-5 conversation turns and trims excess."""
        memory = ChatMemory(max_turns=3)
        # Add 5 complete turns
        for i in range(1, 6):
            memory.add_turn(f"Question {i}", f"Answer {i}", citations=[i])

        recent = memory.get_recent_messages()
        # Max turns 3 = 6 messages
        self.assertEqual(len(recent), 6)
        self.assertEqual(recent[0]["content"], "Question 3")
        self.assertEqual(recent[-1]["content"], "Answer 5")

    def test_query_rewriting_context_detection(self):
        """Verify detection of context-dependent follow-up queries."""
        # Context dependent
        self.assertTrue(is_context_dependent("What about its cost?"))
        self.assertTrue(is_context_dependent("What are its applications?"))
        self.assertTrue(is_context_dependent("Why is that?"))
        self.assertTrue(is_context_dependent("How does it work?"))

        # Standalone
        self.assertFalse(is_context_dependent("What is Simpson's Paradox as illustrated in Chapter 16?"))
        self.assertFalse(is_context_dependent("What are the four levels of the data-to-wisdom hierarchy?"))

    def test_query_rewriting_execution_and_bypass(self):
        """Verify query rewriting resolves coreferences with history and bypasses standalone queries."""
        # 1. Standalone query without history
        res_no_hist = rewrite_query("What is data science?", chat_history=[])
        self.assertFalse(res_no_hist["is_rewritten"])
        self.assertEqual(res_no_hist["query"], "What is data science?")

        # 2. Context-dependent query with mock LLM
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="What are the applications of Data Science?"))]
        mock_resp.usage.prompt_tokens = 30
        mock_resp.usage.completion_tokens = 12
        mock_client.chat.completions.create.return_value = mock_resp

        history = [
            {"role": "user", "content": "What is Data Science?"},
            {"role": "assistant", "content": "Data science is the discipline of learning from data."}
        ]
        res_rewritten = rewrite_query("What are its applications?", chat_history=history, client=mock_client)
        self.assertTrue(res_rewritten["is_rewritten"])
        self.assertEqual(res_rewritten["query"], "What are the applications of Data Science?")
        self.assertEqual(res_rewritten["original_query"], "What are its applications?")

    def test_reranker_ordering_and_metadata(self):
        """Verify cross-encoder reranker attaches initial and final ranks and scores."""
        reranker = CrossEncoderReranker(enabled=False, top_n=2)  # Use calibrated fallback mode

        candidates = [
            {"chunk_id": "c1", "text": "Unrelated fruit and vegetable inventory.", "combined_score": 0.82, "dense_score": 0.60},
            {"chunk_id": "c2", "text": "Data science hierarchy: data, information, knowledge, wisdom.", "combined_score": 0.70, "dense_score": 0.55},
            {"chunk_id": "c3", "text": "Wisdom and data processing procedures.", "combined_score": 0.65, "dense_score": 0.50},
        ]

        reranked = reranker.rerank("What is the data-to-wisdom hierarchy?", candidates, top_n=2)
        self.assertEqual(len(reranked), 2)

        # Check metadata attributes
        for c in reranked:
            self.assertIn("initial_rank", c)
            self.assertIn("final_rank", c)
            self.assertIn("rerank_score", c)
            self.assertIn("reranker_engine", c)

        # Candidate with highest term overlap should be promoted to #1
        self.assertEqual(reranked[0]["chunk_id"], "c2")
        self.assertEqual(reranked[0]["final_rank"], 1)

    def test_confidence_score_calculation(self):
        """Verify heuristic retrieval confidence categorization (High / Medium / Low)."""
        # High confidence
        high_chunks = [{"dense_score": 0.65, "combined_score": 0.75}]
        conf_high = calculate_retrieval_confidence(high_chunks)
        self.assertEqual(conf_high["level"], "High")
        self.assertIn("🟢", conf_high["badge"])
        self.assertIn("not a guaranteed probability", conf_high["disclaimer"])

        # Medium confidence
        med_chunks = [{"dense_score": 0.35, "combined_score": 0.45}]
        conf_med = calculate_retrieval_confidence(med_chunks)
        self.assertEqual(conf_med["level"], "Medium")
        self.assertIn("🟡", conf_med["badge"])

        # Low confidence (or empty)
        conf_low = calculate_retrieval_confidence([])
        self.assertEqual(conf_low["level"], "Low")
        self.assertIn("🔴", conf_low["badge"])

    def test_faithfulness_verification(self):
        """Verify faithfulness check on grounded, refusal, and ungrounded statements."""
        # 1. Out-of-scope refusal should pass as faithful refusal
        refusal_ans = "I don't have information about this in the corpus."
        res_refusal = verify_faithfulness("What is French onion soup?", refusal_ans, [])
        self.assertTrue(res_refusal["is_faithful"])
        self.assertEqual(res_refusal["status"], "Faithful Refusal")

        # 2. Supported answer with lexical grounding
        context = [
            {"page": 8, "text": "The data-to-wisdom hierarchy has four levels: data, information, knowledge, and wisdom."}
        ]
        grounded_ans = "The four levels described in the book are data, information, knowledge, and wisdom [Page 8]."
        res_supported = verify_faithfulness("What are the four levels?", grounded_ans, context)
        self.assertTrue(res_supported["is_faithful"])
        self.assertIn(res_supported["status"], ("Faithful", "Supported", "Partially Supported"))

        # 3. Empty evidence should trigger unsupported status
        res_empty = verify_faithfulness("Explain neural nets", "Neural networks use backprop.", [])
        self.assertFalse(res_empty["is_faithful"])
        self.assertIn(res_empty["status"], ("Unsupported", "Insufficient Evidence"))
        self.assertTrue(res_empty["retry_recommended"])

    def test_feedback_saving_and_persistence(self):
        """Verify saving thumbs up and thumbs down ratings to local JSONL."""
        res_up = save_feedback(
            query="What is Gini impurity?",
            answer="Gini impurity measures tree split purity.",
            rating="thumbs_up",
            metadata={"engine": "rag"}
        )
        self.assertEqual(res_up["status"], "success")

        res_down = save_feedback(
            query="What is unknown topic?",
            answer="I do not know.",
            rating="thumbs_down",
            comment="Answer too short"
        )
        self.assertEqual(res_down["status"], "success")

        # Verify disk read
        records = get_feedback_records()
        self.assertGreater(len(records), 0)
        latest = records[-1]
        self.assertEqual(latest["rating"], "thumbs_down")
        self.assertEqual(latest["comment"], "Answer too short")

    def test_markdown_export(self):
        """Verify chat export to formatted Markdown with citations."""
        memory = ChatMemory()
        memory.add_turn(
            "What are the levels of data?",
            "They are data, information, knowledge, and wisdom.",
            citations=[8]
        )

        md_text, file_path = memory.export_to_markdown(session_title="Test Viva Session")
        self.assertTrue(Path(file_path).exists())
        self.assertIn("# Test Viva Session", md_text)
        self.assertIn("👤 User:", md_text)
        self.assertIn("🤖 Predictors Assistant:", md_text)
        self.assertIn("[Page 8]", md_text)


if __name__ == "__main__":
    unittest.main()
