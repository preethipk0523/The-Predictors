"""
RAG System Tests: Grounding, Citations, and Out-of-Scope Handling.
Verifies:
- Flexible citation parser across single, multiple, comma-separated, and hyphenated formats
- Hallucination guard: validates that citations match retrieved chunks
- Grounding score calculation measuring context support
- Out-of-scope early refusal handling
"""

import unittest
from app.generator import (
    parse_citations,
    calculate_grounding_score,
    generator,
    AnswerGenerator,
)
from app.main import format_chat_answer


class GenerationAndCitationTests(unittest.TestCase):

    def test_citation_parser_formats(self):
        """Verify extraction across diverse citation patterns."""
        t1 = "Data is atomic [Page 19]. Wisdom is the gold standard [Page 12]."
        self.assertEqual(parse_citations(t1), [12, 19])

        t2 = "Refer to the scales of measurement [Pages 49, 50, 54]."
        self.assertEqual(parse_citations(t2), [49, 50, 54])

        t3 = "The methods are discussed in detail [Pages 42-45]."
        self.assertEqual(parse_citations(t3), [42, 43, 44, 45])

        t4 = "Parentheses are called bananas (Page 43)."
        self.assertEqual(parse_citations(t4), [43])

        t5 = "No citations here."
        self.assertEqual(parse_citations(t5), [])

    def test_grounding_score_calculation(self):
        """Verify grounding score computation on supported and unsupported answers."""
        chunks = [
            {"page": 8, "text": "The four levels of the hierarchy are Data, Information, Knowledge, and Wisdom."},
            {"page": 19, "text": "Atomic data types include numeric, categorical, and text data."},
        ]

        # Fully supported answer
        supported_ans = "The hierarchy consists of Data, Information, Knowledge, and Wisdom [Page 8]."
        score_high = calculate_grounding_score(supported_ans, chunks)
        self.assertGreaterEqual(score_high, 0.70)

        # Unsupported answer with hallucinated terms
        unsupported_ans = "Quantum computing uses qubits and entanglement for cryogenic superposition."
        score_low = calculate_grounding_score(unsupported_ans, chunks)
        self.assertLessEqual(score_low, 0.20)

    def test_hallucination_guard_filtering(self):
        """Verify that citations to pages absent from retrieved chunks are flagged as hallucinated."""
        chunks = [
            {"page": 8, "text": "Data and information are fundamental rungs."},
        ]

        # Simulate response with one valid and one hallucinated page
        simulated_text = "Data is fundamental [Page 8], and backpropagation is detailed [Page 99]."
        all_cited = parse_citations(simulated_text)
        retrieved_pages = {c["page"] for c in chunks}

        verified = [p for p in all_cited if p in retrieved_pages]
        hallucinated = [p for p in all_cited if p not in retrieved_pages]

        self.assertEqual(verified, [8])
        self.assertEqual(hallucinated, [99])

    def test_early_refusal_on_empty_chunks(self):
        """Verify that queries with zero retrieved chunks return fast refusal without API overhead."""
        res = generator.generate("What is quantum backpropagation in neural networks?", retrieved_chunks=[])
        self.assertTrue(res["is_refusal"])
        self.assertEqual(res["cost_usd"], 0.0)
        self.assertEqual(res["cited_pages"], [])
        self.assertIn("don't have information", res["answer"].lower())

    def test_citation_validation_against_sources_and_page_303(self):
        """Verify that citations are validated against verified sources and unsupported Page 303 is stripped."""
        text_with_both = "The data-to-wisdom hierarchy includes data and knowledge [Page 8] and listed on index [Page 303]."
        # Only page 8 is verified; page 303 is unverified
        formatted = format_chat_answer(text_with_both, cited_pages=[8])

        # Page 8 should be kept inline and in sources cited
        self.assertIn("[Page 8]", formatted)
        self.assertIn("- [Page 8]", formatted)

        # Unsupported Page 303 must NOT appear inline or in sources cited
        self.assertNotIn("[Page 303]", formatted)
        self.assertNotIn("- [Page 303]", formatted)


if __name__ == "__main__":
    unittest.main()
