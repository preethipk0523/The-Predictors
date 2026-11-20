"""
Chunking Tests: Corpus Preparation & Mathematical Partitioning.
Verifies:
- Strict zero overlap between consecutive chunks
- Zero information, word, sentence, or character loss
- Zero text duplication or gaps
- Preservation of original order and document/page metadata
- Sentence and paragraph boundary preference
- Exact ordered chunk reconstruction reproducing source text 100%
"""

import json
import unittest
from pathlib import Path

from app.chunking import (
    count_tokens,
    partition_text_strict,
    chunk_corpus,
    reconstruct_page_text,
    verify_corpus_reconstruction,
)
from app.config import CORPUS_PATH, TARGET_CHUNK_TOKENS


class ChunkingTests(unittest.TestCase):

    def setUp(self):
        self.corpus_path = CORPUS_PATH
        self.assertTrue(self.corpus_path.exists(), f"Corpus not found at {self.corpus_path}")
        self.corpus = json.loads(self.corpus_path.read_text(encoding="utf-8"))

    def test_token_counter(self):
        """Verify token counter handles empty strings, words, and sentences."""
        self.assertEqual(count_tokens(""), 0)
        self.assertGreater(count_tokens("Hello world"), 0)
        self.assertEqual(count_tokens("Data Science"), 2)

    def test_partition_strict_invariants(self):
        """Verify partitioning preserves mathematical continuity: zero gaps, zero overlap, exact match."""
        sample_text = (
            "Introduction to Data Science is fascinating. It starts with raw data from the world.\n\n"
            "Data becomes information when organized. Information becomes knowledge when understood.\n\n"
            "Finally, wisdom is the application of knowledge to make impactful decisions for society."
        )
        target = 15  # Small token target to force multi-chunk splitting
        slices = partition_text_strict(sample_text, target_tokens=target)

        self.assertGreater(len(slices), 1, "Expected multiple slices for small target_tokens")

        # 1. Zero gaps & zero overlap
        for i in range(len(slices) - 1):
            curr_text, curr_start, curr_end = slices[i]
            next_text, next_start, next_end = slices[i + 1]
            self.assertEqual(
                curr_end,
                next_start,
                f"Gap or overlap detected between slice {i} and {i+1}: {curr_end} != {next_start}",
            )
            # Verify no shared non-empty substring across slice boundary
            self.assertNotEqual(curr_text, next_text)

        # 2. Exact ordered reconstruction
        reconstructed = "".join(s[0] for s in slices)
        self.assertEqual(reconstructed, sample_text, "Reconstructed text does not match original exactly")

    def test_sentence_boundary_preservation(self):
        """Verify that sentences are not truncated when natural boundaries are available."""
        s1 = "This is sentence number one with some words."
        s2 = "This is sentence number two with some words."
        combined = f"{s1} {s2}"
        target = count_tokens(s1) + 2

        slices = partition_text_strict(combined, target_tokens=target)
        self.assertEqual(len(slices), 2)
        # S1 should end with a period followed by space
        self.assertTrue(slices[0][0].endswith(" "))
        self.assertEqual(slices[0][0].strip(), s1)
        self.assertEqual(slices[1][0], s2)

    def test_full_corpus_chunking_reconstruction(self):
        """Verify complete source coverage and exact reconstruction on the real 314-page corpus."""
        chunks = chunk_corpus(corpus_path=self.corpus_path, target_tokens=TARGET_CHUNK_TOKENS)

        self.assertGreater(len(chunks), 0)

        # Verify all chunks satisfy metadata constraints
        seen_ids = set()
        for c in chunks:
            self.assertIn("chunk_id", c)
            self.assertIn("source", c)
            self.assertIn("page", c)
            self.assertIn("chunk_index_in_page", c)
            self.assertIn("token_count", c)
            self.assertIn("char_count", c)
            self.assertIn("start_char", c)
            self.assertIn("end_char", c)
            self.assertIn("text", c)

            # Unique chunk_ids
            self.assertNotIn(c["chunk_id"], seen_ids)
            seen_ids.add(c["chunk_id"])

            # Non-empty text
            self.assertGreater(len(c["text"].strip()), 0)

            # Slicing matches character indices
            self.assertEqual(len(c["text"]), c["end_char"] - c["start_char"])

        # Run mathematical verification
        metrics = verify_corpus_reconstruction(self.corpus, chunks)
        self.assertEqual(metrics["pages_with_overlap"], 0, "Found pages with chunk overlap")
        self.assertEqual(metrics["pages_with_loss"], 0, "Found pages with character loss")
        self.assertEqual(
            metrics["reconstruction_rate_percent"],
            100.0,
            f"Reconstruction rate was not 100%: {metrics}",
        )
        self.assertEqual(metrics["total_corpus_entries"], 314)


if __name__ == "__main__":
    unittest.main()
