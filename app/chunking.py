"""
Chunking module for RAG application: token-bounded, zero-overlap text chunking and reconstruction.
Implements token-bounded, mathematically partitioned text chunking with:
- STRICT ZERO OVERLAP between consecutive chunks
- ZERO information, word, sentence, or character loss
- ZERO text duplication or gaps between chunks
- Preservation of original order and document/page metadata
- Sentence and paragraph boundary preference without truncating long sentences
- Exact ordered chunk reconstruction reproducing the source text 100%
"""

import json
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
import tiktoken

from app.config import (
    CORPUS_PATH,
    TARGET_CHUNK_TOKENS,
    CHUNK_OVERLAP_TOKENS,
)

tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Returns exact token count for the given text using cl100k_base encoding."""
    if not text:
        return 0
    return len(tokenizer.encode(text))


def partition_text_strict(
    text: str, target_tokens: int = TARGET_CHUNK_TOKENS
) -> List[Tuple[str, int, int]]:
    """
    Partitions text into contiguous, non-overlapping slices:
    (chunk_text, start_char, end_char).

    Guarantees:
    - slice[k][2] == slice[k+1][1] (Zero gap, zero overlap)
    - "".join(s[0] for s in slices) == text (Zero character or information loss)
    - Boundaries prefer paragraph breaks (\\n\\n), then sentence endings ([.!?]\\s+),
      then line breaks (\\n), then whitespace (\\s+), without truncating sentences
      unless an individual sentence exceeds target_tokens.
    """
    if not text:
        return []

    total_tokens = count_tokens(text)
    if total_tokens <= target_tokens:
        return [(text, 0, len(text))]

    slices: List[Tuple[str, int, int]] = []
    start = 0
    n = len(text)

    while start < n:
        rem_text = text[start:]
        rem_tokens = count_tokens(rem_text)

        if rem_tokens <= target_tokens:
            slices.append((rem_text, start, n))
            break

        # Binary search for maximum character boundary that fits within target_tokens
        low = start + 1
        high = n
        best_pos = low

        while low <= high:
            mid = (low + high) // 2
            toks = count_tokens(text[start:mid])
            if toks <= target_tokens:
                best_pos = mid
                low = mid + 1
            else:
                high = mid - 1

        # Search backwards from best_pos to find the cleanest natural boundary
        window_start = max(start + 1, best_pos - 400)
        sub = text[window_start:best_pos]

        # 1. Paragraph boundary: double newline (\n\n)
        para_matches = list(re.finditer(r"\n{2,}", sub))
        split_point = None

        if para_matches:
            split_point = window_start + para_matches[-1].end()
        else:
            # 2. Sentence boundary: [.!?] followed by whitespace or newline
            sentence_matches = list(re.finditer(r"[.!?](\s+|\n+)", sub))
            if sentence_matches:
                split_point = window_start + sentence_matches[-1].end()
            else:
                # 3. Single line break (\n)
                nl_matches = list(re.finditer(r"\n+", sub))
                if nl_matches:
                    split_point = window_start + nl_matches[-1].end()
                else:
                    # 4. Whitespace (\s+)
                    space_matches = list(re.finditer(r"\s+", sub))
                    if space_matches:
                        split_point = window_start + space_matches[-1].end()
                    else:
                        # Fallback: strictly at token boundary
                        split_point = best_pos

        # Ensure forward progress
        if split_point <= start:
            split_point = max(start + 1, best_pos)

        chunk_slice = text[start:split_point]
        slices.append((chunk_slice, start, split_point))
        start = split_point

    return slices


def chunk_corpus(
    corpus_path: Path = CORPUS_PATH,
    target_tokens: int = TARGET_CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> List[Dict[str, Any]]:
    """
    Chunks the extracted corpus into coherent, token-bounded chunks.
    Preserves document source, page metadata, character positions, and original ordering.
    Strictly enforces zero overlap (overlap_tokens forced to 0).
    """
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    chunks: List[Dict[str, Any]] = []
    chunk_id = 0

    for entry in corpus:
        source = entry["source"]
        page = entry["page"]
        text = entry.get("text", "")

        # Do not drop short pages; if page is completely blank, skip creating empty chunks
        if not text.strip():
            continue

        slices = partition_text_strict(text, target_tokens=target_tokens)

        for chunk_idx_in_page, (chunk_text, start_char, end_char) in enumerate(slices):
            chunk_tokens = count_tokens(chunk_text)
            chunks.append({
                "chunk_id": chunk_id,
                "source": source,
                "page": page,
                "chunk_index_in_page": chunk_idx_in_page,
                "token_count": chunk_tokens,
                "char_count": len(chunk_text),
                "start_char": start_char,
                "end_char": end_char,
                "text": chunk_text,
            })
            chunk_id += 1

    return chunks


def reconstruct_page_text(page_chunks: List[Dict[str, Any]]) -> str:
    """
    Reconstructs exact source page text from its ordered chunk list.
    """
    sorted_chunks = sorted(page_chunks, key=lambda c: c.get("chunk_index_in_page", c.get("chunk_id", 0)))
    return "".join(c["text"] for c in sorted_chunks)


def verify_corpus_reconstruction(
    corpus: List[Dict[str, Any]], chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Verifies that chunks satisfy all strict requirements:
    1. Complete source coverage (every non-empty page in corpus has chunks)
    2. Zero overlap between consecutive chunks
    3. Zero character / information loss
    4. Zero gaps between consecutive chunks
    5. Exact reconstruction of all source page texts
    """
    # Group chunks by (source, page)
    chunks_by_page: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for c in chunks:
        chunks_by_page.setdefault((c["source"], c["page"]), []).append(c)

    exact_matches = 0
    total_pages = len(corpus)
    pages_with_overlap = 0
    pages_with_loss = 0
    empty_pages_count = 0

    for entry in corpus:
        key = (entry["source"], entry["page"])
        original_text = entry.get("text", "")

        if not original_text.strip():
            empty_pages_count += 1
            if key not in chunks_by_page:
                exact_matches += 1
            continue

        page_chunks = chunks_by_page.get(key, [])
        if not page_chunks:
            pages_with_loss += 1
            continue

        # Check consecutive chunk continuity and zero overlap
        page_chunks_sorted = sorted(page_chunks, key=lambda c: c["chunk_index_in_page"])
        has_overlap = False

        for i in range(len(page_chunks_sorted) - 1):
            curr_c = page_chunks_sorted[i]
            next_c = page_chunks_sorted[i + 1]

            # Character boundary check
            if curr_c["end_char"] != next_c["start_char"]:
                has_overlap = True

        if has_overlap:
            pages_with_overlap += 1

        # Check exact string reconstruction
        reconstructed = reconstruct_page_text(page_chunks)
        if reconstructed == original_text:
            exact_matches += 1
        else:
            pages_with_loss += 1

    return {
        "total_corpus_entries": total_pages,
        "empty_corpus_pages": empty_pages_count,
        "pages_with_chunks": len(chunks_by_page),
        "exact_reconstructions": exact_matches,
        "reconstruction_rate_percent": round(exact_matches * 100 / total_pages, 2),
        "pages_with_overlap": pages_with_overlap,
        "pages_with_loss": pages_with_loss,
        "total_chunks_produced": len(chunks),
    }
