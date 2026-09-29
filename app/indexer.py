"""
Indexer module for RAG application.
Generates embeddings using text-embedding-3-small, builds a FAISS IndexFlatIP
(cosine similarity) and a BM25 sparse index, and persists them to disk.
"""

import json
import pickle
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import faiss
import numpy as np
from openai import OpenAI

from app.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    CHUNKS_PATH,
    INDEX_PATH,
    BM25_PATH,
)
from app.chunking import chunk_corpus
from app.cost_tracker import cost_tracker
from rank_bm25 import BM25Okapi


def get_openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set. Please add it to your .env file."
        )
    return OpenAI(api_key=OPENAI_API_KEY)


def tokenize_for_bm25(text: str) -> List[str]:
    """Simple alphanumeric tokenizer for BM25 keyword matching."""
    import re
    return re.findall(r"\w+", text.lower())


def build_and_save_index(force_rebuild: bool = False) -> Tuple[faiss.IndexFlatIP, List[Dict[str, Any]], BM25Okapi]:
    """
    Builds FAISS index and BM25 index and persists to disk.
    If already saved and force_rebuild=False, loads directly from disk.
    """
    if (
        not force_rebuild
        and INDEX_PATH.exists()
        and CHUNKS_PATH.exists()
        and BM25_PATH.exists()
    ):
        print(f"Loading existing FAISS index from {INDEX_PATH} and chunks from {CHUNKS_PATH}...")
        index = faiss.read_index(str(INDEX_PATH))
        chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        with open(BM25_PATH, "rb") as f:
            bm25 = pickle.load(f)
        return index, chunks, bm25

    print("Building new FAISS and BM25 index from corpus...")
    client = get_openai_client()
    chunks = chunk_corpus()
    print(f"Total chunks to embed: {len(chunks)}")

    batch_size = 64
    all_embeddings = []
    total_tokens_used = 0

    start_time = time.time()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]

        resp = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
        total_tokens_used += resp.usage.total_tokens

        for item in resp.data:
            all_embeddings.append(item.embedding)

        print(
            f"  Embedded chunks {i+1} to {min(i+batch_size, len(chunks))} ({resp.usage.total_tokens} tokens)"
        )

    # Log cost
    cost = cost_tracker.log_embedding(total_tokens_used)
    print(f"Embedding completed in {time.time()-start_time:.2f}s. Cost: ${cost:.5f}")

    # Convert to float32 numpy array and normalize for Cosine Similarity (IndexFlatIP)
    embeddings_np = np.array(all_embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings_np)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings_np)

    # Build BM25 sparse index
    corpus_tokens = [tokenize_for_bm25(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)

    # Save to disk
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    CHUNKS_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    print(f"Saved FAISS index to {INDEX_PATH}")
    print(f"Saved chunks metadata to {CHUNKS_PATH}")
    print(f"Saved BM25 index to {BM25_PATH}")

    return index, chunks, bm25


def load_index() -> Tuple[faiss.IndexFlatIP, List[Dict[str, Any]], BM25Okapi]:
    """Loads index, chunks, and BM25 from disk, building if necessary."""
    return build_and_save_index(force_rebuild=False)


if __name__ == "__main__":
    build_and_save_index(force_rebuild=True)
