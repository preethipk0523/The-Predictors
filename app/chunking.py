"""
Chunking module for RAG application.
Implements token-aware, paragraph-preserving text chunking with overlap.
"""

import json
from typing import List, Dict, Any
from pathlib import Path
import tiktoken
from app.config import (
    CORPUS_PATH,
    TARGET_CHUNK_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    MIN_CHUNK_CHARS,
)

tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def chunk_corpus(
    corpus_path: Path = CORPUS_PATH,
    target_tokens: int = TARGET_CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> List[Dict[str, Any]]:
    """
    Chunks the extracted corpus into coherent, token-bounded chunks
    preserving document source and page metadata.
    """
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    chunks = []
    chunk_id = 0

    for entry in corpus:
        source = entry["source"]
        page = entry["page"]
        text = entry.get("text", "").strip()

        if len(text) < MIN_CHUNK_CHARS:
            continue

        page_token_count = count_tokens(text)

        # If the entire page fits comfortably within target tokens, keep it as a unit chunk
        if page_token_count <= target_tokens + (overlap_tokens // 2):
            chunks.append({
                "chunk_id": chunk_id,
                "source": source,
                "page": page,
                "token_count": page_token_count,
                "text": text,
            })
            chunk_id += 1
            continue

        # Otherwise, split by paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        current_paras = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = count_tokens(para)

            # If a single paragraph is larger than target, split it by sentence or lines
            if para_tokens > target_tokens:
                lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
                for line in lines:
                    line_tokens = count_tokens(line)
                    if current_tokens + line_tokens > target_tokens and current_paras:
                        chunk_text = "\n\n".join(current_paras)
                        chunks.append({
                            "chunk_id": chunk_id,
                            "source": source,
                            "page": page,
                            "token_count": count_tokens(chunk_text),
                            "text": chunk_text,
                        })
                        chunk_id += 1
                        # Maintain overlap
                        overlap_paras = []
                        overlap_count = 0
                        for prev_p in reversed(current_paras):
                            p_cnt = count_tokens(prev_p)
                            if overlap_count + p_cnt <= overlap_tokens:
                                overlap_paras.insert(0, prev_p)
                                overlap_count += p_cnt
                            else:
                                break
                        current_paras = overlap_paras
                        current_tokens = overlap_count

                    current_paras.append(line)
                    current_tokens += line_tokens
                continue

            if current_tokens + para_tokens > target_tokens and current_paras:
                chunk_text = "\n\n".join(current_paras)
                chunks.append({
                    "chunk_id": chunk_id,
                    "source": source,
                    "page": page,
                    "token_count": count_tokens(chunk_text),
                    "text": chunk_text,
                })
                chunk_id += 1

                # Carry over overlap
                overlap_paras = []
                overlap_count = 0
                for prev_p in reversed(current_paras):
                    p_cnt = count_tokens(prev_p)
                    if overlap_count + p_cnt <= overlap_tokens:
                        overlap_paras.insert(0, prev_p)
                        overlap_count += p_cnt
                    else:
                        break
                current_paras = overlap_paras
                current_tokens = overlap_count

            current_paras.append(para)
            current_tokens += para_tokens

        if current_paras:
            chunk_text = "\n\n".join(current_paras)
            chunks.append({
                "chunk_id": chunk_id,
                "source": source,
                "page": page,
                "token_count": count_tokens(chunk_text),
                "text": chunk_text,
            })
            chunk_id += 1

    return chunks
