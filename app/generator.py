"""
Generator module for RAG application using gpt-4o-mini.
Context-grounded answer generation with citation verification and hallucination guards.
Enhanced with:
- Strict context grounding prompt with attribution enforcement
- Flexible citation parser supporting single, comma-separated, hyphenated ranges, and parentheses
- Hallucination guard: validates that cited pages exist in retrieved chunks
- Grounding verification score: measures lexical overlap between generated response and context
- Early refusal guard for definitively out-of-scope / zero-relevance queries
"""

import re
import time
from typing import List, Dict, Any, Tuple, Set
from openai import OpenAI

from app.config import (
    OPENAI_API_KEY,
    CHAT_MODEL,
    SIMILARITY_THRESHOLD,
)
from app.cost_tracker import cost_tracker

SYSTEM_PROMPT = """You are an expert AI assistant and tutor for "The Crystal Ball Instruction Manual" by Stephen Davies.

GUIDELINES:
1. STRICT GROUNDING:
   - Answer the question accurately, clearly, and concisely based strictly and solely on the provided corpus excerpts.
   - Ground every factual statement, definition, method name, or calculation in the provided text.
   - Do NOT use outside world knowledge or introduce concepts not present in the excerpts.

2. CITATION SUPPORT:
   - Include explicit page citations like [Page X] or [Page X, Y] for factual statements, definitions, and code syntax.
   - ONLY cite page numbers that are explicitly listed in the provided excerpts. Never guess or fabricate a page number.
   - At the end of your response, list the sources cited.

3. OUT-OF-SCOPE REFUSAL & UNANSWERABLE QUESTIONS:
   - If the question is outside the scope of this textbook (e.g., deep neural network backpropagation, quantum algorithms, external history) OR the provided excerpts do not contain the specific information requested (e.g. a formula not in the text), state:
     "I don't have information about this in the corpus."
   - For on-topic unanswerable questions (e.g., asking for a formula the book doesn't provide), explain what the book does cover on that topic, and state that the specific formula or detail is not provided in Volume One.

4. AMBIGUOUS QUESTIONS:
   - If the question is vague, broad, or ambiguous (e.g., "How do I filter?", "Which scale is best?"), explicitly explain the ambiguity and describe what the book covers for each interpretation, and invite clarification.
"""


def parse_citations(text: str) -> List[int]:
    """
    Extracts cited page numbers from text across various formats:
    - [Page 42], [Excerpt 1], [Pages 42, 43], [Pages 42-44], (Page 42)
    """
    cited_pages: List[int] = []

    # 1. Bracketed single pages: [Page 42] or [Excerpt 42]
    for match in re.finditer(r"\[(?:Page|Excerpt)\s*(\d+)\]", text, re.IGNORECASE):
        p = int(match.group(1))
        if p not in cited_pages:
            cited_pages.append(p)

    # 2. Comma-separated or listed pages: [Pages 8, 10, 11]
    for match in re.finditer(r"\[Pages?\s*([\d\s,]+)\]", text, re.IGNORECASE):
        for part in match.group(1).split(","):
            part_clean = part.strip()
            if part_clean.isdigit():
                p = int(part_clean)
                if p not in cited_pages:
                    cited_pages.append(p)

    # 3. Hyphenated page ranges: [Pages 42-45] or [Page 42-45]
    for match in re.finditer(r"\[Pages?\s*(\d+)\s*[-–—]\s*(\d+)\]", text, re.IGNORECASE):
        start_p = int(match.group(1))
        end_p = int(match.group(2))
        if start_p <= end_p and (end_p - start_p) <= 10:
            for p in range(start_p, end_p + 1):
                if p not in cited_pages:
                    cited_pages.append(p)

    # 4. Parenthetical pages: (Page 42) or (Pages 42-44)
    for match in re.finditer(r"\((?:Page|Pages)\s*(\d+)\)", text, re.IGNORECASE):
        p = int(match.group(1))
        if p not in cited_pages:
            cited_pages.append(p)

    return sorted(cited_pages)


def calculate_grounding_score(answer: str, retrieved_chunks: List[Dict[str, Any]]) -> float:
    """
    Measures lexical grounding: calculates what fraction of significant content words
    in the answer are supported by the retrieved chunk texts.
    """
    if not answer or not retrieved_chunks:
        return 0.0

    # Stopwords to ignore
    stopwords = {
        "this", "that", "these", "those", "with", "from", "have", "been",
        "were", "will", "would", "could", "should", "there", "their", "about",
        "which", "what", "where", "when", "also", "into", "more", "most",
        "some", "such", "than", "then", "other", "after", "before", "between",
    }

    # Extract answer content words (length >= 4, alphanumeric)
    words = re.findall(r"\b[a-zA-Z]{4,}\b", answer.lower())
    content_words = [w for w in words if w not in stopwords]

    if not content_words:
        return 1.0

    # Build context vocabulary
    context_text = " ".join(c.get("text", "") for c in retrieved_chunks).lower()
    context_vocab = set(re.findall(r"\b[a-zA-Z]{4,}\b", context_text))

    supported_count = sum(1 for w in content_words if w in context_vocab)
    return round(supported_count / len(content_words), 3)


def build_context_block(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Builds a formatted string of context passages for the LLM."""
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        source = chunk.get("source", "crystal1 (1).pdf")
        page = chunk.get("page", "?")
        text = chunk.get("text", "").strip()
        context_parts.append(
            f"--- EXCERPT {i} [Source: {source} | Page: {page}] ---\n{text}\n"
        )
    return "\n".join(context_parts)


class AnswerGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Generates an answer using gpt-4o-mini with citation extraction,
        hallucination validation, grounding verification, and cost tracking.
        """
        retrieved_pages = {c.get("page") for c in retrieved_chunks if c.get("page") is not None}
        max_dense = max((c.get("dense_score", 0.0) for c in retrieved_chunks), default=0.0)
        max_bm25 = max((c.get("bm25_score", 0.0) for c in retrieved_chunks), default=0.0)

        # Early exit for definitively out-of-scope queries (saves API cost & guarantees refusal)
        if not retrieved_chunks or (max_dense < 0.16 and max_bm25 == 0.0):
            return {
                "answer": "I don't have information about this in the corpus.",
                "cited_pages": [],
                "verified_citations": [],
                "hallucinated_citations": [],
                "is_refusal": True,
                "grounding_score": 1.0,
                "generation_time_ms": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
            }

        context_text = build_context_block(retrieved_chunks)
        user_content = f"""Corpus Excerpts:
{context_text}

User Question: {query}"""

        start_time = time.time()

        resp = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=650,
        )

        gen_time_ms = round((time.time() - start_time) * 1000, 2)
        prompt_tokens = resp.usage.prompt_tokens
        completion_tokens = resp.usage.completion_tokens
        cost = cost_tracker.log_chat(prompt_tokens, completion_tokens)

        answer_text = resp.choices[0].message.content.strip()

        # Parse citations
        all_cited_pages = parse_citations(answer_text)

        # Hallucination Guard: Separate verified from ungrounded citations
        verified_citations = [p for p in all_cited_pages if p in retrieved_pages]
        hallucinated_citations = [p for p in all_cited_pages if p not in retrieved_pages]

        # Calculate lexical grounding score
        grounding_score = calculate_grounding_score(answer_text, retrieved_chunks)

        # Determine refusal / out-of-scope status
        refusal_keywords = [
            "i don't have information about this in the corpus",
            "i don't have information about this",
            "not mentioned in the corpus",
            "not covered in the provided",
            "not found in the provided",
            "does not contain information",
            "not provide information",
        ]
        is_refusal = any(kw in answer_text.lower() for kw in refusal_keywords)

        return {
            "answer": answer_text,
            "cited_pages": verified_citations,
            "raw_citations": all_cited_pages,
            "verified_citations": verified_citations,
            "hallucinated_citations": hallucinated_citations,
            "is_refusal": is_refusal,
            "grounding_score": grounding_score,
            "generation_time_ms": gen_time_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost,
        }


# Global singleton instance
generator = AnswerGenerator()
