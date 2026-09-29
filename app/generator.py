"""
Generator module for RAG application using gpt-4o-mini.
Constructs context-grounded prompts, generates answers with citations,
and enforces out-of-scope refusal and ambiguity handling.
"""

import re
import time
from typing import List, Dict, Any
from openai import OpenAI

from app.config import (
    OPENAI_API_KEY,
    CHAT_MODEL,
    SIMILARITY_THRESHOLD,
)
from app.cost_tracker import cost_tracker

SYSTEM_PROMPT = """You are an expert AI assistant and tutor for "The Crystal Ball Instruction Manual" by Stephen Davies.

GUIDELINES:
1. ACCURACY & CONCISENESS:
   - Answer the question accurately, clearly, and concisely based strictly on the provided corpus excerpts.
   - Ground every explanation in the provided text.

2. CITATIONS:
   - Include explicit page citations like [Page X] or [Page X, Y] for factual statements, definitions, or code examples.
   - At the end of your response, list the sources cited.

3. OUT-OF-SCOPE REFUSAL:
   - If the question is outside the domain/scope of this textbook or the provided excerpts contain no relevant information to answer it, state:
     "I don't have information about this in the corpus."
   - Do NOT hallucinate or answer out-of-scope questions using external world knowledge.

4. AMBIGUOUS QUESTIONS:
   - If the question is vague, broad, or ambiguous (e.g., "How do I filter?", "Which scale is best?"), explicitly explain the ambiguity and describe what the book covers for each interpretation, and invite clarification if appropriate.
"""


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
        Generates an answer using gpt-4o-mini with citation extraction and cost tracking.
        """
        # Check if all chunks have very low similarity (clear out-of-scope trigger)
        max_dense = max((c.get("dense_score", 0.0) for c in retrieved_chunks), default=0.0)
        
        # If highest similarity is extremely low (< 0.20), it is definitively out-of-scope
        is_definitive_oos = max_dense < 0.20

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

        # Extract cited pages from response (e.g. [Page 42] or [Page 42, 43] or Page 42)
        cited_pages = []
        for match in re.finditer(r"\[(?:Page|Excerpt)\s*(\d+)\]", answer_text, re.IGNORECASE):
            p = int(match.group(1))
            if p not in cited_pages:
                cited_pages.append(p)
        
        # Also catch comma-separated pages like [Page 8, 10, 11]
        for match in re.finditer(r"\[Page\s*([\d\s,]+)\]", answer_text, re.IGNORECASE):
            for part in match.group(1).split(","):
                part_clean = part.strip()
                if part_clean.isdigit():
                    p = int(part_clean)
                    if p not in cited_pages:
                        cited_pages.append(p)

        # Determine refusal / out-of-scope status
        refusal_keywords = [
            "i don't have information about this in the corpus",
            "i don't have information about this",
            "not mentioned in the corpus",
            "not covered in the provided",
            "not found in the provided",
            "does not contain information",
        ]
        is_refusal = any(kw in answer_text.lower() for kw in refusal_keywords) or is_definitive_oos

        return {
            "answer": answer_text,
            "cited_pages": cited_pages,
            "is_refusal": is_refusal,
            "generation_time_ms": gen_time_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost,
        }


# Global singleton instance
generator = AnswerGenerator()
