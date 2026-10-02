"""
Faithfulness Verification Module.
Performs post-generation verification to ensure the generated answer is strictly supported
by the retrieved textbook evidence. Detects unsupported claims and incorrect citations.
Classifies results as:
- Faithful
- Partially Supported
- Unsupported
- Faithful Refusal
"""

import re
import json
from typing import List, Dict, Any, Optional, Set
from openai import OpenAI
from app.config import OPENAI_API_KEY, CHAT_MODEL
from app.generator import calculate_grounding_score, parse_citations
from app.cost_tracker import cost_tracker


def verify_faithfulness(
    query: str,
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    client: Optional[OpenAI] = None,
) -> Dict[str, Any]:
    """
    Verifies whether the answer's factual statements and citations are supported by retrieved excerpts.

    Args:
        query: User question.
        answer: Generated answer text.
        retrieved_chunks: Retrieved textbook excerpts.
        client: Optional OpenAI client instance.

    Returns:
        Dict containing:
            - is_faithful (bool)
            - status (str: "Faithful", "Partially Supported", "Unsupported", "Faithful Refusal")
            - score (float: 0.0 to 1.0)
            - badge (str)
            - explanation (str)
            - unsupported_claims (List[str])
            - unsupported_citations (List[int])
            - retry_recommended (bool)
    """
    # 1. Check for standard refusal / out-of-scope response
    refusal_phrases = [
        "i don't have information about this in the corpus",
        "i don't have information about this",
        "not mentioned in the corpus",
        "not covered in the provided",
        "does not contain information",
        "outside the scope",
    ]
    is_refusal = any(p in answer.lower() for p in refusal_phrases)
    if is_refusal:
        return {
            "is_faithful": True,
            "status": "Faithful Refusal",
            "score": 1.0,
            "badge": "🛡️ Faithful Refusal",
            "explanation": "Assistant correctly recognized and refused an out-of-scope or unanswerable query.",
            "unsupported_claims": [],
            "unsupported_citations": [],
            "retry_recommended": False,
        }

    # 2. Check for empty evidence
    if not retrieved_chunks:
        return {
            "is_faithful": False,
            "status": "Unsupported",
            "score": 0.0,
            "badge": "❌ Unsupported",
            "explanation": "No evidence was retrieved to substantiate this answer.",
            "unsupported_claims": ["All claims lack retrieved grounding"],
            "unsupported_citations": parse_citations(answer),
            "retry_recommended": True,
        }

    # 3. Citation validation against retrieved chunk pages
    cited_pages = parse_citations(answer)
    retrieved_pages: Set[int] = {
        int(c.get("page")) for c in retrieved_chunks if c.get("page") is not None
    }
    unsupported_citations = [p for p in cited_pages if p not in retrieved_pages]

    # 4. Lexical grounding score
    lexical_score = calculate_grounding_score(answer, retrieved_chunks)

    # 5. LLM claim-by-claim verification (with robust fallback)
    context_snippets = []
    for idx, c in enumerate(retrieved_chunks[:4], 1):
        p = c.get("page", "?")
        t = c.get("text", "")[:350].strip()
        context_snippets.append(f"Excerpt {idx} [Page {p}]: {t}")
    combined_context = "\n".join(context_snippets)

    verification_prompt = f"""Context from textbook:
{combined_context}

Question: {query}
Generated Answer: {answer}

Task:
Determine whether each factual claim and citation in the Generated Answer is supported by the provided Context.
Classify the status as ONE of:
- "Faithful" (all claims directly supported by context)
- "Partially Supported" (main claim supported, but minor claims or specifics lack context)
- "Unsupported" (key claims contradict or are not found in context)

Respond ONLY in valid JSON with these keys:
{{
  "status": "Faithful", "Partially Supported", or "Unsupported",
  "score": float between 0.0 and 1.0,
  "explanation": "1-2 sentence explanation of why the answer received this status",
  "unsupported_claims": ["list any specific claims not supported by the context"]
}}"""

    status = "Faithful"
    score = 0.95
    explanation = "All claims in the answer are supported by the retrieved textbook passages."
    unsupported_claims: List[str] = []

    try:
        openai_client = client or OpenAI(api_key=OPENAI_API_KEY)
        resp = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": verification_prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=250,
        )

        if resp.usage:
            cost_tracker.log_chat(resp.usage.prompt_tokens, resp.usage.completion_tokens)

        data = json.loads(resp.choices[0].message.content)
        raw_status = str(data.get("status", "Faithful")).strip()

        # Map to canonical statuses
        if "partial" in raw_status.lower():
            status = "Partially Supported"
        elif "unsupported" in raw_status.lower() or "insufficient" in raw_status.lower():
            status = "Unsupported"
        else:
            status = "Faithful"

        score = float(data.get("score", 0.95 if status == "Faithful" else (0.55 if status == "Partially Supported" else 0.15)))
        explanation = str(data.get("explanation", "Verification completed."))
        unsupported_claims = list(data.get("unsupported_claims", []))

    except Exception:
        # Robust heuristic fallback using lexical grounding
        score = lexical_score
        if lexical_score >= 0.55:
            status = "Faithful"
            explanation = f"Lexical alignment confirms answer vocabulary is well-grounded in evidence ({int(lexical_score * 100)}% term overlap)."
            unsupported_claims = []
        elif lexical_score >= 0.28:
            status = "Partially Supported"
            explanation = f"Moderate lexical grounding ({int(lexical_score * 100)}% term overlap). Core concepts present in excerpts."
            unsupported_claims = ["Some details lack explicit phrasing in excerpts"]
        else:
            status = "Unsupported"
            explanation = f"Low lexical alignment ({int(lexical_score * 100)}% match) between answer and retrieved passages."
            unsupported_claims = ["Key claims cannot be verified from retrieved passages"]

    # 6. Apply citation discrepancies to faithfulness status
    if unsupported_citations:
        unsupported_claims.append(f"Answer cited unretrieved page(s): {sorted(unsupported_citations)}")
        if status == "Faithful":
            status = "Partially Supported"
            score = min(score, 0.65)
            explanation += f" Note: Cited page(s) {sorted(unsupported_citations)} were not present in retrieved excerpts."
        elif status == "Partially Supported" and len(unsupported_citations) > 1:
            status = "Unsupported"
            score = min(score, 0.25)
            explanation += f" Multiple unsupported citations detected: {sorted(unsupported_citations)}."

    # UI Badges
    badge_map = {
        "Faithful": "✅ Faithful",
        "Partially Supported": "⚠️ Partially Supported",
        "Unsupported": "❌ Unsupported",
        "Faithful Refusal": "🛡️ Faithful Refusal",
    }

    is_faithful = status in ("Faithful", "Faithful Refusal")
    retry_recommended = (status == "Unsupported")

    return {
        "is_faithful": is_faithful,
        "status": status,
        "score": round(score, 3),
        "badge": badge_map.get(status, "ℹ️ Checked"),
        "explanation": explanation,
        "unsupported_claims": unsupported_claims,
        "unsupported_citations": unsupported_citations,
        "retry_recommended": retry_recommended,
    }
