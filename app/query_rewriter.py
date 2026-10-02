"""
Query Rewriter Module.
Determines whether a user query depends on previous conversation context (coreference / follow-up),
and rewrites it into a standalone search query for optimal hybrid retrieval.
"""

import re
from typing import List, Dict, Any, Optional
from openai import OpenAI
from app.config import OPENAI_API_KEY, CHAT_MODEL
from app.cost_tracker import cost_tracker

# Keywords and regex patterns indicating conversational dependency
DEPENDENT_PATTERNS = [
    r"\b(it|its|they|them|their|this|that|these|those)\b",
    r"\b(the second|the third|the first|the latter|the former)\b",
    r"^(what about|how about|and for|why\?|tell me more|explain further|give an example)\b",
    r"\b(same thing|previous|above mentioned|what else)\b",
]


def is_context_dependent(query: str) -> bool:
    """Detects whether a query likely refers to entities from previous conversation turns."""
    q_clean = query.strip().lower()

    # Very short queries are almost always follow-ups
    if len(q_clean.split()) <= 4 and ("why" in q_clean or "how" in q_clean or "what" in q_clean):
        return True

    for pattern in DEPENDENT_PATTERNS:
        if re.search(pattern, q_clean):
            return True

    return False


def rewrite_query(
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    client: Optional[OpenAI] = None,
) -> Dict[str, Any]:
    """
    Evaluates whether the query requires coreference resolution.
    If context-dependent and history exists, rewrites into a standalone search query.

    Returns:
        Dict with keys:
            - is_rewritten: bool
            - query: str (the query to execute: rewritten if modified, else original)
            - original_query: str
            - reason: str (explanation for trace/inspection)
    """
    if not chat_history:
        return {
            "is_rewritten": False,
            "query": query,
            "original_query": query,
            "reason": "Standalone query (no prior conversation history)",
        }

    # Extract user and assistant turns
    recent_dialogue = []
    for turn in chat_history[-6:]:
        role = turn.get("role", "user")
        content = turn.get("content", "").strip()
        if content:
            speaker = "User" if role == "user" else "Assistant"
            recent_dialogue.append(f"{speaker}: {content[:300]}")

    if not recent_dialogue or not is_context_dependent(query):
        return {
            "is_rewritten": False,
            "query": query,
            "original_query": query,
            "reason": "Standalone query (independent concepts)",
        }

    context_str = "\n".join(recent_dialogue)
    prompt = f"""You are a query formulation assistant for a textbook search engine over 'The Crystal Ball Instruction Manual (Introduction to Data Science)'.

Conversation History:
{context_str}

Follow-up User Question: "{query}"

Task:
Determine what specific concept or subject from the conversation history the user is asking about.
Rewrite the follow-up question into a single, complete, self-contained search query.
Do NOT answer the question. Output ONLY the rewritten standalone question.

Rewritten Standalone Question:"""

    try:
        openai_client = client or OpenAI(api_key=OPENAI_API_KEY)
        resp = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )

        if resp.usage:
            cost_tracker.log_chat(resp.usage.prompt_tokens, resp.usage.completion_tokens)

        rewritten = resp.choices[0].message.content.strip()
        rewritten = rewritten.strip('"\'')

        if rewritten and rewritten.lower() != query.lower():
            return {
                "is_rewritten": True,
                "query": rewritten,
                "original_query": query,
                "reason": f"Resolved coreferences from chat context (original: '{query}')",
            }

    except Exception as exc:
        # Fallback to heuristic resolution if LLM is unavailable
        last_turn = recent_dialogue[-1] if recent_dialogue else ""
        return {
            "is_rewritten": False,
            "query": query,
            "original_query": query,
            "reason": f"Query rewriting bypassed due to fallback: {exc}",
        }

    return {
        "is_rewritten": False,
        "query": query,
        "original_query": query,
        "reason": "Query already sufficiently specific",
    }
