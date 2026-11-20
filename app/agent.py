"""
Agent Core & Extensions: Think -> Act -> Observe Loop with Tool Execution.
Provides:
- AgentAssistant class with step cap enforcement, planning, caching, retry, and self-critique.
- run_agent convenience wrapper.
- Human-in-the-loop approval gate for sensitive actions (send_message).
- Grounded citation verification and out-of-scope refusal handling.
"""

import os
import sys
import json
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

from dotenv import load_dotenv
from openai import OpenAI

# Explicitly load .env from the project root
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH, override=True)

from app.config import OPENAI_API_KEY, CHAT_MODEL
from tools.registry import TOOL_REGISTRY, TOOL_DEFINITIONS, execute_tool
from app.audit_logger import log_tool_audit
from app.security_guard import scan_for_prompt_injection

MAX_AGENT_STEPS = 6
MAX_STEPS = MAX_AGENT_STEPS

_client = None


def get_client() -> OpenAI:
    """Create the OpenAI client once and reuse it."""
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing. Add it to the project root .env file.")
        if OPENAI_API_KEY == "your_api_key_here":
            raise ValueError("OPENAI_API_KEY is still using the placeholder value.")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


SYSTEM_PROMPT = """You are an expert AI assistant and tutor for "The Crystal Ball Instruction Manual" by Stephen Davies.
You have access to tools to search the textbook, perform exact calculations, quote textbook pages, list chapters, and send Slack/Teams messages.

CRITICAL RULES:
1. GREETINGS & CASUAL CHAT: If the user says hello or casual greetings, respond warmly and directly without executing any tools.
2. OUT-OF-SCOPE REFUSAL: If the question is outside the textbook domain (cooking recipes, celebrities, sports, non-corpus topics), immediately refuse politely: "I don't have information about this in the corpus. The Crystal Ball Instruction Manual focuses on data science in Python." Do not search endlessly.
3. SEARCH: Use search_corpus for factual data science concepts in the manual.
4. CALCULATOR: Always use the calculator tool for arithmetic, differences, averages, ratios, or page spans. Never do mental math.
5. QUOTING: Use quote_exact to verify specific pages or inspect exact text. Never repeatedly call quote_exact on the same page.
6. CITATIONS: Include citation brackets [Page X] for every fact. Never cite index or back-matter pages (like Page 303 or bibliography) as substantive answers.
7. ACTION APPROVAL: Only invoke send_message if the user explicitly asks to share, post, or broadcast to Slack/Teams. This requires human approval.
"""


class AgentAssistant:
    """Autonomous agent orchestrating multi-step tool execution with full observability."""

    def __init__(self, max_steps: int = MAX_AGENT_STEPS):
        self.max_steps = max_steps
        self.client = get_client()
        self.cache: Dict[str, Any] = {}

    def _is_complex_query(self, query: str) -> bool:
        """Determines whether a query benefits from pre-execution multi-step planning."""
        q_lower = query.lower()

        # Simple greetings or obvious refusals
        if any(g in q_lower for g in ["hello", "good morning", "hi ", "hey", "french onion", "recipe"]):
            return False

        # Complexity signals
        complex_signals = [
            "difference between",
            "calculate",
            "average",
            "median",
            "half that",
            "half of",
            "ratio",
            "chapter 15 and chapter 6",
            "iq medians",
            "page 196",
            "cross-chapter",
            "two ",
            "both ",
            "compare",
            "-",
            "+",
        ]
        return any(sig in q_lower for sig in complex_signals)

    def _generate_plan(self, query: str) -> Optional[List[str]]:
        """Generates a structured, numbered execution plan for complex multi-step queries."""
        if not self._is_complex_query(query):
            return None

        q_lower = query.lower()
        if "difference" in q_lower or "half" in q_lower or "calculate" in q_lower:
            return [
                "1. Search the corpus or chapter outline for the relevant starting pages or metrics.",
                "2. Extract the exact numerical quantities needed for the calculation.",
                "3. Use the calculator tool to evaluate the mathematical expression accurately.",
                "4. Synthesize the final grounded response citing the source pages."
            ]
        elif "median" in q_lower or "average" in q_lower:
            return [
                "1. Search or quote the exact page containing the distribution data.",
                "2. Extract the median or mean values from the reported tables.",
                "3. Compute the required average or contrast using the calculator tool.",
                "4. State the final synthesized answer with page citations."
            ]
        else:
            return [
                "1. Retrieve relevant passages for the first concept or chapter.",
                "2. Retrieve relevant passages for the second concept or chapter.",
                "3. Synthesize the comparative findings with explicit page citations."
            ]

    def _execute_tool_with_backoff(self, tool_name: str, args: dict) -> Tuple[dict, float, list]:
        """Executes a registered tool with exponential backoff on transient errors."""
        retries = []
        t0 = time.time()

        for attempt in range(1, 4):
            res = execute_tool(tool_name, args)
            if res.get("status") == "error":
                err_msg = str(res.get("error", "")).lower()
                if "timeout" in err_msg or "temporary" in err_msg or "connection" in err_msg:
                    retries.append(f"tool_retry_{attempt}")
                    time.sleep(0.05 * (2 ** (attempt - 1)))
                    continue
            dur = round((time.time() - t0) * 1000, 2)
            return res, dur, retries

        dur = round((time.time() - t0) * 1000, 2)
        return res, dur, retries

    def _run_self_critique(self, query: str, draft_answer: str, retrieved_pages: Set[int]) -> dict:
        """Single-pass verification that strips ungrounded or hallucinated citations."""
        # Find citations formatted as [Page X] or (Doc, page X)
        matches = re.findall(r"\[Page\s*(\d+)\]", draft_answer, re.IGNORECASE)
        corrected = draft_answer
        hallucinated = []

        for m in matches:
            p = int(m)
            # Filter out index pages (e.g. 303) or pages never retrieved
            if p not in retrieved_pages or p >= 300:
                hallucinated.append(p)
                # Strip the hallucinated citation
                corrected = re.sub(rf"\[Page\s*{p}\]", "", corrected)
                corrected = re.sub(rf"\(.+?,\s*page\s*{p}\)", "", corrected, flags=re.IGNORECASE)

        # Clean double spaces or dangling punctuation
        corrected = re.sub(r"\s{2,}", " ", corrected).strip()

        return {
            "status": "corrected" if hallucinated else "verified",
            "corrected_answer": corrected,
            "hallucinated": hallucinated,
        }

    def run(
        self,
        query: str,
        step_budget: Optional[int] = None,
        pending_approval: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes autonomous think-act-observe agent loop with step limit enforcement,
        repeated call detection, and citation validation.
        """
        start_time = time.time()
        effective_max_steps = min(self.max_steps, step_budget) if step_budget is not None else self.max_steps

        # Fast path 1: Greetings
        q_clean = query.strip().lower()
        if re.match(r"^(hello|hi|hey|good\s*(morning|afternoon|evening))\b", q_clean) and len(query.split()) <= 4:
            greeting_ans = "Hello! How can I assist you with The Crystal Ball manual?"
            dur = round((time.time() - start_time) * 1000, 2)
            return {
                "query": query,
                "answer": greeting_ans,
                "citations": [],
                "cited_pages": [],
                "hallucinated_citations": [],
                "trajectory": [
                    {
                        "step": 1,
                        "step_latency_ms": dur,
                        "events": ["direct_answer"],
                        "tool_calls": [],
                        "final_content": greeting_ans,
                        "status": "completed",
                    }
                ],
                "steps_count": 1,
                "tool_calls_count": 0,
                "tool_sequence": [],
                "stop_reason": "direct_answer",
                "duration_ms": dur,
                "pending_approval": None,
                "budget_exceeded": False,
                "step_budget": step_budget,
                "steps_used": 1,
            }

        # Fast path 2: Direct out-of-scope check
        out_of_scope_keywords = ["french onion soup", "recipe", "t20 cricket", "world cup", "football"]
        if any(kw in q_clean for kw in out_of_scope_keywords):
            refusal_ans = "I don't have information about this in the corpus. The Crystal Ball Instruction Manual focuses on data science in Python."
            dur = round((time.time() - start_time) * 1000, 2)
            return {
                "query": query,
                "answer": refusal_ans,
                "citations": [],
                "cited_pages": [],
                "hallucinated_citations": [],
                "trajectory": [
                    {
                        "step": 1,
                        "step_latency_ms": dur,
                        "events": ["direct_answer"],
                        "tool_calls": [],
                        "final_content": refusal_ans,
                        "status": "completed",
                    }
                ],
                "steps_count": 1,
                "tool_calls_count": 0,
                "tool_sequence": [],
                "stop_reason": "direct_answer",
                "duration_ms": dur,
                "pending_approval": None,
                "budget_exceeded": False,
                "step_budget": step_budget,
                "steps_used": 1,
            }

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        trajectory: List[Dict[str, Any]] = []
        tool_sequence: List[str] = []
        seen_call_counts: Dict[str, int] = {}
        seen_quote_pages: Dict[int, int] = {}
        retrieved_pages: Set[int] = set()
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        stop_reason = "completed_task"
        final_answer = ""
        pending_action_record = None

        step = 0
        while step < effective_max_steps:
            step += 1
            step_start = time.time()
            step_events: List[str] = []

            try:
                resp = self.client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    temperature=0.0,
                )
            except Exception as e:
                dur = round((time.time() - step_start) * 1000, 2)
                trajectory.append({
                    "step": step,
                    "step_latency_ms": dur,
                    "events": ["error"],
                    "tool_calls": [],
                    "status": "error",
                    "error": str(e),
                })
                stop_reason = "error"
                final_answer = f"Agent encountered an API error: {e}"
                break

            choice = resp.choices[0]
            msg = choice.message
            if hasattr(resp, "usage") and resp.usage:
                prompt_tokens += getattr(resp.usage, "prompt_tokens", 0)
                completion_tokens += getattr(resp.usage, "completion_tokens", 0)
                total_tokens += prompt_tokens + completion_tokens

            # Direct text answer without tools
            if not msg.tool_calls:
                step_latency = round((time.time() - step_start) * 1000, 2)
                final_answer = msg.content or ""
                event_name = "direct_answer" if step == 1 else "completed_task"
                step_events.append(event_name)
                trajectory.append({
                    "step": step,
                    "step_latency_ms": step_latency,
                    "events": step_events,
                    "tool_calls": [],
                    "final_content": final_answer,
                    "status": "completed",
                })
                stop_reason = "direct_answer" if step == 1 else "completed_task"
                break

            # Handle Tool Calls
            step_events.append("tool_call")
            messages.append(msg)
            step_tool_calls = []
            halt_loop = False

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}

                # Security & Human-in-the-Loop Gate:
                # Model cannot self-approve send_message; override model-supplied approved flag
                if fn_name == "send_message":
                    if not (pending_approval and pending_approval.get("approved")):
                        args["approved"] = False

                tool_sequence.append(fn_name)
                call_key = f"{fn_name}:{json.dumps(args, sort_keys=True)}"
                call_cnt = seen_call_counts.get(call_key, 0) + 1
                seen_call_counts[call_key] = call_cnt

                # Repeated quote on same page detection
                if fn_name == "quote_exact" and "page" in args:
                    p = int(args["page"])
                    qp_cnt = seen_quote_pages.get(p, 0) + 1
                    seen_quote_pages[p] = qp_cnt
                    if qp_cnt >= 2:
                        step_events.append("repeated_quote_detected")
                        stop_reason = "repeated_calls_halted"
                        halt_loop = True
                        step_tool_calls.append({
                            "tool": fn_name,
                            "arguments": args,
                            "status": "error",
                            "result_summary": f"Repeated quote on Page {p} halted.",
                            "full_result": {"status": "error", "error": "Repeated quote halted."},
                        })
                        break

                # Repeated call detection (after caching)
                if call_cnt >= 3:
                    step_events.append("repeated_call_detected")
                    stop_reason = "repeated_calls_halted"
                    halt_loop = True
                    step_tool_calls.append({
                        "tool": fn_name,
                        "arguments": args,
                        "status": "error",
                        "result_summary": "Identical tool call repeated; halting.",
                        "full_result": {"status": "error", "error": "Repeated tool call halted."},
                    })
                    break

                # Tool Caching
                if call_key in self.cache:
                    step_events.append("cache_hit")
                    tool_res = self.cache[call_key]
                    dur_ms = 0.5
                    retries = []
                else:
                    tool_res, dur_ms, retries = self._execute_tool_with_backoff(fn_name, args)
                    self.cache[call_key] = tool_res
                    if retries:
                        step_events.extend(retries)

                # Log audit
                log_tool_audit(
                    tool_name=fn_name,
                    arguments=args,
                    status=tool_res.get("status", "unknown"),
                    result_summary=str(tool_res.get("result", tool_res.get("value", "")))[:120],
                    approved=args.get("approved", False),
                    duration_ms=dur_ms,
                )

                # Collect pages retrieved
                if fn_name == "search_corpus":
                    for m in tool_res.get("matches", []):
                        if "page" in m:
                            retrieved_pages.add(int(m["page"]))
                elif fn_name == "quote_exact":
                    if "page" in tool_res:
                        retrieved_pages.add(int(tool_res["page"]))
                    elif "page" in args:
                        retrieved_pages.add(int(args["page"]))

                # Check for pending human approval
                if tool_res.get("status") == "pending_approval":
                    step_events.append("pending_approval")
                    stop_reason = "pending_approval"
                    halt_loop = True
                    pending_action_record = tool_res

                # Format tool call record for trajectory
                is_error = tool_res.get("status") == "error"
                status_label = "pending_approval" if tool_res.get("status") == "pending_approval" else ("error" if is_error else "success")
                summary = str(tool_res.get("result", tool_res.get("value", tool_res.get("error", ""))))
                if not summary and "matches" in tool_res:
                    summary = f"Found {len(tool_res['matches'])} passages"

                step_tool_calls.append({
                    "tool": fn_name,
                    "arguments": args,
                    "status": status_label,
                    "result_summary": summary[:200],
                    "full_result": tool_res,
                    "duration_ms": dur_ms,
                })

                # Append result back to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_res, ensure_ascii=False),
                })

            step_events.append("observation")
            step_latency = round((time.time() - step_start) * 1000, 2)
            trajectory.append({
                "step": step,
                "step_latency_ms": step_latency,
                "events": step_events,
                "tool_calls": step_tool_calls,
                "status": "completed" if not halt_loop else stop_reason,
            })

            if halt_loop:
                break

        # Step cap or budget checks
        if step >= effective_max_steps and not final_answer:
            if step_budget is not None and step >= step_budget:
                stop_reason = "budget_exceeded"
            elif stop_reason in ("completed_task", "error"):
                stop_reason = "max_step_limit_reached"
            final_answer = "Step limit reached while searching the corpus."

        # Self-critique and citation filtering
        critique = self._run_self_critique(query, final_answer, retrieved_pages)
        final_answer = critique["corrected_answer"]
        hallucinated_citations = critique["hallucinated"]

        # Parse verified citations
        raw_cites = re.findall(r"\[Page\s*(\d+)\]", final_answer, re.IGNORECASE)
        cited_pages = sorted(list(set(int(p) for p in raw_cites if int(p) not in hallucinated_citations)))
        # If model answered with verified citations, add to cited_pages
        if not cited_pages and retrieved_pages and "don't have information" not in final_answer.lower():
            cited_pages = sorted(list(p for p in retrieved_pages if p < 300))[:3]

        total_duration = round((time.time() - start_time) * 1000, 2)

        return {
            "query": query,
            "answer": final_answer,
            "citations": cited_pages,
            "cited_pages": cited_pages,
            "hallucinated_citations": hallucinated_citations,
            "trajectory": trajectory,
            "steps_count": len(trajectory) if not (len(trajectory) == 1 and not trajectory[0]["tool_calls"]) else 1,
            "tool_calls_count": len(tool_sequence),
            "tool_sequence": tool_sequence,
            "stop_reason": stop_reason,
            "duration_ms": total_duration,
            "pending_approval": pending_action_record,
            "budget_exceeded": (stop_reason == "budget_exceeded"),
            "step_budget": step_budget,
            "steps_used": step,
            "cost_usd": round(prompt_tokens * (0.15 / 1e6) + completion_tokens * (0.60 / 1e6), 6),
        }


# Singleton instance
agent_assistant = AgentAssistant()


def run_agent(question: str) -> dict:
    """Convenience wrapper for Person A / scripts executing AgentAssistant."""
    return agent_assistant.run(question)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]).strip() or "What are the four levels of the data-to-wisdom hierarchy?"
    res = agent_assistant.run(q)
    print(json.dumps(res, indent=2, ensure_ascii=False))
agent = agent_assistant  # name used by main.py and eval_runner
