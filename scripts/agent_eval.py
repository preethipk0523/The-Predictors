"""
Agentic project - evaluate the agent (requirement 11) and read its trajectories.

Runs data/questions.json (Assignment 1) and data/agent_questions.json (the multi-step
questions YOU add) through the agent, then prints a scorecard:
  * pass rate by category, citation of the expected page, out-of-scope refusals
  * tool usage: did the agent call the tools you expected, in that order?
  * steps per question, questions that hit the step cap, cost per question

data/agent_questions.json format (write these yourselves, like Assignment 1):
[
  {"question": "...", "category": "multi-step",
   "expected_tools": ["search_corpus", "calculator"],   # required tools, in order
   "expected_answer": "...", "expected_pages": [8]}
]

Usage:
    python scripts/agent_eval.py --limit 10          # first 10 of questions.json + all agent questions
    python scripts/agent_eval.py --only-agent        # only the multi-step questions
    python scripts/agent_eval.py --show-traces       # also print every tool call
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.agent import MAX_STEPS, REFUSAL_TEXT, run_agent  # noqa: E402
from app.config import DATA_DIR, QUESTIONS_PATH  # noqa: E402

AGENT_Q = DATA_DIR / "agent_questions.json"
OUT = DATA_DIR / "agent_eval_results.json"


def cited_pages(answer):
    pages = set()
    for m in re.finditer(r"\[Pages?\s*([\d,\s-]+)\]", answer or "", re.I):
        pages.update(int(x) for x in re.findall(r"\d+", m.group(1)))
    return pages


def tools_in_order(trace, expected):
    """True if the expected tools appear in the trace in that order (other calls allowed in between)."""
    it = iter(t["tool"] for t in trace)
    return all(any(tool == e for tool in it) for e in expected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only-agent", action="store_true")
    ap.add_argument("--show-traces", action="store_true")
    args = ap.parse_args()

    items = []
    if not args.only_agent:
        qs = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
        items += qs[: args.limit] if args.limit else qs
    if AGENT_Q.exists():
        items += json.loads(AGENT_Q.read_text(encoding="utf-8"))

    rows = []
    for i, q in enumerate(items, 1):
        res = run_agent(q["question"])
        cat = q["category"]
        refused = REFUSAL_TEXT.lower() in (res["answer"] or "").lower()
        exp_pages = set(q.get("expected_pages") or
                        [c["page"] for c in (q.get("expected_chunks") or []) if isinstance(c.get("page"), int)])
        tools = [t["tool"] for t in res["trace"]]
        if cat == "out-of-scope":
            ok = refused
        elif cat == "ambiguous":
            ok = not refused
        elif q.get("sub_type") == "on_topic_unanswerable":
            ok = refused
        else:
            ok = (not refused) and (not exp_pages or bool(exp_pages & cited_pages(res["answer"])))
        if q.get("expected_tools"):
            ok = ok and tools_in_order(res["trace"], q["expected_tools"])
        row = {"id": i, "question": q["question"], "category": cat, "ok": ok, "refused": refused,
               "tools": tools, "steps": res["steps"], "hit_cap": (res["answer"] or "").startswith("I stopped after reaching the step limit"),
               "cost_usd": res["cost_usd"], "answer": res["answer"], "trace": res["trace"]}
        rows.append(row)
        print(f"[{i:02d}] {'✅' if ok else '❌'} {cat:15} steps={res['steps']} tools={tools} :: {q['question'][:60]}")
        if args.show_traces:
            for t in res["trace"]:
                print(f"       -> {t['tool']}({t['arguments']}) => {t['result'][:100]!r}")

    by = defaultdict(list)
    for r in rows:
        by[r["category"]].append(r)
    print("\n=========== AGENT SCORECARD ===========")
    for cat, rs in sorted(by.items()):
        print(f"  {cat:16} {sum(r['ok'] for r in rs)}/{len(rs)} pass | avg steps {sum(r['steps'] for r in rs)/len(rs):.1f}")
    n = len(rows) or 1
    print(f"  TOTAL            {sum(r['ok'] for r in rows)}/{len(rows)} pass")
    print(f"  hit step cap     {sum(r['hit_cap'] for r in rows)}  (fix tool descriptions/prompt, don't raise the cap)")
    print(f"  no tool used     {sum(not r['tools'] for r in rows)}")
    print(f"  total cost       ${sum(r['cost_usd'] for r in rows):.4f} (avg ${sum(r['cost_usd'] for r in rows)/n:.5f}/question)")
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {OUT}. Full trajectories are also in data/trajectories.jsonl")


if __name__ == "__main__":
    main()
