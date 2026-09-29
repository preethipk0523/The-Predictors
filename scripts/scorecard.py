"""
Assignment 2 - Scorecard + LLM-as-judge + regression check (RECOMMENDED).
Run it right AFTER `python scripts/evaluate.py`. It reads data/eval_results.json and adds
what the toolkit asks for that evaluate.py does not print:

  * breakdown by difficulty (easy / medium / hard) and by category
  * answer correctness with an LLM judge (--judge): correct / partial / wrong
  * false refusals: real questions the system wrongly refused
  * list of failed questions WITH the reason: RETRIEVAL miss vs GENERATION error
  * regression check: every run is saved in data/eval_history/, and compared with the
    previous run -> which questions got FIXED and which BROKE

Usage:
    python scripts/evaluate.py            # 1) run the pipeline on all questions
    python scripts/scorecard.py --judge   # 2) grade answers (~45 small calls, ~$0.005) + report
    python scripts/scorecard.py           #    report only, no API call
"""
import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import DATA_DIR, CHAT_MODEL  # noqa: E402

RESULTS = DATA_DIR / "eval_results.json"
HISTORY = DATA_DIR / "eval_history"
ANSWERABLE = {"factual", "cross-reference", "stress-test"}
REFUSAL_PHRASES = ["i don't have information", "not covered", "not mentioned",
                   "outside the scope", "does not contain"]


def is_refusal(answer):
    return any(p in (answer or "").lower() for p in REFUSAL_PHRASES)


def judge_all(rows):
    from openai import OpenAI
    from app.config import OPENAI_API_KEY
    from app.cost_tracker import cost_tracker
    client = OpenAI(api_key=OPENAI_API_KEY)
    for r in rows:
        if r["category"] not in ANSWERABLE or not r.get("expected_answer"):
            continue
        resp = client.chat.completions.create(
            model=CHAT_MODEL, temperature=0, max_tokens=3,
            messages=[
                {"role": "system", "content":
                    "You grade a study assistant. Compare the ANSWER with the EXPECTED answer. "
                    "Reply with ONE word: correct (all key facts present, nothing contradicting), "
                    "partial (some key facts missing), or wrong (incorrect, or refused although "
                    "the expected answer exists)."},
                {"role": "user", "content":
                    f"QUESTION: {r['question']}\nEXPECTED: {r['expected_answer']}\nANSWER: {r['generated_answer']}"},
            ])
        cost_tracker.log_chat(resp.usage.prompt_tokens, resp.usage.completion_tokens)
        word = resp.choices[0].message.content.strip().lower().strip(".")
        r["judge"] = word if word in {"correct", "partial", "wrong"} else "wrong"
        print(f"  judged [{r['id']:02d}] {r['judge']}")


def status(r):
    """One pass/fail verdict per question, used for the failure list and the regression check."""
    cat = r["category"]
    if cat == "out-of-scope":
        return ("pass", "") if is_refusal(r["generated_answer"]) else ("fail", "GENERATION: answered an out-of-scope question")
    if cat == "ambiguous":
        return ("pass", "") if r.get("ambiguity_ok") else ("fail", "GENERATION: did not flag the ambiguity")
    unanswerable = r.get("sub_type") == "on_topic_unanswerable"
    if unanswerable:
        return ("pass", "") if is_refusal(r["generated_answer"]) else ("fail", "GENERATION: invented an answer the book does not give")
    if not r["retrieval_eval"]["hit"]:
        return "fail", "RETRIEVAL: expected page(s) not in top-k"
    if is_refusal(r["generated_answer"]):
        return "fail", "GENERATION: false refusal (right page retrieved, still refused)"
    if r.get("judge") in {"wrong", "partial"}:
        return "fail", f"GENERATION: answer judged {r['judge']} although retrieval was OK"
    return "pass", ""


def pct(n, d):
    return f"{100 * n / d:5.1f}% ({n}/{d})" if d else "   n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true", help="grade answers with gpt-4o-mini")
    args = ap.parse_args()

    rows = json.loads(RESULTS.read_text(encoding="utf-8"))
    if args.judge:
        print("Judging answers ...")
        judge_all(rows)
        RESULTS.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    for r in rows:
        r["status"], r["reason"] = status(r)

    print("\n" + "=" * 70 + "\nSCORECARD\n" + "=" * 70)
    ans = [r for r in rows if r["category"] in ANSWERABLE and r.get("sub_type") != "on_topic_unanswerable"]
    print("Retrieval hit (answerable)  :", pct(sum(r["retrieval_eval"]["hit"] for r in ans), len(ans)))
    judged = [r for r in ans if "judge" in r]
    if judged:
        print("Answers judged correct      :", pct(sum(r["judge"] == "correct" for r in judged), len(judged)))
    print("False refusals              :", pct(sum(is_refusal(r["generated_answer"]) for r in ans), len(ans)))
    oos = [r for r in rows if r["category"] == "out-of-scope"]
    print("Out-of-scope refused (text) :", pct(sum(is_refusal(r["generated_answer"]) for r in oos), len(oos)))
    pri = [r for r in rows if r.get("priority")]
    print("Priority questions passing  :", pct(sum(r["status"] == "pass" for r in pri), len(pri)))
    print("OVERALL PASS                :", pct(sum(r["status"] == "pass" for r in rows), len(rows)))

    for field in ("category", "difficulty"):
        groups = defaultdict(list)
        for r in rows:
            groups[r.get(field)].append(r)
        print(f"\nBy {field}:")
        for g, rs in sorted(groups.items(), key=lambda x: str(x[0])):
            a = [r for r in rs if r["category"] in ANSWERABLE]
            ret = pct(sum(r["retrieval_eval"]["hit"] for r in a), len(a)) if a else "   n/a"
            print(f"  {str(g):16} pass {pct(sum(r['status'] == 'pass' for r in rs), len(rs))} | retrieval {ret}")

    fails = [r for r in rows if r["status"] == "fail"]
    print(f"\nFAILED QUESTIONS ({len(fails)}) -> debug with: python scripts/retrieval_checker.py --id N --answer")
    for r in fails:
        print(f"  [{r['id']:02d}] {r['reason']}\n       {r['question'][:85]}")

    # ---------- regression check ----------
    HISTORY.mkdir(exist_ok=True)
    previous = sorted(HISTORY.glob("run_*.json"))
    if previous:
        old = {r["question"]: r for r in json.loads(previous[-1].read_text(encoding="utf-8"))}
        fixed = [r for r in rows if r["question"] in old and old[r["question"]].get("status") == "fail" and r["status"] == "pass"]
        broke = [r for r in rows if r["question"] in old and old[r["question"]].get("status") == "pass" and r["status"] == "fail"]
        print(f"\nREGRESSION vs {previous[-1].name}: {len(fixed)} fixed, {len(broke)} broken")
        for r in fixed:
            print(f"  ✅ FIXED  [{r['id']:02d}] {r['question'][:80]}")
        for r in broke:
            print(f"  ⚠️ BROKEN [{r['id']:02d}] {r['question'][:80]}")
    snap = HISTORY / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json"
    snap.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved snapshot {snap.relative_to(DATA_DIR.parent)} (next run compares against it)")


if __name__ == "__main__":
    main()
