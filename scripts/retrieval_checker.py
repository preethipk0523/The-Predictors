import argparse, json, os, sys

# let this script import from the app/ folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.retriever import retrieve   # CHANGE if your function has a different name

def norm(source):
    return os.path.basename(str(source))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--questions", default="data/questions.json")  # CHANGE to your real path
    args = p.parse_args()

    chunks = retrieve(args.question, args.k)

    print(f"\nQuestion: {args.question}\n")
    for i, c in enumerate(chunks, 1):
        print(f"#{i} | score {c['score']:.3f} | {norm(c['source'])} p.{c['page']}")
        print(f"   {c['text'][:200].strip()}...\n")

    # look up the question in questions.json
    try:
        with open(args.questions, encoding="utf-8") as f:
            questions = json.load(f)
    except FileNotFoundError:
        print(f"{args.questions} not found, skipping check.")
        return

    entry = next((q for q in questions if q["question"].strip() == args.question.strip()), None)
    if not entry:
        print("Question not in questions.json, skipping check.")
        return

    got = {(norm(c["source"]), c["page"]) for c in chunks}
    missed = 0
    print(f"Category: {entry.get('category')} | Difficulty: {entry.get('difficulty')}\n")
    for exp in entry["expected_chunks"]:
        key = (norm(exp["source"]), exp["page"])
        ok = key in got
        missed += not ok
        print("✅" if ok else "❌", f"{key[0]} p.{key[1]}")

    if missed == 0:
        print("\nRetrieval OK. If the answer is wrong, the problem is generation (prompt/LLM).")
    else:
        print(f"\nRetrieval problem: {missed} expected chunk(s) missed. The LLM never saw them.")

if __name__ == "__main__":
    main()