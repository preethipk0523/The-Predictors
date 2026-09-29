"""
Assignment 2 - Retrieval checker (REQUIRED).
Shows exactly what the hybrid retriever finds for ONE question, and whether the
expected pages from data/questions.json were retrieved.

Usage (from the repo root):
    python scripts/retrieval_checker.py "What is Simpson's Paradox?"
    python scripts/retrieval_checker.py --id 12            # question #12 in questions.json (1-based, like evaluate.py)
    python scripts/retrieval_checker.py --id 12 -k 10      # look deeper than the default top-k
    python scripts/retrieval_checker.py --id 12 --answer   # also generate the answer (1 chat call)

Read the result like this:
    expected page ❌ missed  -> RETRIEVAL problem (chunking, k, BM25 weight, wording)
    expected page ✅ found but the answer is wrong -> GENERATION problem (prompt)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import QUESTIONS_PATH, TOP_K, SIMILARITY_THRESHOLD  # noqa: E402
from app.retriever import retriever  # noqa: E402


def expected_pages_of(item):
    pages = set()
    for c in item.get("expected_chunks") or []:
        p = c.get("page")
        pages.update(p if isinstance(p, list) else [p])
    return sorted(p for p in pages if p is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", help="a free-text question")
    ap.add_argument("--id", type=int, help="1-based index of a question in data/questions.json")
    ap.add_argument("-k", type=int, default=TOP_K, help=f"chunks to retrieve (default {TOP_K})")
    ap.add_argument("--answer", action="store_true", help="also run the generator")
    args = ap.parse_args()

    item, question = None, args.question
    if args.id is not None:
        questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
        item = questions[args.id - 1]
        question = item["question"]
    if not question:
        ap.error("give a question string or --id N")

    print("=" * 90)
    print(f"QUESTION : {question}")
    if item:
        print(f"CATEGORY : {item['category']} / {item.get('sub_type', '-')} | "
              f"difficulty {item.get('difficulty')} | priority {item.get('priority')}")
        print(f"EXPECTED : {item.get('expected_answer')}")
    print("=" * 90)

    chunks = retriever.retrieve(question, top_k=args.k)
    best_dense = max((c["dense_score"] for c in chunks), default=0.0)
    print(f"Best dense (cosine) score: {best_dense:.3f}   "
          f"-> {'ABOVE' if best_dense >= SIMILARITY_THRESHOLD else 'BELOW'} threshold {SIMILARITY_THRESHOLD}\n")

    for rank, c in enumerate(chunks, 1):
        print(f"#{rank}  page {c['page']:>3} | combined {c['combined_score']:.3f} "
              f"(dense {c['dense_score']:.3f}, bm25 {c['bm25_score']:.3f}) | {c['source']}")
        print("     " + c["text"][:300].replace("\n", " ") + " ...\n")

    if item:
        exp = expected_pages_of(item)
        got = {c["page"] for c in chunks}
        if exp:
            print("EXPECTED PAGES:")
            for p in exp:
                rank = next((r for r, c in enumerate(chunks, 1) if c["page"] == p), None)
                print(f"  page {p}: " + (f"✅ retrieved at rank {rank}" if rank else "❌ MISSED"))
            verdict = "ALL FOUND" if set(exp) <= got else ("PARTIAL" if set(exp) & got else "NONE FOUND")
            print(f"\nRETRIEVAL: {verdict}")
        else:
            print("No expected pages (out-of-scope / ambiguous): the answer should refuse or ask to clarify.")

    if args.answer:
        from app.generator import generator
        res = generator.generate(question, chunks)
        print("\n" + "-" * 90 + "\nANSWER:\n" + res["answer"])
        print(f"\ncited pages: {res['cited_pages']} | refusal: {res['is_refusal']} | cost ${res['cost_usd']:.5f}")


if __name__ == "__main__":
    main()
