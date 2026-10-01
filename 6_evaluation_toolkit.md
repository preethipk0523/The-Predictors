# Assignment 2: Evaluation & Debugging Toolkit

> **Goal:** Build the tools you need to systematically test and improve your RAG system.
>
> **When:** During the build phase (Monday PM – Tuesday). Have this ready before the Tuesday 2PM checkpoint.

---

## Why this matters

A RAG system that "seems to work" on 3 questions will fall apart on Demo Day. The teams that score highest are the ones that tested systematically, found failures, and fixed them. These tools make that process fast and repeatable.

---

## The tools

### 1. Retrieval checker (required)

A script that takes a question, runs retrieval, and shows you exactly what was found.

**Input:** a question string
**Output:**
- The top-k retrieved chunks (text, source, page, similarity score)
- Whether the expected chunks (from your `questions.json`, see [Assignment 1](3_evaluation_questions.md)) were retrieved ✅ or missed ❌

This is your most important debugging tool. When an answer is wrong, run this first, if the right chunks weren't retrieved, the LLM never had a chance.

### 2. Failure log (required)

For each failed question, document in `scripts/FAILURE_LOG.md`:

| Question | Category | Retrieval OK? | Problem | Fix applied | Fixed? |
|----------|----------|:---:|---------|-------------|:---:|
| "What does Article 12.3 say?" | exact_term | ❌ | Vector search missed exact term | Added BM25 hybrid search | ✅ |
| "Is remote attendance allowed?" | yes_no | ✅ | LLM gave a paragraph instead of yes/no | Added "answer yes or no first" to prompt | ✅ |

Update this as you iterate. This is your debugging history, it shows the jury you worked methodically.

### 3. Eval runner (recommended)

A script that runs your full `questions.json` (from Assignment 1) through the pipeline and produces a scorecard.

**For each question, log:**
- The question and category
- Retrieved chunks (source, page, score)
- Whether expected chunks were found (retrieval hit)
- The generated answer
- Whether the answer is correct (manual check or LLM-as-judge)

**Summary output:**
- Total retrieval hit rate (% of questions where expected chunks were in top-k)
- Breakdown by category (factual, cross-reference, stress test, out-of-scope, ambiguous)
- Breakdown by difficulty (easy, medium, hard)
- List of failed questions with reason (retrieval miss vs generation error)

Save results to `data/eval_results.json` so you can compare runs.

### 4. Regression check (recommended)

After every change (new chunk size, updated prompt, added BM25), re-run your eval questions on ALL questions. Check that:
- The question you fixed is now passing
- No previously passing questions broke

If a fix breaks something else, note it in the failure log and decide which tradeoff to make.

---

## Workflow

```
1. Run eval runner → see which questions fail
2. Pick a failed question → run retrieval checker → identify if retrieval or generation problem
3. Apply fix (chunk size, prompt, hybrid search, etc.)
4. Re-run eval runner → check fix worked + no regressions
5. Log in failure log
6. Repeat
```

---

## Deliverables

By the Tuesday 2PM checkpoint:

**Required:**
- [ ] `scripts/retrieval_checker.py`, inspects retrieval for a single question
- [ ] `scripts/FAILURE_LOG.md`, documented failures and fixes

**Recommended:**
- [ ] `scripts/eval_runner.py`, runs all questions, produces scorecard
- [ ] `data/eval_results.json`, latest eval run results

---

## Tips

- Run the eval runner after every significant change, don't wait until the end
- Focus on your 5 priority questions first, if those don't work, nothing else matters
- A retrieval hit rate above 80% on factual questions is a good target
- Don't optimize for edge cases before your core questions work
