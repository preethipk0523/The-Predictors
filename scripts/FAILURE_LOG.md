# Failure Log — The Predictors

Workflow for every entry:
`python scripts/evaluate.py` → `python scripts/scorecard.py` → pick a failure →
`python scripts/retrieval_checker.py --id N --answer` → retrieval or generation problem? →
apply ONE fix → re-run evaluate + scorecard (regression check) → log it here.

**Baseline config (v1):** chunks 400 tokens / 60 overlap · hybrid 0.65 dense + 0.35 BM25 · top-k 7 · refusal threshold 0.20 (generator) · gpt-4o-mini, temperature 0.

## Failures and fixes

| # | Question (id) | Category / sub-type | Retrieval OK? | Problem | Fix applied | Fixed? |
|---|---------------|---------------------|:---:|---------|-------------|:---:|
| 1 | All out-of-scope questions (code review) | out-of-scope | n/a | Evaluation counted a question as "refused" whenever the best dense score was < 0.20, even if the LLM had actually written an answer. The UI still showed that answer, so the 100% refusal rate could hide hallucinations. The generator's 0.20 cut-off also ignored `SIMILARITY_THRESHOLD` (0.25) in `config.py`. | Generator now returns the refusal sentence **without calling the LLM** when the best dense score is below `SIMILARITY_THRESHOLD`; `evaluate_refusal` now checks the answer text only. | ⬜ re-run eval |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |

## Run history

| Run | Date | Change | Retrieval hit (all) | Factual hit | Out-of-scope refused | False refusals | Ambiguity handled | Cost |
|-----|------|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| v1 | 29 Sep | Baseline (EVALUATION_REPORT.md) | 82.9% | 91.7% | 100%* | ? | 40% | $0.022 |
| v2 | | Fix #1 (refusal logic) | | | | | | |

\* measured with the old refusal logic, see failure #1.
