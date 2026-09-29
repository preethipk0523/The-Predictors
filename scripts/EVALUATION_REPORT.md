# Assignment 1 & 2: Comprehensive Evaluation Report

## 1. Executive Summary
- **Corpus**: *The Crystal Ball Instruction Manual (Volume One: Introduction to Data Science, v1.2)*
- **Evaluated Questions**: 45 total test cases from `data/questions.json`
- **Evaluation Duration**: 99.10 seconds
- **Overall Retrieval Recall@7 (Hit Rate)**: **82.9%**
- **Full Context Retrieval Rate**: **68.6%**
- **Out-of-Scope Correct Refusal Rate**: **100.0%**
- **Ambiguity Handling Rate**: **40.0%**
- **Average Query Latency**: **2201.9 ms**
- **Total Evaluation Spend**: **$0.02205**
- **Total Team Session Spend**: **$0.02673 / $5.00** (0.53% of budget used)

---

## 2. Category Performance Breakdown

| Category | Questions | Retrieval Hit Rate | Refusal / Clarification Rate | Citations Present | Avg Latency (ms) | Category Cost ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **factual** | 12 | 91.7% | 100.0% | 91.7% | 2357.0 ms | $0.00573 |
| **cross-reference** | 9 | 88.9% | 100.0% | 88.9% | 2526.2 ms | $0.00456 |
| **stress-test** | 14 | 71.4% | 100.0% | 92.9% | 2070.3 ms | $0.00672 |
| **out-of-scope** | 5 | 100.0% | 100.0% | 100.0% | 980.8 ms | $0.00215 |
| **ambiguous** | 5 | 100.0% | 40.0% | 100.0% | 2836.1 ms | $0.00289 |

---

## 3. Priority Questions Evaluation (Core Use Case)
The 5 designated priority questions represent the most critical user interactions:

| ID | Category | Question | Retrieval Hit | Verified Correct | Latency |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | factual | What are the four levels of the data-to-wisdom hierarchy described in Chapter 1? | ✅ Yes | ✅ Passed | 3298 ms |
| 13 | cross-reference | How does the conceptual idea of an associative array in Chapter 7 transition into Pandas Series in Chapter 11? | ✅ Yes | ✅ Passed | 2868 ms |
| 22 | stress-test | How does the author explain the difference between a predictor feature and a target variable in classification? | ✅ Yes | ✅ Passed | 2967 ms |
| 36 | out-of-scope | What is the backpropagation algorithm for multi-layer neural networks and how are weight gradients computed? | ✅ Yes | ✅ Refused | 915 ms |
| 41 | ambiguous | How do you evaluate performance? | ✅ Yes | ✅ Clarified | 2711 ms |

---

## 4. Key Strengths & Observations
1. **Accurate Grounding with Citations**: The system consistently includes explicit `[Page X]` citations for factual questions, pointing users to the exact physical page in the textbook.
2. **Robust Out-of-Scope Refusal**: Queries concerning subjects outside the corpus (e.g., neural network backpropagation, GDP figures, international cricket) are reliably refused with *"I don't have information about this in the corpus."*, avoiding hallucinations.
3. **Ambiguity Resolution**: Queries with underspecified terms (e.g., *"How do I filter?"*) prompt the assistant to explain the ambiguity, provide answers for both DataFrame filtering and array outlier filtering, and offer clarification.
4. **Strict Budget Adherence**: Complete evaluation of all test cases used minimal tokens, leaving over 99% of the $5.00 team budget intact.
