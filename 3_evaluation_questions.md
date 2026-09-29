# Assignment 1: Evaluation Questions

> **Goal:** Prepare a set of test questions to evaluate your RAG system during the hackathon.
>
> **Deadline:** before Kaggle week

---

## Why this matters

On Demo Day, your system will be tested with real questions, including tricky ones. Teams that prepare good test questions early can systematically find and fix problems during the build phase. Without test questions, you're guessing whether your system works.

This is your quality assurance toolkit. The better your questions, the better your final system.

> ⚠️ Write these questions yourself by reading your corpus. Do not use GenAI tools for this assignment. The point is to deeply understand your documents.

---

## The task

Read through your corpus and prepare **40 to 50 questions** across five categories:

> The per-category minimums below add up to 33. Distribute the remaining questions across any categories to reach 40 to 50, weighting them toward your core use case (the questions real users would actually ask).

> 💡 Split the work between team members, each person can own a category. Just make sure someone reviews the full set for consistency before submitting.

### 1. Factual questions (minimum 10)

Questions that have a clear, verifiable answer found in a specific part of your corpus. Use `"category": "factual"` in the JSON.

- Vary the difficulty: some simple lookups, some requiring reasoning within a section
- Include questions that need precise answers (numbers, dates, names)

**Example** (for a FIFA Laws of the Game corpus):
| Question | Expected answer | Source | Page |
|----------|----------------|--------|------|
| How long is a standard football match? | 90 minutes (two 45-minute halves) | laws_of_the_game.pdf | 47 |
| What is the minimum number of players a team must have to continue a match? | 7 players | laws_of_the_game.pdf | 30 |

### 2. Cross-reference questions (minimum 8)

Questions where the answer requires information from multiple places. Use `"category": "cross-reference"`. Cover at least 5 of the following 6 sub-types (at least one question each):

**a) Adjacent chunks** (`"adjacent_chunks"`): the answer spans two consecutive paragraphs or sections in the same chapter. Tests whether your chunking and overlap preserve context at boundaries.

**b) Same document, different sections** (`"same_document"`): the answer combines information from e.g. chapter 2 and chapter 8 of the same PDF.

**c) Different documents** (`"different_documents"`): the answer requires combining information from two different source files (e.g. a rule in one PDF and an exception in another).

**d) Repeated information** (`"repeated"`): the same fact appears in multiple places. Tests whether the system finds the most complete or authoritative version.

**e) Contradictory information** (`"contradictory"`): two sources give slightly different answers (e.g. different dates or numbers). Tests how the system handles conflicts. If your corpus has no contradictions, note that and skip this one.

**f) Summary vs detail** (`"summary_vs_detail"`): one place has a brief mention, another has the full explanation. Tests whether retrieval finds the detailed version.

**Example:**
| Question | Sub-type | Expected answer | Sources |
|----------|----------|----------------|---------|
| How do the penalty rules differ between regular play and knockout rounds? | same document, different sections | Regular play: penalty kick from the mark. Knockout: penalty shootout with specific rules. | laws_of_the_game.pdf p.47 + p.52 |
| How do player eligibility rules differ between domestic and international competitions? | different documents | Domestic: club registration required. International: nationality + FIFA eligibility. | laws_of_the_game.pdf p.30 + competition_rules.pdf p.12 |
| What is the maximum number of substitutions allowed? | repeated information | 5 substitutions in 3 windows. | laws_of_the_game.pdf p.30 + competition_rules.pdf p.8 |

In your `questions.json`, add a `"sub_type"` field for cross-reference questions. Allowed values: `"adjacent_chunks"`, `"same_document"`, `"different_documents"`, `"repeated"`, `"contradictory"`, `"summary_vs_detail"`.

### 3. Retrieval stress tests (minimum 10)

Questions designed to expose specific weaknesses in your RAG pipeline. Use `"category": "stress-test"`. Cover at least 8 of the following 11 sub-types (at least one question each):

**a) Paraphrased** (`"paraphrased"`): ask the same thing using very different wording from the corpus. Tests embedding quality.
> Corpus says "candidates must hold a bachelor's degree" → Question: "What diploma do I need to apply?"

**b) Exact term lookup** (`"exact_term"`): questions with specific codes, article numbers, or acronyms. Tests whether vector search alone is enough.
> "What does Article 12.3 say?"

**c) Yes/No** (`"yes_no"`): forces the system to give a direct answer, not just dump a chunk.
> "Is remote attendance allowed?"

**d) Temporal** (`"temporal"`): questions about dates, deadlines, or sequences. Tests retrieval of time-specific information.
> "What changed between the 2023 and 2024 versions?"

**e) Negation** (`"negation"`): questions about what is NOT allowed or NOT included. Retrieval often finds the positive case instead.
> "What are students NOT allowed to bring to the exam?"

**f) Multi-hop reasoning** (`"multi_hop"`): answer requires chaining two facts together.
> "If a student fails course A, can they still enroll in course B?" (requires: A is prerequisite for B + failure policy)

**g) Table/list extraction** (`"table_list"`): questions whose answer is in a table or bullet list. Tests whether chunking preserved structured content.
> "What are the 5 evaluation criteria and their weights?"

**h) Edge-of-chunk** (`"edge_of_chunk"`): questions whose answer likely sits at the boundary between two chunks. Tests your overlap strategy. Pick a fact that appears mid-paragraph in a long section.

**i) Language variation** (`"language_variation"`): ask using informal, slang, or differently-styled language than the corpus. Tests embedding robustness.
> Corpus (formal): "Late submissions incur a 10% penalty per day." → Question: "What's the deal with late submissions?"

**j) On-topic but unanswerable** (`"on_topic_unanswerable"`): the question is about your corpus topic, but the specific answer isn't in the documents. More subtle than out-of-scope, tests whether the system hallucinates a plausible-sounding answer.
> Corpus covers 2024 rules → "What were the rules in 2019?"

**k) Multi-part question** (`"multi_part"`): a single question with multiple parts. Tests whether the system addresses all parts or just the last one.
> "What are the prerequisites for course A, how many credits is it, and when does it start?"

In your `questions.json`, add a `"sub_type"` field for stress test questions. Allowed values: `"paraphrased"`, `"exact_term"`, `"yes_no"`, `"temporal"`, `"negation"`, `"multi_hop"`, `"table_list"`, `"edge_of_chunk"`, `"language_variation"`, `"on_topic_unanswerable"`, `"multi_part"`.

### 4. Out-of-scope questions (minimum 3)

Questions your system should NOT answer, they're outside your corpus. Use `"category": "out-of-scope"`. A good system says "I don't have information about this" instead of making something up.

**Example:**
| Question | Expected behavior |
|----------|------------------|
| What's the weather in Paris today? | Refuse / say out of scope |
| Who won the 2024 Ballon d'Or? | Refuse / say out of scope |

### 5. Ambiguous questions (minimum 2)

Questions with vague or unclear phrasing. Use `"category": "ambiguous"`. A good system should either ask for clarification or provide a broad answer acknowledging the ambiguity.

**Example:**
| Question | Why it's ambiguous |
|----------|-------------------|
| Tell me about the requirements. | Which requirements? Multiple sections cover different requirements. |
| What are the rules about hands? | Could refer to handball, throw-in, or goalkeeper rules. |

---

## Expected output format

Save your questions as `data/questions.json`. Use the same `source` and `page` fields as your `corpus.json` so you can programmatically verify answers later.

The `category` field must be exactly one of: `"factual"`, `"cross-reference"`, `"stress-test"`, `"out-of-scope"`, `"ambiguous"`. The `page` field may be a single integer, or a list of integers when the answer spans multiple pages (as in the cross-reference example below).

```json
[
  {
    "question": "How long is a standard football match?",
    "expected_answer": "90 minutes (two 45-minute halves)",
    "source": "laws_of_the_game.pdf",
    "page": 47,
    "category": "factual",
    "difficulty": "easy",
    "priority": true,
    "expected_chunks": [
      {"source": "laws_of_the_game.pdf", "page": 47}
    ]
  },
  {
    "question": "How do the penalty rules differ between regular play and knockout rounds?",
    "expected_answer": "Regular play: penalty kick from the mark. Knockout: penalty shootout with specific rules.",
    "source": "laws_of_the_game.pdf",
    "page": [47, 52],
    "category": "cross-reference",
    "sub_type": "same_document",
    "difficulty": "hard",
    "priority": false,
    "expected_chunks": [
      {"source": "laws_of_the_game.pdf", "page": 47},
      {"source": "laws_of_the_game.pdf", "page": 52}
    ]
  },
  {
    "question": "What diploma do I need to apply?",
    "expected_answer": "Candidates must hold a bachelor's degree.",
    "source": "program_guide.pdf",
    "page": 3,
    "category": "stress-test",
    "sub_type": "paraphrased",
    "difficulty": "medium",
    "priority": false,
    "expected_chunks": [
      {"source": "program_guide.pdf", "page": 3}
    ]
  },
  {
    "question": "What's the weather in Paris today?",
    "expected_answer": null,
    "source": null,
    "page": null,
    "category": "out-of-scope",
    "difficulty": "easy",
    "priority": false,
    "expected_chunks": null
  },
  {
    "question": "Tell me about the requirements.",
    "expected_answer": null,
    "source": null,
    "page": null,
    "category": "ambiguous",
    "difficulty": "medium",
    "priority": false,
    "expected_chunks": null
  }
]
```

---

## Metadata

For every question, add:

- **`difficulty`**: `"easy"`, `"medium"`, or `"hard"`, how hard you expect this question to be for your RAG system (not for a human). Think about: will the system find the right passage? Is the wording very different from the corpus? Does it need multiple pieces of information?
- **`priority`**: `true` for the **5 most important questions** a real user would ask about your corpus. These represent your core use case, if your system can't answer these, it's not useful.
- **`expected_chunks`**: for factual, cross-reference, and stress test questions, note which specific passages should be retrieved. Use source + page. This lets you later verify: "did my system retrieve the right chunks?" For out-of-scope and ambiguous questions, set to `null`.

All fields are shown in the JSON example above.

---

## Tips

- Read your corpus carefully, the best test questions come from actually knowing your documents
- For factual questions, write down the exact expected answer and where it is in the corpus
- Think about what a real user would ask about your topic
- A good factual question is specific and has one clear answer, avoid vague questions like "Tell me about football"

---

## Checklist

Before submitting, verify:

- [ ] 40 to 50 questions total
- [ ] At least 10 factual, 8 cross-reference (covering at least 5 sub-types), 10 stress tests (covering at least 8 sub-types), 3 out-of-scope, 2 ambiguous
- [ ] Every factual, cross-reference, and stress test question has an expected answer, source, and page
- [ ] Cross-reference and stress test questions have a `sub_type` field
- [ ] Every question has a `difficulty` field (`easy`, `medium`, or `hard`)
- [ ] Exactly 5 questions are marked `"priority": true`
- [ ] Factual, cross-reference, and stress test questions have `expected_chunks`
- [ ] Source and page values match your `corpus.json` entries
- [ ] Questions are diverse (not all from the same section/document)
- [ ] `data/questions.json` follows the expected format

---

## Submission

1. Upload `questions.json` to the `data/` folder in your Teams channel's Files tab
2. Post a message in your private channel with:
   - Number of questions per category
