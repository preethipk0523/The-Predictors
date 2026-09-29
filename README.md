# Predictors RAG Assistant: The Crystal Ball Manual

An autonomous, production-grade Retrieval-Augmented Generation (RAG) system built **from scratch** (no LangChain, no LlamaIndex, raw Python only) to provide grounded, citation-backed answers to questions about ***The Crystal Ball Instruction Manual (Volume One: Introduction to Data Science, v1.2)*** by Stephen Davies, Ph.D.

Developed by **Team Predictors** for the AI Hackathon.

---

## 🌟 Key Features

1. **Built Completely From Scratch**:
   - Zero reliance on RAG frameworks (no LangChain, no LlamaIndex).
   - Direct integration with `openai` Python SDK, `faiss-cpu`, and `rank_bm25`.

2. **Strictly Compliant Model Architecture**:
   - **Embedding Model**: `text-embedding-3-small` (1536-dimensional vectors, L2 normalized).
   - **Generation Model**: `gpt-4o-mini` (temperature: 0.0 for deterministic factual fidelity).
   - **Vector Index**: FAISS `IndexFlatIP` (exact Cosine Similarity).
   - **Sparse Index**: BM25 Okapi for keyword and exact code symbol retrieval.

3. **Hybrid Search Fusion**:
   - Dynamically blends dense semantic similarity (65% weight) with BM25 keyword matching (35% weight).
   - Accurately captures both conceptual paraphrasing and exact technical keywords (e.g., `.iloc`, `.strip()`, `bananas`, `Simpson's Paradox`).

4. **Rigorous Grounding & Out-of-Scope Protection**:
   - Every factual answer is backed by explicit in-text bracket citations (e.g., `[Page 8]`, `[Page 155]`).
   - Strict confidence thresholding and prompt engineering guarantee that queries outside the textbook scope are refused (*"I don't have information about this in the corpus."*), preventing hallucinations.
   - Ambiguous queries are recognized and answered with multifaceted contextual explanations.

5. **Real-Time Cost & Performance Tracking**:
   - Tracks every embedding and generation token.
   - Calculates exact dollar costs against the **$5.00 team budget** and logs cumulative spend to `data/cost_tracker.json`.

6. **Modern, Responsive Web UI**:
   - Built with Gradio featuring a sleek dark-glassmorphic theme, expandable evidence inspection cards, live latency indicators, and interactive preset query examples.

---

## 📁 Repository Structure

```
predictors/
├── README.md                          # Project overview, installation, and usage
├── .gitignore                         # Excludes .venv/, .env, and data/
├── .env                               # Stores OPENAI_API_KEY
├── 2_corpus_preparation.md           # Assignment 0 instructions
├── 3_evaluation_questions.md          # Assignment 1 instructions
├── 5_rag_project.md                   # Assignment 2 project instructions
├── crystal1 (1).pdf                   # Original textbook source PDF
│
├── scripts/
│   ├── extract.py                     # Document parser & cleaner (PyMuPDF)
│   ├── EXTRACTION_REPORT.md           # Report on extraction stats & issues
│   ├── generate_questions.py          # Builds data/questions.json (45 questions)
│   ├── evaluate.py                    # Evaluates pipeline against questions.json
│   └── EVALUATION_REPORT.md           # Comprehensive evaluation benchmark report
│
├── data/                              # Local data directory (gitignored)
│   ├── sources/
│   │   └── crystal1 (1).pdf           # Corpus source document
│   ├── corpus.json                    # Extracted clean text with page metadata
│   ├── sample.json                    # First 10 entries of corpus
│   ├── chunks.json                    # Chunked text with token counts & metadata
│   ├── faiss_index.bin                # Persisted FAISS vector index
│   ├── bm25.pkl                       # Persisted BM25 keyword index
│   ├── questions.json                 # 45 evaluation questions across 5 categories
│   ├── eval_results.json              # Full evaluation results
│   └── cost_tracker.json              # Cumulative session token & USD spend tracker
│
└── app/
    ├── __init__.py
    ├── config.py                      # Models, paths, hyperparams, and pricing
    ├── chunking.py                    # Token-bounded semantic paragraph chunking
    ├── cost_tracker.py                # Budget & token logging module
    ├── indexer.py                     # Offline FAISS + BM25 indexing engine
    ├── retriever.py                   # Hybrid dense + sparse retriever
    ├── generator.py                   # Context-grounded response generator (gpt-4o-mini)
    ├── pipeline.py                    # End-to-end RAG orchestrator
    └── main.py                        # Gradio Web Application UI
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.12
- [uv](https://github.com/astral-sh/uv) (fast Python package manager)

### 1. Clone & Environment Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd predictors

# Create Python 3.12 virtual environment with uv
uv venv --python 3.12

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
uv pip install openai faiss-cpu pymupdf pdfplumber beautifulsoup4 numpy tiktoken gradio python-dotenv rank_bm25
```

### 2. Configure API Key
Create a `.env` file in the root directory:
```bash
OPENAI_API_KEY=sk-proj-...
```

---

## 🛠️ How to Run

### Step 1: Extract Corpus Text
Extracts clean, structured text page-by-page from `data/sources/` into `data/corpus.json`:
```bash
python scripts/extract.py
```
*Corpus stats:* 314 pages, 382,242 characters, 16 structural blank pages cataloged.

### Step 2: Build Embeddings & FAISS Index (One-Time)
Chunks the corpus and indexes it into FAISS and BM25, saving artifacts to disk:
```bash
python -m app.indexer
```
*Embedding cost:* Embedded 365 chunks (103,761 tokens) in 3.5 seconds for **$0.00208** (0.04% of budget).

### Step 3: Launch the Web UI
Launches the interactive Gradio web application at `http://127.0.0.1:7860`:
```bash
python app/main.py
```

### Step 4: Run Comprehensive Evaluation Benchmark
Runs all 45 test questions from `data/questions.json` through the RAG pipeline and prints complete metrics:
```bash
python scripts/evaluate.py
```

---

## 📊 Evaluation & Benchmark Results

The evaluation benchmark covers **45 questions** across 5 distinct categories specified in the assignment:
- **12 Factual Questions**: Specific facts, definitions, and code syntax.
- **9 Cross-Reference Questions**: Spanning adjacent chunks, chapters, and summary vs. detail.
- **14 Retrieval Stress Tests**: Paraphrasing, exact terms, negations, multi-hop reasoning, temporal questions, and tables.
- **5 Out-of-Scope Questions**: Topics outside the textbook domain.
- **5 Ambiguous Questions**: Underspecified queries requiring clarification.
- **Exactly 5 Priority Questions**: Designated core use-case questions.

### Performance Highlights
| Metric | Result | Benchmark Goal |
| :--- | :---: | :---: |
| **Retrieval Recall@7 (Hit Rate)** | **94.3%** | High retrieval coverage |
| **Out-of-Scope Refusal Accuracy** | **100.0%** | 100% correct refusal |
| **Ambiguity Handling Rate** | **100.0%** | Proactive clarification |
| **Average Query Latency** | **~2,200 ms** | Fast user response |
| **Total Evaluation Cost** | **~$0.015** | Well under $5.00 limit |
| **Total Team Spend** | **<$0.02** | **<0.5% of $5.00 budget** |

---

## 💰 Budget & Cost Breakdown

| Component | Tokens Used | Cost (USD) | % of $5.00 Budget |
| :--- | :---: | :---: | :---: |
| **Corpus Indexing (Once to Disk)** | 103,761 | $0.00208 | 0.04% |
| **Single User Query (Average)** | ~1,600 | $0.00035 | 0.007% |
| **Full 45-Question Benchmark** | ~75,000 | ~$0.0160 | 0.32% |
| **Total Budget Remaining** | — | **>$4.98** | **>99.6%** |

---

## 👥 Team
- **Team Name**: Predictors
