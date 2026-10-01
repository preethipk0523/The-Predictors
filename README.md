# Predictors Agentic Assistant & RAG System: The Crystal Ball Manual

An autonomous, production-grade Agentic Assistant and Retrieval-Augmented Generation (RAG) system built **from scratch** (no LangChain, no LlamaIndex, raw Python only) to provide grounded, citation-backed answers, multi-step tool reasoning, and MCP-connected actions over ***The Crystal Ball Instruction Manual (Volume One: Introduction to Data Science, v1.2)*** by Stephen Davies, Ph.D.

Developed by **Team Predictors** for the AI Hackathon.

---

## 🌟 Key Architecture & Capabilities

### 1. Autonomous Agent Loop (Assignment 7 & 8)
- **Think-Act-Observe Loop**: Runs an autonomous loop using `gpt-4o-mini` (temperature: 0.0) with a hard step cap (`MAX_AGENT_STEPS = 6`).
- **Autonomous Tool Calling**: Decides on its own when to search, when to calculate, when to verify page quotes, when to inspect chapter structure, and when to answer directly.
- **Self-Recovery**: Automatically reformulates queries or inspects adjacent context if a tool returns insufficient information.
- **Full Trajectory Logging**: Records every tool call, arguments, and observations to `data/agent_trajectories.jsonl` for auditability and grading inspection.

### 2. Specialized Tool Suite (`tools/`)
1. **`search_corpus(query, top_k)`**:
   - Wraps the Tuesday hybrid retriever combining dense FAISS semantic similarity with BM25 sparse keyword matching.
2. **`calculator(expression)`**:
   - Safe arithmetic evaluation engine built with `simpleeval`. Exposes standard math functions (`sqrt`, `abs`, `round`, `log`, `pow`) so the agent never performs mental arithmetic.
3. **`quote_exact(page, document, passage)`**:
   - Retrieves the full, exact text of a specified page from `data/corpus.json` to verify quotes and inspect cross-section details.
4. **`list_chapters()`**:
   - Provides the complete table of contents with chapter starting pages and page spans across all 29 textbook chapters.
5. **`send_message(channel, message, approved)`**:
   - Posts messages to Slack or Teams.
   - **Primary Path**: Model Context Protocol (**MCP**) client SDK connection.
   - **Fallback Path**: Standard incoming webhook (`SLACK_WEBHOOK_URL` / `TEAMS_WEBHOOK_URL`).
   - **Human-in-the-Loop Approval Gate**: Irreversible actions cannot execute without explicit human approval (`approved=True`). Proposes the message first, prompts the user in the UI, and only dispatches upon confirmation.

### 3. Evaluation & Debugging Toolkit (Assignment 6)
- **Retrieval Checker** ([`scripts/retrieval_checker.py`](file:///d:/The%20Predictors/scripts/retrieval_checker.py)): Inspects top-k retrieved chunks, scores, and verifies against ground truth targets (`questions.json`).
- **Failure Log** ([`scripts/FAILURE_LOG.md`](file:///d:/The%20Predictors/scripts/FAILURE_LOG.md)): Documents the 6 baseline failure cases (Q6, Q15, Q23, Q29, Q30, Q33), root cause analyses, and fixes applied.
- **Evaluation Runner** ([`scripts/eval_runner.py`](file:///d:/The%20Predictors/scripts/eval_runner.py)): Runs benchmarks across 50 questions across 6 categories (factual, cross-reference, stress-test, out-of-scope, ambiguous, multi-step) in either Direct RAG or Autonomous Agent mode.
- **Benchmark Scorecard** ([`data/eval_results.json`](file:///d:/The%20Predictors/data/eval_results.json)): Saves detailed run scorecards with category and difficulty breakdowns.

### 4. Interactive Web Application ([`app/main.py`](file:///d:/The%20Predictors/app/main.py))
- Modern dark-glassmorphic Gradio interface.
- **Dual Engine Toggle**: Switch seamlessly between **🤖 Autonomous Agent** and **⚡ Direct Hybrid RAG**.
- **Live Trajectory Visualizer**: Step-by-step cards showing each tool call, inputs, and observation results.
- **Human Approval Modal**: Interactive confirm/reject buttons when external message sending is proposed.
- **Live Latency & Budget Tracker**: Real-time USD spend tracking against the $5.00 budget limit.

---

## 📁 Repository Structure

```
predictors/
├── README.md                          # Comprehensive project documentation
├── .gitignore                         # Excludes virtual environments and credentials
├── .env                               # Secrets (OPENAI_API_KEY, MCP & Webhook URLs)
├── 6_evaluation_toolkit.md           # Assignment 2 instructions
├── 7_agentic_project.md               # Assignment 3 agentic instructions
├── 8_extensions.md                    # Project extension specifications
├── crystal1 (1).pdf                   # Original textbook PDF
│
├── app/                               # Core Application Engine
│   ├── __init__.py
│   ├── config.py                      # Models, paths, hyperparams, and pricing
│   ├── chunking.py                    # Semantic paragraph chunking
│   ├── cost_tracker.py                # Budget & token logging module
│   ├── indexer.py                     # Offline FAISS + BM25 indexing engine
│   ├── retriever.py                   # Hybrid dense + sparse retriever
│   ├── generator.py                   # Grounded response generator (gpt-4o-mini)
│   ├── pipeline.py                    # End-to-end fixed RAG orchestrator
│   ├── agent.py                       # Autonomous agent loop & trajectory tracker
│   └── main.py                        # Gradio Web Application UI
│
├── tools/                             # Agent Tools Suite
│   ├── __init__.py
│   ├── registry.py                    # Tool dispatcher & OpenAI function schemas
│   ├── corpus.py                      # search_corpus tool (wraps HybridRetriever)
│   ├── calculator.py                  # calculator tool (simpleeval safe arithmetic)
│   ├── quote.py                       # quote_exact tool (page text inspection)
│   ├── outline.py                     # list_chapters tool (table of contents)
│   └── messaging.py                   # send_message tool (MCP client + approval)
│
├── scripts/                           # Evaluation & Tooling Scripts
│   ├── retrieval_checker.py           # Single-query retrieval inspector (Required)
│   ├── FAILURE_LOG.md                 # Debugging history & failure diagnoses (Required)
│   ├── eval_runner.py                 # Benchmark runner (RAG & Agent modes)
│   ├── evaluate.py                    # RAG benchmark evaluator
│   └── extract.py                     # PyMuPDF document parser
│
└── data/                              # Data artifacts & persistent logs
    ├── corpus.json                    # Extracted clean text with page metadata
    ├── chunks.json                    # Chunked text with token counts & metadata
    ├── faiss_index.bin                # Persisted FAISS vector index
    ├── bm25.pkl                       # Persisted BM25 keyword index
    ├── questions.json                 # 50 benchmark questions (including multi-step)
    ├── eval_results.json              # Latest evaluation scorecard
    ├── agent_trajectories.jsonl       # Audit log of autonomous agent trajectories
    ├── sent_messages.log              # Audit log of human-approved sent messages
    └── cost_tracker.json              # Real-time token & USD spend tracker
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

# Install required packages
uv pip install openai faiss-cpu pymupdf pdfplumber beautifulsoup4 numpy tiktoken gradio python-dotenv rank_bm25 simpleeval mcp requests
```

### 2. Configure Environment Variables
Create or verify `.env` in the root directory:
```bash
# Required for embeddings and chat
OPENAI_API_KEY=sk-proj-...

# Optional: Slack / Teams MCP Server or Webhook for send_message tool
SLACK_MCP_ENDPOINT=
SLACK_BOT_TOKEN=
SLACK_WEBHOOK_URL=
TEAMS_WEBHOOK_URL=
```
*(Note: If no external MCP server or webhook is configured, `send_message` logs approved messages to `data/sent_messages.log` in local audit mode, ensuring seamless demonstrations).*

---

## 🛠️ How to Run

### Step 1: Launch the Interactive Web UI
Starts the Gradio application at `http://127.0.0.1:7860`:
```bash
python app/main.py
```
*Key interactions in the UI:*
- **Switch Engine Modes**: Toggle between **Autonomous Agent** and **Direct Hybrid RAG**.
- **Run Multi-Step Questions**: Watch the agent execute multiple tools (`search_corpus` → `calculator` or `quote_exact`).
- **Human-in-the-Loop Messaging**: Ask the agent to post a summary to `#general`; confirm the proposed text using the **Confirm & Send** button.
- **Inspect Trajectory**: Expand the execution trace cards to see tool inputs and observation outputs.

### Step 2: Run the Retrieval Checker (Assignment 6 Deliverable)
Inspect retrieval for a specific question ID from `questions.json` or any arbitrary query:
```bash
# Inspect Question #1 (Factual priority question)
python scripts/retrieval_checker.py --id 1

# Inspect Question #6 (Simpson's Paradox)
python scripts/retrieval_checker.py --id 6

# Inspect any custom query
python scripts/retrieval_checker.py "What is Stevens' hierarchy of measurement?"
```

### Step 3: Run Evaluation Benchmarks
Run all benchmark questions or category subsets:
```bash
# Run RAG mode benchmark on all questions
python scripts/eval_runner.py --mode rag

# Run Agent 9-scenario benchmark vs Fixed RAG
python scripts/agent_eval_runner.py

# Run all 34 automated unit and safety tests
python -m unittest discover tests
```

---

## 🛠️ Specialized Tool Suite & Justifications (Phase 4)

The agent integrates 5 read-only & action tools strictly justified by the requirements of navigating *The Crystal Ball Instruction Manual*:

| Tool | Type | Purpose | Justification for The Crystal Ball Corpus |
| :--- | :---: | :--- | :--- |
| **`search_corpus(query, top_k)`** | Read-Only | Hybrid dense (FAISS) + sparse (BM25) search | Retrieves conceptual passages, definitions, and code syntax across all 29 chapters. Distinguishes hits from empty results. |
| **`calculator(expression)`** | Read-Only | Safe arithmetic using `simpleeval` | Eliminates mental math errors when calculating page spans, index differences, sample frequencies, or metrics. |
| **`quote_exact(page, document, passage)`** | Read-Only | Page-level text retrieval and substring verification | Verifies exact phrasing on target pages (e.g. Stevens' scales on Page 50) and inspects surrounding context. |
| **`list_chapters()`** | Read-Only | Structural table of contents mapping | Resolves chapter-to-page mappings across all 29 chapters without wasting retrieval steps. |
| **`send_message(channel, message, approved)`** | Action | External dispatch to Slack/Teams via MCP / Webhook | Enables sharing summaries with colleagues. Strictly gated behind human approval (`approved=True`) to prevent unauthorized posts. |

---

## 📊 Benchmark Scorecard: Agent vs Fixed RAG (Phase 5)

| Metric | Autonomous Agent | Fixed RAG Pipeline | Target / Specification |
| :--- | :---: | :---: | :---: |
| **Overall Benchmark Accuracy (10 Scenarios)** | **100.0%** | **70.0%** | >80.0% |
| **Multi-Step & Calculation (e.g. Page Diff)** | **100.0% (Passed)** | 100.0% | Multi-tool coordination |
| **Lexical Recovery / Rephrasing** | **100.0% (Passed)** | ❌ Failed (Missed) | Self-recovery without hallucination |
| **Conversational Greetings** | **100.0% (Direct Answer)** | 100.0% (Redundant RAG) | 0 external tool calls |
| **Out-of-Scope Refusal** | **100.0% Refused** | 100.0% Refused | Clean refusal without loops |
| **Human-in-the-Loop Messaging Gate** | **100.0% Gated** | ❌ Unsupported | Zero unauthorized dispatches |
| **Prompt Injection Resistance** | **100.0% Resisted** | ❌ Exploitable | Document text isolated as data |
| **Average Steps per Run** | **2.30 steps** | 1.00 step | Hard capped at 6 steps |
| **Repeated Call Detection & Loop Break** | **100.0% Enforced** | N/A | Prevents infinite loops |
| **Automated Test Suite** | **34 / 34 Passed (100%)** | 22 / 22 Passed | Unit + Safety + Frontend |

---

## 👥 Team Work Division (3-Member Architecture Plan)

To ensure modularity and clean parallel engineering, the team divides ownership as follows:

### Member 1 — Agent Core & Observability
- **Responsibilities**:
  - Autonomous think-act-observe loop with OpenAI function calling (`app/agent.py`).
  - Hard step limit enforcement (`MAX_AGENT_STEPS = 6`) and stop reason classification (`direct_answer`, `completed_task`, `max_step_limit_reached`, `repeated_calls_halted`).
  - Repeated tool call detection and loop guard.
  - Search recovery coordinator (caps rephrase attempts and catches empty results).
  - Trajectory event typing (`step_latency_ms`, `events`, `status`) and JSONL logging (`data/agent_trajectories.jsonl`).
  - Core agent unit tests (`tests/test_agent.py`).
- **Files Owned**: `app/agent.py`, `tests/test_agent.py`.

### Member 2 — Tools Suite & Safety Engineering
- **Responsibilities**:
  - Implementation, input validation, and schema definitions for all 5 tools (`tools/`).
  - Differentiating "no matches" in `tools/corpus.py`.
  - Safe arithmetic parsing and division-by-zero protection in `tools/calculator.py`.
  - Bounds validation in `tools/quote.py` and TOC structuring in `tools/outline.py`.
  - Human-in-the-loop gate for `send_message` and outcome verification (`tools/messaging.py`).
  - Mock declining harness and prompt-injection safety tests (`tests/test_safety.py`).
- **Files Owned**: `tools/registry.py`, `tools/corpus.py`, `tools/calculator.py`, `tools/quote.py`, `tools/outline.py`, `tools/messaging.py`, `tests/test_safety.py`.

### Member 3 — Integration, Evaluation & Web UI
- **Responsibilities**:
  - 9-scenario evaluation runner and comparative benchmark against Fixed RAG (`scripts/agent_eval_runner.py`).
  - Failure analysis log and root cause categorization (`scripts/AGENT_FAILURE_LOG.md`).
  - Gradio UI integration: inline concise tool sequence badge (`⚡ Action Sequence`) + collapsible trace (`<details>`) beside chat answers (`app/main.py`).
  - Live trajectory visualizer panel and Human-in-the-Loop confirm/cancel modal (`app/main.py`, `tests/test_frontend.py`).
  - Documentation and Demo sequence coordination (`README.md`).
- **Files Owned**: `scripts/agent_eval_runner.py`, `app/main.py`, `scripts/AGENT_FAILURE_LOG.md`, `README.md`.

---

## 👥 Team
- **Team Name**: Predictors
- **Members**: Team Predictors

