# RAG Project

## What to build

Build a RAG (Retrieval-Augmented Generation) application that answers questions about your team's corpus, with source citations and a web interface.

Your app must:
1. Load your `corpus.json` and chunk the text
2. Embed and index all chunks using FAISS (do this once, save to disk)
3. Accept a user question via a web UI
4. Retrieve relevant chunks and generate an answer with citations
5. Display the source (document + page) alongside the answer
6. Handle out-of-scope questions gracefully ("I don't have information about this")
7. Score well on your evaluation questions (`questions.json`), aim for high accuracy on factual questions and correct refusal on out-of-scope ones

## Rules

- **From scratch**: no LangChain, no LlamaIndex, no RAG frameworks. Raw Python only.
- **Models:** `gpt-4o-mini` for chat, `text-embedding-3-small` for embeddings. No other models.
- **Python 3.12** with **uv** for environment management
- **Budget:** $5 per team for the entire week (shared across RAG and Agentic AI projects). Embed once, save to disk, don't waste tokens.
- **Allowed libraries:** `openai`, `faiss-cpu`, `pymupdf`, `pdfplumber`, `beautifulsoup4`, `numpy`, `tiktoken`, `gradio`, `python-dotenv`, `rank_bm25` (optional), `sentence-transformers` (optional)

- **GenAI tools** (ChatGPT, Copilot) are allowed, but you must be able to explain every line of your code.

## Deliverables

**By the RAG checkpoint:**
- A working RAG chatbot with web UI, source citations, and out-of-scope handling

**On your GitHub repo (by Demo Day):**
- Complete RAG application code in `app/`
- Updated `README.md` with project description, how to install, and how to run

**On Demo Day:**
- 15-minute slot per team: live demo of both projects + Q&A
- Show your RAG app answering questions from your corpus
- Be ready for questions from the jury (including tricky/out-of-scope ones)

## Tips

- **Start with the code reference** (`../crash_course/rag/code_reference.md`), don't start from a blank page
- **Get it working first, then improve**: a basic pipeline that works beats a fancy one that doesn't. You need a working chatbot by the RAG checkpoint.
- **One person embeds, everyone shares**: only one teammate should embed the corpus and build the FAISS index. Share `my_index.faiss` and `chunks.json` via your Teams channel. Everyone else loads from disk.
- **Test with your eval set**: run your `questions.json` through the system after every change
- **Track your costs**: use the cost tracker from the code reference. Budget is shared with the Agentic AI project.

## Resources

- [Crash course notes](../crash_course/rag/notes.md), concepts and theory
- [Code reference](../crash_course/rag/code_reference.md), all runnable code
- [Evaluation toolkit](6_evaluation_toolkit.md), debugging and testing tools for the build phase
- [Extensions & ideas](8_extensions.md), optional improvements to go further
- [Gradio docs](https://www.gradio.app/docs), for building your web UI
