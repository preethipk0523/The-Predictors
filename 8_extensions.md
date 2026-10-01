# Project Extensions & Ideas

> Optional improvements to make your system better. None of these are required, focus on your core pipeline first. Pick what's relevant to your corpus and your team's strengths. The RAG extensions apply to your Tuesday project, and several of them (better retrieval especially) also improve the agent's `search_corpus` tool. The agentic extensions apply to your Wednesday/Thursday project.

---

## RAG extensions

### Retrieval improvements

- **Hybrid search (BM25 + vectors)**: combine keyword search with semantic search using Reciprocal Rank Fusion. Essential if your corpus has codes, article numbers, or acronyms. Library: `rank_bm25`.
- **Reranking**: retrieve top-20 with FAISS, then rerank to top-5 with a cross-encoder for more precise results. Library: `sentence-transformers`.
- **Confidence threshold**: log the top retrieval score per question. Set a threshold (e.g., 0.5) below which the system says "I'm not confident about this answer." Track false positives vs false negatives.
- **Query expansion**: use the LLM to rephrase the user's question into 2-3 variants, retrieve for each, merge results. Helps when the user's wording doesn't match the corpus.

### Chunking improvements

- **Sentence-boundary splitting**: instead of cutting at fixed character counts, split on sentence boundaries first, then group sentences until you hit the target size.
- **Semantic chunking**: use embeddings to detect topic shifts within a document and split there. More complex but produces more coherent chunks.
- **Chunk browser**: a simple script or Gradio mini-app to browse all chunks: search by keyword, filter by source/page, see char count. Helps spot bad chunking visually.

### Prompt engineering

- **Prompt versioning**: save each system prompt iteration to a file (`prompts/v1.txt`, `prompts/v2.txt`, ...) with a note on what changed and why. Prevents losing a good prompt after a bad edit.
- **Few-shot examples**: include 1-2 example question-answer pairs in the system prompt to show the model the expected format and style.
- **Dynamic prompts**: adjust the system prompt based on the question type (e.g., stricter instructions for yes/no questions, broader for open-ended ones).

### Evaluation & debugging

- **Cost tracker**: log every API call's cost to `data/cost_log.json`. Track spend over time and by operation (embedding vs chat).
- **Side-by-side comparison**: run the same question with different settings (chunk size 300 vs 800, k=3 vs k=5) and display results side by side.
- **LLM-as-judge**: automate evaluation by having the LLM score answers for faithfulness, relevancy, and completeness. See the code reference for a working example.
- **Demo Day dry run**: a script that runs 10 random questions, checks the app doesn't crash, verifies citations are present, and measures response time.

### UX improvements

- **Conversation memory**: maintain chat history so users can ask follow-up questions ("What about the second requirement?").
- **Source display**: show the retrieved chunks alongside the answer so users can see exactly what the system based its answer on.
- **Feedback buttons**: let users mark answers as helpful or not. Log this for later analysis.
- **Loading indicators**: show progress while the system retrieves and generates (embedding → searching → generating).
- **Export**: let users export the conversation or a specific answer as text/PDF.

### Advanced

- **Multi-language support**: if your corpus has documents in multiple languages, test whether the embedding model handles cross-language retrieval.
- **Document update pipeline**: a script that detects new/changed files in `data/sources/`, re-extracts, re-chunks, re-embeds only the changed documents, and updates the index.
- **Caching**: cache embeddings for repeated questions to save API calls and speed up responses.
- **Analytics dashboard**: a Gradio page showing usage stats: questions asked, retrieval scores, cost, most-queried topics.

---

## Agentic extensions

### More tools

- **New read-only tools**: give the agent more ways to see the corpus, for example a `get_document(source)` tool that returns a full document by name, or a `today()` tool so it can answer date-relative questions. Keep them read-only.
- **More acting tools (each with its own approval gate)**: beyond the required `send_message` tool, add another action such as saving an answer to a file or exporting a summary. Every irreversible action gets its own human approval step, and give the person enough context to decide (show what will be sent or written before they confirm).

### Planning and control

- **Plan-then-execute**: for complex multi-part questions, ask the model to write a numbered plan first, then run the loop against it. Compare answer quality and step count against the default decide-as-you-go loop.
- **Tune the harness**: experiment with the step cap, the number of tools, and the system prompt, and measure the effect on reliability and cost. Report what changed.

### Evaluation and observability

- **Trajectory scoring with LLM-as-judge**: extend the RAG judge to score the whole trajectory ("was this a sensible sequence of actions?"), not just the final answer, and give the judge an "unclear" option so it doesn't invent a verdict.
- **A trajectory viewer**: a small panel or script that renders each run as a readable step list (tool, arguments, result), so you can spot wrong-tool and redundant-step problems at a glance.

### Routing

- **Add a router**: classify each question first (simple lookup vs multi-step) and send simple ones straight to the RAG path, reserving the agent loop for the rest. Saves cost and latency on the common case.

### Industry standards

- **Go further with MCP**: your project already connects to an existing MCP server as a client (for `send_message`). To go deeper, flip to the other side and build your own MCP server that exposes your corpus tools (`search_corpus`, `quote_exact`) so any MCP client could use them. Or connect a second, public MCP server and give your agent one of its tools.
