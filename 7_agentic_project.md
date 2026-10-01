# Agentic AI Project

## What to build

Extend your RAG app into an agentic assistant: instead of always retrieving then answering, it decides for itself which tools to use, in what order, and how many times, to answer questions your fixed RAG pipeline can't.

Your agent must:
1. Wrap your Tuesday `retrieve()` function as a `search_corpus` tool the model can call when it decides to
2. Add a `calculator` tool (using `simpleeval`) for arithmetic the model shouldn't do in its head
3. Add a `quote_exact` tool that returns the exact text of a given page from a named document (verify or quote a passage a search pointed to)
4. Run these tools in an agent loop (think, act, observe, repeat) with a hard step cap
5. Decide on its own when to search, when to calculate, when to search again, and when to answer directly without any tool
6. Handle multi-step questions that need more than one action (for example, search for a figure, then calculate with it)
7. Recover when a tool returns nothing useful (rephrase and try again) instead of giving up or making something up
8. Serve the agent through your existing web UI: users ask a question and get an answer with citations, exactly like the RAG app, and the UI shows the tool calls the agent made (which tools, in what order) alongside the answer, so you can see how it reached the result
9. Log its trajectory (each tool call, arguments, and result) so you can inspect runs during the build
10. Add a `send_message` tool that posts the final answer to a Slack or Teams channel. Connect to that service's official **MCP server** as an **MCP client** (this is the primary path, and how tools are wired to agents in industry). If you can't get the MCP server's app/OAuth set up in time, a plain webhook or bot API call is an accepted fallback. Either way, this action can't be undone, so gate it behind human approval: the agent proposes the message, a person confirms (yes/no), and only then does it send. This is the human-in-the-loop pattern from the crash course, applied to the one tool in this project that acts on the outside world.
11. Score well on your evaluation questions (`questions.json` from Assignment 1), plus a few multi-step questions you add that force tool use

## Rules

- **From scratch:** no LangChain, no LlamaIndex, no agent frameworks. Raw Python only. This rule is about agent frameworks that hide the loop and tool calling; it does not forbid the MCP client SDK, which is a protocol library for connecting to tool servers, not an agent framework.
- **Models:** `gpt-4o-mini` for chat, `text-embedding-3-small` for embeddings. No other models.
- **Python 3.12** with **uv** for environment management
- **Budget:** fixed per team for the entire week (shared with the RAG project). An agent loop makes several model calls per question, so cap the steps, run at temperature 0, and track every call.
- **Allowed libraries:** everything allowed for the RAG project, plus `simpleeval` for the calculator tool, and for the `send_message` tool the `mcp` client SDK (primary path) or `requests` (webhook fallback). The RAG list is: `openai`, `faiss-cpu`, `pymupdf`, `pdfplumber`, `beautifulsoup4`, `numpy`, `tiktoken`, `gradio`, `python-dotenv`, `rank_bm25` (optional), `sentence-transformers` (optional).
- **Keep secrets out of git.** Any credentials for the `send_message` tool (MCP OAuth client secret, bot token, or webhook URL) go in `.env` (already gitignored), never in your code or committed files.
- **GenAI tools** (ChatGPT, Copilot) are allowed, but you must be able to explain every line of your code.

## Deliverables

**By the Agentic AI checkpoint:**
- A working agent behind your web UI that calls its tools on its own and shows the tool calls it made

**On your GitHub repo (by Demo Day):**
- Complete agent code in `app/`, extending your RAG application
- Updated `README.md` with what the agent does, how to install, and how to run

**On Demo Day:**
- Live demo of the agent answering questions from your corpus, including at least one multi-step question that needs more than one tool
- Show the `send_message` tool posting an answer to your channel, including the approval step (the agent proposes, you confirm, then it sends)
- Be ready for questions from the jury, including tricky or out-of-scope ones

## Tips

- **Start with the code reference** (`../crash_course/agentic/code_reference.md`), don't start from a blank page
- **Get the loop working first, then add tools.** A bare loop with one tool that works beats four tools wired into a loop that doesn't. Get `search_corpus` running end to end, then add `calculator` and `quote_exact`, and add `send_message` (with its approval step) last, since it is the most involved.
- **Read your trajectories.** Log every tool call, its arguments, and its result, then read them. Most agent bugs (wrong tool, repeated steps, no recovery) are obvious in the trace and invisible in the final answer.
- **Test with your eval set.** Run your `questions.json` through the agent, and add a few multi-step questions that force a search plus a calculation.
- **Watch the step count and cost.** If the agent regularly hits the step cap, fix your tool descriptions and system prompt rather than raising the cap. Budget is shared with the RAG project.

## Resources

- [Crash course notes](../crash_course/agentic/notes.md), concepts and theory
- [Code reference](../crash_course/agentic/code_reference.md), all runnable code
- [Evaluation toolkit](6_evaluation_toolkit.md), debugging and testing tools for the build phase
- [Extensions & ideas](8_extensions.md), optional improvements to go further
- [Gradio docs](https://www.gradio.app/docs), for your web UI
- [Build an MCP client](https://modelcontextprotocol.io/docs/develop/build-client), the official guide for the `send_message` tool's primary path
- [Slack MCP server](https://docs.slack.dev/ai/mcp-server), the send-message tool, connection endpoint, and auth if you use Slack
