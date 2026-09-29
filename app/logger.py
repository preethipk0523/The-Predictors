"""Trajectory logging + cost tracking (Person A)."""
import json
import os
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = Path(os.getenv("AGENT_LOG_DIR", "logs"))
USAGE_FILE = LOG_DIR / "usage.json"
UI_RESULT_MAX_CHARS = 500

# gpt-4o-mini prices in USD per 1M tokens. CHECK the OpenAI pricing page and update if needed.
PRICE_INPUT_PER_M = 0.15
PRICE_OUTPUT_PER_M = 0.60


def truncate(text: str, limit: int = UI_RESULT_MAX_CHARS) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f"... [truncated, {len(text)} chars]"


class RunLogger:
    """One instance per agent run."""

    def __init__(self, question: str):
        self.question = question
        self.started = time.time()
        self.trajectory = []          # full results (written to file)
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.model_calls = 0

    # ---- tool calls ----
    def log_call(self, step: int, tool: str, args, result: str):
        self.trajectory.append({
            "step": step, "tool": tool, "args": args,
            "result": str(result), "timestamp": time.time(),
        })

    def ui_trajectory(self):
        """Trajectory for the UI: same data but results cut to ~500 chars."""
        return [
            {"step": t["step"], "tool": t["tool"], "args": t["args"],
             "result": truncate(t["result"])}
            for t in self.trajectory
        ]

    # ---- cost ----
    def add_usage(self, usage):
        self.model_calls += 1
        if usage is not None:
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens

    @property
    def cost(self) -> float:
        return (self.prompt_tokens * PRICE_INPUT_PER_M
                + self.completion_tokens * PRICE_OUTPUT_PER_M) / 1_000_000

    def cost_line(self) -> str:
        total = self.prompt_tokens + self.completion_tokens
        return f"Run cost: ${self.cost:.4f} ({total:,} tokens, {self.model_calls} calls)"

    # ---- persistence ----
    def finish(self, answer: str, stopped_reason: str, steps: int) -> Path:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = LOG_DIR / f"run_{int(self.started * 1000)}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"type": "question", "question": self.question,
                                "timestamp": self.started}) + "\n")
            for t in self.trajectory:
                f.write(json.dumps({"type": "tool_call", **t}, default=str) + "\n")
            f.write(json.dumps({
                "type": "final", "answer": answer, "stopped_reason": stopped_reason,
                "steps": steps, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cost_usd": round(self.cost, 6), "timestamp": time.time(),
            }) + "\n")
        self._update_usage()
        print(self.cost_line())
        return path

    def _update_usage(self):
        try:
            data = json.loads(USAGE_FILE.read_text()) if USAGE_FILE.exists() else {}
        except Exception:
            data = {}
        data.setdefault("runs", 0)
        data.setdefault("prompt_tokens", 0)
        data.setdefault("completion_tokens", 0)
        data.setdefault("cost_usd", 0.0)
        data["runs"] += 1
        data["prompt_tokens"] += self.prompt_tokens
        data["completion_tokens"] += self.completion_tokens
        data["cost_usd"] = round(data["cost_usd"] + self.cost, 6)
        USAGE_FILE.write_text(json.dumps(data, indent=2))
