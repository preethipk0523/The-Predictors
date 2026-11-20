"""
Chat Memory module for managing multi-turn conversation state and exporting chat history.
Keeps the last 3-5 conversation messages for coreference resolution and follow-up contextualization.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path
from app.config import CHAT_MEMORY_TURNS, CHAT_EXPORTS_DIR


class ChatMemory:
    """Manages short-term conversation memory and exports to Markdown."""

    def __init__(self, max_turns: int = CHAT_MEMORY_TURNS):
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []

    def add_user_message(self, content: str):
        """Append user message to history."""
        self.history.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, content: str, citations: Optional[List[int]] = None):
        """Append assistant response to history."""
        msg: Dict[str, Any] = {"role": "assistant", "content": content}
        if citations:
            msg["citations"] = citations
        self.history.append(msg)
        self._trim()

    def add_turn(self, user_msg: str, assistant_msg: str, citations: Optional[List[int]] = None):
        """Convenience method to record a complete turn."""
        self.add_user_message(user_msg)
        self.add_assistant_message(assistant_msg, citations=citations)

    def _trim(self):
        """Keep only the last max_turns turns (up to max_turns * 2 messages)."""
        max_messages = self.max_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def get_recent_messages(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Return the most recent messages."""
        n = (limit or self.max_turns) * 2
        return self.history[-n:] if len(self.history) > n else list(self.history)

    def format_context_prompt(self) -> str:
        """Format recent history into a concise conversation context string."""
        if not self.history:
            return ""
        lines = []
        for msg in self.history:
            speaker = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{speaker}: {msg['content']}")
        return "\n".join(lines)

    def clear(self):
        """Clear conversation memory."""
        self.history = []

    def export_to_markdown(self, session_title: str = "The Predictors Chat Session") -> Tuple[str, Path]:
        """
        Export conversation history, answers, and citations to a formatted Markdown file.
        Returns: (markdown_text, file_path)
        """
        CHAT_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        now_utc = datetime.now(timezone.utc)
        timestamp_str = now_utc.strftime("%Y-%m-%d_%H%M%S")
        filename = f"chat_export_{timestamp_str}.md"
        file_path = CHAT_EXPORTS_DIR / filename

        md_lines = [
            f"# {session_title}",
            f"*Exported on: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}*",
            f"*Corpus: The Crystal Ball Instruction Manual (Volume One) by Stephen Davies, Ph.D.*",
            "",
            "---",
            "",
        ]

        if not self.history:
            md_lines.append("*No conversation turns recorded in this session.*")
        else:
            turn_idx = 1
            i = 0
            while i < len(self.history):
                msg = self.history[i]
                if msg["role"] == "user":
                    md_lines.append(f"### Turn {turn_idx}")
                    md_lines.append(f"**👤 User:** {msg['content']}")
                    md_lines.append("")
                    # Check if next message is assistant response
                    if i + 1 < len(self.history) and self.history[i + 1]["role"] == "assistant":
                        asst_msg = self.history[i + 1]
                        md_lines.append(f"**🤖 Predictors Assistant:**")
                        md_lines.append(f"{asst_msg['content']}")
                        if asst_msg.get("citations"):
                            cites = ", ".join(f"[Page {p}]" for p in asst_msg["citations"])
                            md_lines.append("")
                            md_lines.append(f"*Verified Citations: {cites}*")
                        md_lines.append("")
                        md_lines.append("---")
                        md_lines.append("")
                        i += 2
                        turn_idx += 1
                        continue
                else:
                    md_lines.append(f"**🤖 Predictors Assistant:** {msg['content']}")
                    md_lines.append("")
                i += 1

        md_text = "\n".join(md_lines)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md_text)

        return md_text, file_path


# Global singleton instance for the session
chat_memory = ChatMemory()
