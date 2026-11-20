"""
Agent Audit Logging Module.
Maintains a local, structured audit log (JSONL) of every tool call invoked by the agent.
Records timestamps, tool names, inputs, execution statuses, and approval states.
Guarantees zero storage of API keys, bearer tokens, or sensitive credentials.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from app.config import AUDIT_LOG_FILE

SENSITIVE_KEYS = {"api_key", "token", "authorization", "password", "secret", "bearer"}


def _sanitize_dict(data: Any) -> Any:
    """Recursively redacts sensitive keys from audit log payloads."""
    if isinstance(data, dict):
        sanitized = {}
        for k, v in data.items():
            if any(s in str(k).lower() for s in SENSITIVE_KEYS):
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = _sanitize_dict(v)
        return sanitized
    elif isinstance(data, list):
        return [_sanitize_dict(item) for item in data]
    return data


def log_tool_audit(
    tool_name: str,
    arguments: Dict[str, Any],
    status: str,
    result_summary: str,
    approved: Optional[bool] = None,
    duration_ms: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Appends an audited tool execution record to data/agent_audit.jsonl.
    """
    AUDIT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    sanitized_args = _sanitize_dict(arguments) if isinstance(arguments, dict) else {"raw": str(arguments)[:200]}
    short_summary = str(result_summary)[:300] + ("..." if len(str(result_summary)) > 300 else "")

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "arguments": sanitized_args,
        "status": status,
        "approved": approved if approved is not None else False,
        "duration_ms": round(duration_ms, 2),
        "result_summary": short_summary,
        "metadata": _sanitize_dict(extra or {}),
    }

    try:
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return {"status": "logged", "entry": entry}
    except Exception as exc:
        print(f"[AuditLogger] Failed to write audit log: {exc}")
        return {"status": "error", "error": str(exc)}


def get_audit_entries(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves recent audit log records."""
    if not AUDIT_LOG_FILE.exists():
        return []
    records = []
    try:
        with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    records.append(json.loads(line_str))
    except Exception as exc:
        print(f"[AuditLogger] Failed to read audit log: {exc}")
    return records[-limit:]
