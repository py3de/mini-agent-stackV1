"""Governance layer: who may do what, who approved it, and what happened.

Four controls, applied to every tool call the agent wants to make:

1. Permission policy   every tool is classified as `read` or `write`. Unknown tools are treated
                       as `write` (deny by default), so a new tool can never silently change data.
2. Governance mode     approve    -> reads run freely, every write needs a human decision (default)
                       read_only  -> writes are always blocked (safe investigation mode)
                       auto       -> writes run without approval (tests, evals, trusted batch jobs)
3. Approval gate       in `approve` mode the agent pauses before a write (LangGraph interrupt) and
                       resumes after a human approves or rejects each pending action.
4. Audit trail         every run, tool call, policy decision, approval and escalation is appended
                       to a JSONL log with a run ID and timestamp.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

Permission = Literal["read", "write"]
Mode = Literal["approve", "read_only", "auto"]
MODES: tuple[str, ...] = ("approve", "read_only", "auto")

TOOL_PERMISSIONS: dict[str, Permission] = {
    "search_contacts": "read",
    "search_deals": "read",
    "search_knowledge_base": "read",
    "create_note": "write",
    "update_deal_stage": "write",
}


def permission_for(tool_name: str) -> Permission:
    """Deny by default: a tool that isn't classified is treated as a write."""
    return TOOL_PERMISSIONS.get(tool_name, "write")


def current_mode(mode: str | None = None) -> Mode:
    mode = (mode or os.getenv("GOVERNANCE_MODE", "approve")).lower()
    if mode not in MODES:
        raise ValueError(f"Unknown governance mode '{mode}'. Choose one of: {', '.join(MODES)}")
    return mode  # type: ignore[return-value]


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


class AuditLog:
    """Append-only JSONL audit trail. One line per event, safe to read with any log tool."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or os.getenv("AUDIT_LOG_PATH", "audit/audit.jsonl"))
        self._lock = threading.Lock()

    def record(self, event: str, run_id: str, **data: Any) -> dict[str, Any]:
        entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "run_id": run_id, "event": event, **data}
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return entry

    def events(self, run_id: str | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [r for r in rows if run_id is None or r["run_id"] == run_id]


_audit: AuditLog | None = None


def get_audit() -> AuditLog:
    global _audit
    if _audit is None:
        _audit = AuditLog()
    return _audit


def set_audit(log: AuditLog | None) -> None:
    """Swap the audit log (used by tests)."""
    global _audit
    _audit = log
