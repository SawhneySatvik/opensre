"""In-memory session storage backend.

A :class:`~interactive_shell.harness.llm_context.session.types.SessionStorage` implementation that
keeps records in process memory instead of on disk. Useful for tests and any
caller that wants session writes without touching the filesystem. Mirrors the
observable semantics of :class:`JsonlSessionStorage` (open/append/flush/reopen,
empty-session deletion, idempotent flush) without JSON serialization.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from interactive_shell.harness.llm_context.session.types import CHAT_KINDS, SessionPersistenceSource

_TRIGGER_MAX_CHARS = 200


class InMemorySessionStorage:
    """SessionStorage backend that stores records in a per-session dict."""

    def __init__(self) -> None:
        self._files: dict[str, list[dict[str, Any]]] = {}

    def read(self, session_id: str) -> list[dict[str, Any]]:
        """Return a copy of the records written for ``session_id`` (test helper)."""
        return list(self._files.get(session_id, []))

    def open_session(self, session: SessionPersistenceSource) -> None:
        self._files[session.session_id] = [
            {
                "type": "session_start",
                "session_id": session.session_id,
                "started_at": datetime.fromtimestamp(session.started_at, tz=UTC).isoformat(),
            }
        ]

    def _is_finalized(self, session_id: str) -> bool:
        records = self._files.get(session_id)
        if not records:
            return False
        return records[-1].get("type") == "session_end"

    def _ensure_session_open(self, session_id: str) -> None:
        if self._is_finalized(session_id):
            self.reopen_session(session_id)

    def append_turn(self, session: SessionPersistenceSource, kind: str, text: str) -> None:
        records = self._files.get(session.session_id)
        if records is None:
            return
        self._ensure_session_open(session.session_id)
        records.append({"type": "turn", "kind": kind, "text": text})

    def append_turn_detail(
        self,
        session_id: str,
        kind: str,
        prompt: str,
        *,
        response: str | None = None,
        turn_id: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        latency_ms: int | None = None,
    ) -> None:
        records = self._files.get(session_id)
        if records is None:
            return
        record: dict[str, Any] = {"type": "turn_detail", "kind": kind, "prompt": prompt}
        if response is not None:
            record["response"] = response
        if turn_id is not None:
            record["turn_id"] = turn_id
        if model is not None:
            record["model"] = model
        if provider is not None:
            record["provider"] = provider
        if latency_ms is not None:
            record["latency_ms"] = latency_ms
        records.append(record)

    def append_tool_call(
        self,
        session_id: str,
        *,
        tool: str,
        arguments: dict[str, Any],
        result: str,
        ok: bool,
        source: str | None = None,
    ) -> None:
        records = self._files.get(session_id)
        if records is None:
            return
        self._ensure_session_open(session_id)
        record: dict[str, Any] = {
            "type": "tool_call",
            "ts": datetime.now(UTC).isoformat(),
            "tool": tool,
            "arguments": arguments,
            "ok": ok,
            "result": result,
        }
        if source is not None:
            record["source"] = source
        records.append(record)

    def append_investigation_result(
        self,
        session_id: str,
        state: dict[str, Any],
        *,
        trigger: str = "",
    ) -> str:
        investigation_id = uuid.uuid4().hex[:8]
        records = self._files.get(session_id)
        if records is None:
            return investigation_id
        self._ensure_session_open(session_id)
        report = state.get("problem_md") or state.get("slack_message") or state.get("report") or ""
        records.append(
            {
                "type": "investigation_result",
                "investigation_id": investigation_id,
                "completed_at": datetime.now(UTC).isoformat(),
                "trigger": trigger.strip()[:_TRIGGER_MAX_CHARS],
                "root_cause": str(state.get("root_cause") or ""),
                "report": str(report),
                "root_cause_category": str(state.get("root_cause_category") or ""),
                "alert_name": str(state.get("alert_name") or ""),
                "run_id": str(state.get("run_id") or ""),
            }
        )
        return investigation_id

    def flush(self, session: SessionPersistenceSource) -> None:
        records = self._files.get(session.session_id)
        if records is None:
            return
        if records and records[-1].get("type") == "session_end":
            return

        total_turns = sum(1 for r in records if r.get("type") == "turn")
        detail_turns = sum(1 for r in records if r.get("type") == "turn_detail")
        if total_turns == 0 and detail_turns == 0:
            del self._files[session.session_id]
            return

        chat_turns = sum(
            1 for r in records if r.get("type") == "turn" and r.get("kind") in CHAT_KINDS
        )
        investigation_turns = sum(
            1
            for r in records
            if r.get("type") == "turn" and r.get("kind") in ("alert", "incoming_alert")
        )
        now = datetime.now(UTC)
        started_at = datetime.fromtimestamp(session.started_at, tz=UTC)
        duration_secs = max(0, int((now - started_at).total_seconds()))

        if session.cli_agent_messages or session.accumulated_context:
            snapshot: dict[str, Any] = {"type": "conversation_snapshot"}
            if session.cli_agent_messages:
                snapshot["cli_agent_messages"] = [list(m) for m in session.cli_agent_messages]
            if session.accumulated_context:
                snapshot["accumulated_context"] = dict(session.accumulated_context)
            records.append(snapshot)

        records.append(
            {
                "type": "session_end",
                "ended_at": now.isoformat(),
                "duration_secs": duration_secs,
                "total_turns": total_turns,
                "chat_turns": chat_turns,
                "investigation_turns": investigation_turns,
            }
        )

    def reopen_session(self, session_id: str) -> None:
        records = self._files.get(session_id)
        if not records:
            return
        if records[-1].get("type") == "session_end":
            records.pop()
        if records and records[-1].get("type") == "conversation_snapshot":
            records.pop()
