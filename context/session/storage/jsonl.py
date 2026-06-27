"""JSONL-backed per-session storage.

Design: incremental writes, one JSONL file per session under
``~/.opensre/sessions/``.

- open_session()       — writes session_start immediately when the REPL starts
- append_turn()        — appends a turn stub (kind + text) for stats counting
- append_turn_detail() — appends a full turn record (prompt + response) for /resume
- append_tool_call()   — appends one integration/API tool-call result
- append_investigation_result() — appends a completed RCA record for /rca history
- flush()              — writes conversation_snapshot + session_end on exit or /new;
                         deletes the file if no turns were recorded (empty session)
- reopen_session()     — strips trailing session_end so /resume can append to the file

Cross-session queries (list, load, RCA history) live in
:class:`~context.session.repo.JsonlSessionRepo`.
"""

from __future__ import annotations

import contextlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config.version import get_version
from context.session.paths import session_path
from context.session.types import CHAT_KINDS, SessionPersistenceSource

_TRIGGER_MAX_CHARS = 200


class JsonlSessionStorage:
    """Per-session JSONL writer.

    Stateless: every method resolves the session file from the session id on
    each call, so a single instance is safe to share across the whole REPL.
    All I/O errors are suppressed so the REPL never crashes on a bad path.
    """

    def open_session(self, session: SessionPersistenceSource) -> None:
        """Write session_start record, creating the session file on disk.

        Called once at REPL start and again after every /new (which rotates
        the session_id).
        """
        with contextlib.suppress(Exception):
            path = session_path(session.session_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "type": "session_start",
                "session_id": session.session_id,
                "started_at": datetime.fromtimestamp(session.started_at, tz=UTC).isoformat(),
                "opensre_version": get_version(),
            }
            with path.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _session_is_finalized(path: Path) -> bool:
        if not path.exists():
            return False
        with contextlib.suppress(Exception):
            lines = path.read_text(encoding="utf-8").splitlines()
            if not lines:
                return False
            with contextlib.suppress(json.JSONDecodeError):
                record = json.loads(lines[-1])
                return isinstance(record, dict) and record.get("type") == "session_end"
        return False

    def _ensure_session_open(self, session_id: str) -> None:
        """Reopen a finalized session file so append paths can continue writing."""
        path = session_path(session_id)
        if self._session_is_finalized(path):
            self.reopen_session(session_id)

    def append_turn(self, session: SessionPersistenceSource, kind: str, text: str) -> None:
        """Append a turn stub to the session file for stats counting.

        Called by ReplSession.record() on every interaction. Stubs carry kind
        and the full input text (no truncation). No-ops silently if the file
        does not exist (e.g. the non-interactive initial_input path).
        """
        with contextlib.suppress(Exception):
            path = session_path(session.session_id)
            if not path.exists():
                return
            self._ensure_session_open(session.session_id)
            record = {
                "type": "turn",
                "kind": kind,
                "text": text,
            }
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

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
        """Append a full turn record (prompt + response) for /resume reconstruction.

        Called by PromptRecorder.flush() after each LLM turn completes.
        These records make session files self-contained: /resume can rebuild
        cli_agent_messages from turn_detail records when no conversation_snapshot
        is present (e.g. old files or crash before flush).
        No-ops silently if the session file does not exist.
        """
        with contextlib.suppress(Exception):
            path = session_path(session_id)
            if not path.exists():
                return
            record: dict[str, Any] = {
                "type": "turn_detail",
                "kind": kind,
                "prompt": prompt,
            }
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
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

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
        """Append one integration/API tool-call result to the session file.

        Written by the conversational data-gathering loop after each tool runs,
        so a session file carries the actual evidence each turn fetched (tool
        name, arguments, and a bounded result snippet) rather than only the
        final prose answer. Callers MUST pass already-redacted, already-truncated
        values: this writer stays a dumb sink and pulls in no agent/tool imports.
        No-ops silently if the session file does not exist.
        """
        with contextlib.suppress(Exception):
            path = session_path(session_id)
            if not path.exists():
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
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def flush(self, session: SessionPersistenceSource) -> None:
        """Write conversation_snapshot + session_end and close the session file.

        Idempotent: no-ops if session_end is already the last line, so
        double-calling (e.g. /new flow + entrypoint finally) is safe.
        If no turns were recorded the file is deleted instead.
        Writes a conversation_snapshot record before session_end so /resume
        can restore cli_agent_messages and accumulated_context exactly.
        """
        with contextlib.suppress(Exception):
            path = session_path(session.session_id)
            if not path.exists():
                return

            lines = path.read_text(encoding="utf-8").splitlines()

            # Idempotency guard — already finalized, nothing to do.
            if lines:
                with contextlib.suppress(json.JSONDecodeError):
                    if json.loads(lines[-1]).get("type") == "session_end":
                        return

            # Count stats from turn stub records.
            chat_turns = 0
            investigation_turns = 0
            total_turns = 0
            detail_turns = 0
            for line in lines:
                with contextlib.suppress(json.JSONDecodeError):
                    rec = json.loads(line)
                    rec_type = rec.get("type")
                    if rec_type == "turn":
                        total_turns += 1
                        kind = rec.get("kind", "")
                        if kind in CHAT_KINDS:
                            chat_turns += 1
                        elif kind in ("alert", "incoming_alert"):
                            investigation_turns += 1
                    elif rec_type == "turn_detail":
                        detail_turns += 1

            if total_turns == 0 and detail_turns == 0:
                # Empty session — nothing useful happened; remove the file.
                path.unlink(missing_ok=True)
                return

            now = datetime.now(UTC)
            started_at = datetime.fromtimestamp(session.started_at, tz=UTC)
            duration_secs = max(0, int((now - started_at).total_seconds()))

            # Write conversation snapshot so /resume can restore exact LLM context.
            # Isolated suppress: a serialization failure must not prevent session_end.
            with contextlib.suppress(Exception):
                if session.agent.messages or session.accumulated_context:
                    snapshot: dict[str, Any] = {"type": "conversation_snapshot"}
                    if session.agent.messages:
                        snapshot["cli_agent_messages"] = [list(m) for m in session.agent.messages]
                    if session.accumulated_context:
                        snapshot["accumulated_context"] = dict(session.accumulated_context)
                    with path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

            record = {
                "type": "session_end",
                "ended_at": now.isoformat(),
                "duration_secs": duration_secs,
                "total_turns": total_turns,
                "chat_turns": chat_turns,
                "investigation_turns": investigation_turns,
            }
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def reopen_session(self, session_id: str) -> None:
        """Reopen a finalized session file so new turns append to the same file.

        Strips trailing ``conversation_snapshot`` and ``session_end`` records
        written by :meth:`flush`. No-op when the file is missing or still open.
        """
        with contextlib.suppress(Exception):
            path = session_path(session_id)
            if not path.exists():
                return

            lines = path.read_text(encoding="utf-8").splitlines()
            if not lines:
                return

            changed = False
            with contextlib.suppress(json.JSONDecodeError):
                if json.loads(lines[-1]).get("type") == "session_end":
                    lines.pop()
                    changed = True

            if lines:
                with contextlib.suppress(json.JSONDecodeError):
                    if json.loads(lines[-1]).get("type") == "conversation_snapshot":
                        lines.pop()
                        changed = True

            if not changed:
                return

            with path.open("w", encoding="utf-8") as fh:
                for line in lines:
                    fh.write(line + "\n")

    @staticmethod
    def _investigation_record_from_state(
        state: dict[str, Any],
        *,
        trigger: str,
        investigation_id: str | None = None,
    ) -> dict[str, Any]:
        report = state.get("problem_md") or state.get("slack_message") or state.get("report") or ""
        return {
            "type": "investigation_result",
            "investigation_id": investigation_id or uuid.uuid4().hex[:8],
            "completed_at": datetime.now(UTC).isoformat(),
            "trigger": trigger.strip()[:_TRIGGER_MAX_CHARS],
            "root_cause": str(state.get("root_cause") or ""),
            "report": str(report),
            "root_cause_category": str(state.get("root_cause_category") or ""),
            "alert_name": str(state.get("alert_name") or ""),
            "run_id": str(state.get("run_id") or ""),
        }

    def append_investigation_result(
        self,
        session_id: str,
        state: dict[str, Any],
        *,
        trigger: str = "",
    ) -> str:
        """Append a completed RCA record to the session file for /rca history.

        Returns the generated investigation_id. No-ops silently when the session
        file is missing or not writable.
        """
        investigation_id = uuid.uuid4().hex[:8]
        with contextlib.suppress(Exception):
            path = session_path(session_id)
            if not path.exists():
                return investigation_id
            self._ensure_session_open(session_id)
            record = self._investigation_record_from_state(
                state,
                trigger=trigger,
                investigation_id=investigation_id,
            )
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return investigation_id
