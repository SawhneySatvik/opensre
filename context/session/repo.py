"""JSONL-backed cross-session repository.

Where :class:`~context.session.storage.jsonl.JsonlSessionStorage`
owns writes to a single session file, this repository owns read queries across
every session file under ``~/.opensre/sessions/``: listing recent sessions for
``/sessions``, loading one for ``/resume``, and browsing persisted RCA reports
for ``/rca``.
"""

from __future__ import annotations

import contextlib
import importlib
import json
from pathlib import Path
from typing import Any

paths = importlib.import_module("context.session.paths")
derive_name = paths.derive_name

_ROOT_CAUSE_PREVIEW_CHARS = 80
_DEFAULT_RCA_HISTORY_LIMIT = 50

# Turn kinds counted as conversational when computing per-session stats.
_CHAT_KINDS: frozenset[str] = frozenset({"chat", "cli_agent", "follow_up"})


class JsonlSessionRepo:
    """Read-only queries over all stored session files."""

    def load_recent(self, n: int = 20) -> list[dict[str, Any]]:
        """Return up to n session summaries, newest first.

        For completed sessions (have session_end), stats come from that record.
        For in-progress or crashed sessions (no session_end), stats are
        computed by scanning the turn records in the file so /sessions always
        shows accurate counts for the current session.
        """
        root = paths.sessions_dir()
        if not root.exists():
            return []

        # Sort by mtime descending so we only read the n most recent files
        # instead of every file in the directory. Guard against files that
        # disappear between the glob and the stat call (concurrent delete).
        def _mtime(p: Path) -> float:
            with contextlib.suppress(OSError):
                return p.stat().st_mtime
            return 0.0

        all_paths = sorted(root.glob("*.jsonl"), key=_mtime, reverse=True)

        results: list[dict[str, Any]] = []
        for path in all_paths[: n * 2]:  # 2× buffer for skipped/malformed files
            with contextlib.suppress(Exception):
                lines = path.read_text(encoding="utf-8").splitlines()
                if not lines:
                    continue

                start_record: dict[str, Any] | None = None
                with contextlib.suppress(json.JSONDecodeError):
                    start_record = json.loads(lines[0])

                if start_record is None or start_record.get("type") != "session_start":
                    continue

                end_record: dict[str, Any] | None = None
                has_snapshot = False
                with contextlib.suppress(json.JSONDecodeError):
                    last = json.loads(lines[-1])
                    if last.get("type") == "session_end":
                        end_record = last

                for line in lines:
                    with contextlib.suppress(json.JSONDecodeError):
                        if json.loads(line).get("type") == "conversation_snapshot":
                            has_snapshot = True
                            break

                if end_record is not None:
                    total_turns = end_record.get("total_turns")
                    chat_turns = end_record.get("chat_turns")
                    investigation_turns = end_record.get("investigation_turns")
                    duration_secs = end_record.get("duration_secs")
                else:
                    # In-progress or crashed — count from turn records
                    total_turns = 0
                    chat_turns = 0
                    investigation_turns = 0
                    for line in lines[1:]:
                        with contextlib.suppress(json.JSONDecodeError):
                            rec = json.loads(line)
                            if rec.get("type") != "turn":
                                continue
                            total_turns += 1
                            kind = rec.get("kind", "")
                            if kind in _CHAT_KINDS:
                                chat_turns += 1
                            elif kind in ("alert", "incoming_alert"):
                                investigation_turns += 1
                    duration_secs = None

                results.append(
                    {
                        "session_id": start_record.get("session_id", path.stem),
                        "name": paths.derive_name(lines),
                        "started_at": start_record.get("started_at"),
                        "opensre_version": start_record.get("opensre_version"),
                        "duration_secs": duration_secs,
                        "total_turns": total_turns,
                        "chat_turns": chat_turns,
                        "investigation_turns": investigation_turns,
                        "is_ended": end_record is not None,
                        "has_snapshot": has_snapshot,
                    }
                )

        results.sort(key=lambda x: x.get("started_at") or "", reverse=True)
        return results[:n]

    def count_prefix_matches(self, prefix: str) -> int:
        """Return how many session files whose stem starts with prefix.

        Used by /resume to distinguish 'not found' (0) from 'ambiguous' (>1)
        without re-scanning the directory with a fragile inline import.
        """
        root = paths.sessions_dir()
        if not root.exists():
            return 0
        with contextlib.suppress(OSError):
            return sum(1 for p in root.glob("*.jsonl") if p.stem.startswith(prefix))
        return 0

    def load_session(self, session_id_prefix: str) -> dict[str, Any] | None:
        """Load a session file and extract conversation data for /resume.

        Accepts a session ID prefix (e.g. the first 8 chars shown by /sessions).
        Returns None if no match found or the prefix is ambiguous.

        Resolution order for cli_agent_messages:
        1. conversation_snapshot (written at clean exit) — exact fidelity
        2. turn_detail records (written per-turn by PromptRecorder) — fallback
           for old files pre-enrichment or sessions that crashed before flush

        Returned dict keys:
          session_id, name, started_at, cli_agent_messages (list[tuple[str,str]]),
          accumulated_context, history (turn stubs), turn_details, has_snapshot
        """
        root = paths.sessions_dir()
        if not root.exists():
            return None

        target_path: Path | None = None
        for path in root.glob("*.jsonl"):
            if path.stem.startswith(session_id_prefix):
                if target_path is not None:
                    return None  # ambiguous prefix — caller should ask for more chars
                target_path = path

        if target_path is None:
            return None

        with contextlib.suppress(Exception):
            lines = target_path.read_text(encoding="utf-8").splitlines()
            if not lines:
                return None

            start_record: dict[str, Any] | None = None
            with contextlib.suppress(json.JSONDecodeError):
                start_record = json.loads(lines[0])

            if start_record is None or start_record.get("type") != "session_start":
                return None

            cli_agent_messages: list[tuple[str, str]] = []
            accumulated_context: dict[str, Any] = {}
            history: list[dict[str, Any]] = []
            turn_details: list[dict[str, Any]] = []
            has_snapshot = False

            for line in lines[1:]:
                with contextlib.suppress(json.JSONDecodeError):
                    rec = json.loads(line)
                    rec_type = rec.get("type")
                    if rec_type == "turn":
                        history.append(rec)
                    elif rec_type == "turn_detail":
                        turn_details.append(rec)
                    elif rec_type == "conversation_snapshot":
                        has_snapshot = True
                        msgs = rec.get("cli_agent_messages")
                        if msgs:
                            cli_agent_messages = [
                                (str(m[0]), str(m[1])) for m in msgs if len(m) >= 2
                            ]
                        ctx = rec.get("accumulated_context")
                        if ctx and isinstance(ctx, dict):
                            accumulated_context = ctx

            # Fall back to turn_detail reconstruction when no snapshot exists.
            if not cli_agent_messages and turn_details:
                for td in turn_details:
                    if td.get("kind") in ("chat", "follow_up"):
                        prompt = td.get("prompt") or ""
                        response = td.get("response") or ""
                        if prompt:
                            cli_agent_messages.append(("user", prompt))
                        if response:
                            cli_agent_messages.append(("assistant", response))

            return {
                "session_id": start_record.get("session_id", target_path.stem),
                "name": paths.derive_name(lines),
                "started_at": start_record.get("started_at"),
                "cli_agent_messages": cli_agent_messages,
                "accumulated_context": accumulated_context,
                "history": history,
                "turn_details": turn_details,
                "has_snapshot": has_snapshot,
            }

        return None

    @staticmethod
    def _collect_investigation_records(
        path: Path,
        *,
        lines: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if lines is None:
            with contextlib.suppress(Exception):
                lines = path.read_text(encoding="utf-8").splitlines()
            if not lines:
                return []

        session_id = path.stem
        session_name = paths.derive_name(lines)
        started_at: str | None = None
        with contextlib.suppress(json.JSONDecodeError):
            start = json.loads(lines[0])
            if start.get("type") == "session_start":
                session_id = str(start.get("session_id") or session_id)
                started_at = start.get("started_at")

        records: list[dict[str, Any]] = []
        for line in lines[1:]:
            with contextlib.suppress(json.JSONDecodeError):
                rec = json.loads(line)
                if rec.get("type") != "investigation_result":
                    continue
                root_cause = str(rec.get("root_cause") or "")
                preview = root_cause.replace("\n", " ").strip()
                if len(preview) > _ROOT_CAUSE_PREVIEW_CHARS:
                    preview = preview[: _ROOT_CAUSE_PREVIEW_CHARS - 1] + "…"
                records.append(
                    {
                        "investigation_id": str(rec.get("investigation_id") or ""),
                        "session_id": session_id,
                        "session_name": session_name,
                        "session_started_at": started_at,
                        "completed_at": rec.get("completed_at"),
                        "trigger": rec.get("trigger") or "",
                        "root_cause_preview": preview,
                        "root_cause": root_cause,
                        "report": str(rec.get("report") or ""),
                        "root_cause_category": rec.get("root_cause_category") or "",
                        "alert_name": rec.get("alert_name") or "",
                        "run_id": rec.get("run_id") or "",
                    }
                )
        return records

    def load_investigation_history(
        self, n: int = _DEFAULT_RCA_HISTORY_LIMIT
    ) -> list[dict[str, Any]]:
        """Return persisted RCA records across sessions, newest first."""
        root = paths.sessions_dir()
        if not root.exists():
            return []

        def _mtime(p: Path) -> float:
            with contextlib.suppress(OSError):
                return p.stat().st_mtime
            return 0.0

        all_paths = sorted(root.glob("*.jsonl"), key=_mtime, reverse=True)
        results: list[dict[str, Any]] = []
        for path in all_paths:
            with contextlib.suppress(Exception):
                lines = path.read_text(encoding="utf-8").splitlines()
                if not lines:
                    continue
                results.extend(self._collect_investigation_records(path, lines=lines))
            if len(results) >= n * 3:
                break

        results.sort(key=lambda item: item.get("completed_at") or "", reverse=True)
        return results[:n]

    @staticmethod
    def _scan_investigation_prefix(normalized: str) -> tuple[dict[str, Any] | None, int]:
        root = paths.sessions_dir()
        if not root.exists():
            return None, 0

        match: dict[str, Any] | None = None
        count = 0
        for path in root.glob("*.jsonl"):
            with contextlib.suppress(Exception):
                lines = path.read_text(encoding="utf-8").splitlines()
                for rec in JsonlSessionRepo._collect_investigation_records(path, lines=lines):
                    inv_id = str(rec.get("investigation_id") or "").lower()
                    if not inv_id.startswith(normalized):
                        continue
                    count += 1
                    if count == 1:
                        match = rec
                    else:
                        match = None
        return match, count

    def lookup_investigation(
        self, investigation_id_prefix: str
    ) -> tuple[dict[str, Any] | None, int]:
        """Return ``(record, match_count)`` for a prefix lookup.

        ``record`` is populated only when ``match_count == 1``.
        """
        normalized = investigation_id_prefix.strip().lower()
        if not normalized:
            return None, 0
        return self._scan_investigation_prefix(normalized)

    def load_investigation(self, investigation_id_prefix: str) -> dict[str, Any] | None:
        """Load one persisted RCA record by investigation_id prefix."""
        record, count = self.lookup_investigation(investigation_id_prefix)
        return record if count == 1 else None

    def count_investigation_prefix_matches(self, prefix: str) -> int:
        _, count = self.lookup_investigation(prefix)
        return count
