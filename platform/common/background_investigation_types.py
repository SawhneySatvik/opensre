"""Background-investigation value types shared across opensre.

``BackgroundInvestigationRecord`` describes one completed or in-flight
background RCA. It lives in ``platform/common`` so the vendor notification
adapters under ``integrations/`` can depend on the record contract without
importing the CLI package, which the layering contract forbids.

The session-scoped notification preferences that select which channels a
record is delivered to stay in ``interactive_shell.session`` (a CLI-runtime
concern).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BackgroundInvestigationRecord:
    """One completed or in-flight background investigation tracked by the REPL."""

    task_id: str
    status: str
    command: str
    investigation_id: str = ""
    root_cause: str = ""
    top_analysis: tuple[str, ...] = ()
    next_steps: tuple[str, ...] = ()
    stats: dict[str, Any] = field(default_factory=dict)
    final_state: dict[str, Any] = field(default_factory=dict)
    notification_results: dict[str, str] = field(default_factory=dict)


__all__ = ["BackgroundInvestigationRecord"]
