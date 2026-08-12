"""Background-investigation value types.

Lives in ``platform/common`` so the vendor notification adapters under
``integrations/`` can depend on the record contract without importing the CLI
package, which the layering contract forbids. The session-scoped notification
preferences stay in ``interactive_shell.session``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


def moment(value: Any) -> float:
    """Coerce a stored epoch-seconds field; anything unusable becomes ``0.0``.

    Total by contract, because rows come from a file another writer may have
    edited and an ordering comparison must not be able to raise. Non-finite is
    rejected too: ``inf`` outranks every real timestamp and ``nan`` makes sort
    order arbitrary. ``json.loads`` produces both from ordinary input, and an
    integer literal too large for a float raises ``OverflowError``.
    """
    if not isinstance(value, int | float):
        return 0.0
    try:
        coerced = float(value)
    except OverflowError:
        return 0.0
    return coerced if math.isfinite(coerced) else 0.0


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
    # Epoch seconds, stamped by the store on write. Not default_factory=time.time:
    # a wall-clock default makes two otherwise identical records unequal.
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the durable store.

        ``final_state`` is omitted: it is unbounded and nothing reads it back.
        """
        return {
            "task_id": self.task_id,
            "status": self.status,
            "command": self.command,
            "investigation_id": self.investigation_id,
            "root_cause": self.root_cause,
            "top_analysis": list(self.top_analysis),
            "next_steps": list(self.next_steps),
            "stats": dict(self.stats),
            "notification_results": dict(self.notification_results),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BackgroundInvestigationRecord | None:
        """Rebuild a record from a stored row, or ``None`` when the row is unusable.

        Row-level validation matching ``TaskRecord.from_dict``: one corrupt entry
        is skipped rather than failing the whole document.
        """
        task_id = data.get("task_id")
        status = data.get("status")
        command = data.get("command")
        if not isinstance(task_id, str) or not isinstance(status, str):
            return None
        if not isinstance(command, str):
            return None

        def _text(key: str) -> str:
            value = data.get(key)
            return value if isinstance(value, str) else ""

        def _items(key: str) -> tuple[str, ...]:
            value = data.get(key)
            if not isinstance(value, list):
                return ()
            return tuple(str(item) for item in value)

        def _mapping(key: str) -> dict[str, Any]:
            value = data.get(key)
            return dict(value) if isinstance(value, Mapping) else {}

        return cls(
            task_id=task_id,
            status=status,
            command=command,
            investigation_id=_text("investigation_id"),
            root_cause=_text("root_cause"),
            top_analysis=_items("top_analysis"),
            next_steps=_items("next_steps"),
            stats=_mapping("stats"),
            notification_results={
                str(k): str(v) for k, v in _mapping("notification_results").items()
            },
            created_at=moment(data.get("created_at")),
            updated_at=moment(data.get("updated_at")),
        )


__all__ = ["BackgroundInvestigationRecord", "moment"]
