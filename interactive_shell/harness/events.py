"""Shell-agent event contracts.

This module is intentionally UI-free. The shell turn agent emits these events;
runtime presentation code decides how to render them.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Literal

AgentEventType = Literal["turn_start", "turn_interrupted", "turn_error", "turn_end"]


@dataclass(frozen=True)
class AgentEvent:
    """Lifecycle event emitted during one submitted shell turn."""

    type: AgentEventType
    text: str | None = None
    error: Exception | None = None


AgentEventSink = Callable[[AgentEvent], None]
AsyncAgentEventSink = Callable[[AgentEvent], Coroutine[Any, Any, None]]


__all__ = [
    "AgentEvent",
    "AgentEventSink",
    "AgentEventType",
    "AsyncAgentEventSink",
]
