"""Shared recent-conversation context for interactive-shell prompt builders.

Single source of truth for rendering the recent CLI conversation so the action
planner and the conversational assistant see the same multi-turn history.
"""

from __future__ import annotations

from collections.abc import Sequence

from context.models import ConversationMessage, coerce_messages
from context.state import (
    MAX_CONVERSATION_MESSAGES,
    MAX_CONVERSATION_TURNS,
)

NO_HISTORY_PLACEHOLDER = "(no prior messages in this CLI thread)"

__all__ = [
    "MAX_CONVERSATION_MESSAGES",
    "MAX_CONVERSATION_TURNS",
    "NO_HISTORY_PLACEHOLDER",
    "format_recent_conversation",
]


def format_recent_conversation(
    messages: Sequence[ConversationMessage | tuple[str, str]],
    *,
    max_turns: int = MAX_CONVERSATION_TURNS,
) -> str:
    """Render recent CLI-agent turns as ``User:``/``Assistant:`` lines.

    Accepts a sequence of :class:`ConversationMessage` or raw ``(role, content)``
    pairs (oldest first) and normalizes either shape. Returns at most
    ``max_turns`` turns (oldest first, most recent last). Returns
    :data:`NO_HISTORY_PLACEHOLDER` when empty so prompt builders always have a
    stable, non-empty block. Never raises.
    """
    cap = max(max_turns, 0) * 2
    if not cap:
        return NO_HISTORY_PLACEHOLDER

    # Slice the raw tail first (matching the legacy cap-then-filter order), then
    # normalize, so malformed entries don't shift which turns are kept.
    recent = coerce_messages(messages[-cap:])
    lines = [message.render_line() for message in recent]
    return "\n".join(lines) if lines else NO_HISTORY_PLACEHOLDER
