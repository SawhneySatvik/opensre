"""Conversational-agent state: transcript and per-turn observation.

``ConversationState`` is the interactive-shell analog of a stateful agent's
state object: it owns the multi-turn ``(role, content)`` transcript and the
ephemeral read-only discovery observation a turn may leave for summarization.
``ReplSession`` composes one of these instead of carrying the raw fields
itself, keeping all agent state in a single dedicated module.

This module is the canonical source of truth for the conversation-length caps;
prompt builders import them from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MAX_CONVERSATION_TURNS = 12
MAX_CONVERSATION_MESSAGES = MAX_CONVERSATION_TURNS * 2


@dataclass
class ConversationState:
    """Per-session conversational-agent state.

    ``messages`` is the alternating ``("user"|"assistant", text)`` transcript,
    oldest first, trimmed to the most recent ``MAX_CONVERSATION_MESSAGES``
    entries on each recorded turn. ``last_observation`` is the compact textual
    result of a read-only discovery command run during the current turn; it is
    set by discovery commands and reset at the start of every agent turn.
    """

    messages: list[tuple[str, str]] = field(default_factory=list)
    last_observation: str | None = None

    def record_turn(self, user_message: str, assistant_message: str) -> None:
        """Append one user/assistant exchange and trim to the message cap."""
        self.messages.append(("user", user_message))
        self.messages.append(("assistant", assistant_message))
        if len(self.messages) > MAX_CONVERSATION_MESSAGES:
            self.messages[:] = self.messages[-MAX_CONVERSATION_MESSAGES:]

    def reset_observation(self) -> None:
        """Clear any discovery observation left by a prior turn."""
        self.last_observation = None

    def clear(self) -> None:
        """Reset the transcript and any pending observation (used by /new)."""
        self.messages.clear()
        self.last_observation = None


__all__ = [
    "MAX_CONVERSATION_MESSAGES",
    "MAX_CONVERSATION_TURNS",
    "ConversationState",
]
