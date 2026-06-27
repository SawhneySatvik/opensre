"""Dedicated conversational-agent state for the interactive-shell harness.

Owns the agent's per-session runtime state (the multi-turn transcript and the
per-turn read-only discovery observation) so it lives in one focused place
rather than as loose fields scattered across ``ReplSession``.
"""

from __future__ import annotations

from interactive_shell.harness.state.conversation_state import (
    MAX_CONVERSATION_MESSAGES,
    MAX_CONVERSATION_TURNS,
    ConversationState,
)

__all__ = [
    "MAX_CONVERSATION_MESSAGES",
    "MAX_CONVERSATION_TURNS",
    "ConversationState",
]
