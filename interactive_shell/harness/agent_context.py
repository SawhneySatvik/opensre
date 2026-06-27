"""Agent-owned per-turn immutable context snapshot.

Assembled once at the start of each turn from the live ``ReplSession``.
All fields reflect session state at turn-start and do not change while the
turn runs, so downstream code reads a stable snapshot rather than a live,
concurrently-mutated object.

Usage::

    agent_ctx = AgentContext.from_session(text, session)
    # pass agent_ctx to action agent + conversational assistant
    # keep passing session for writes (recording history, token usage, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from interactive_shell.harness.llm_context.conversation_history import MAX_CONVERSATION_MESSAGES
from interactive_shell.harness.llm_context.models import ConversationMessage

if TYPE_CHECKING:
    from config.llm_reasoning_effort import ReasoningEffortChoice
    from interactive_shell.harness.llm_context.session import ReplSession


@dataclass(frozen=True)
class AgentContext:
    """Immutable per-turn snapshot assembled from ``ReplSession`` at turn start.

    Carries everything the action agent and conversational assistant need to
    build prompts and ground answers, frozen at the moment the turn begins.

    The live ``ReplSession`` is still passed separately to callers that need
    to write state (recording history, persisting token usage, updating intent).
    """

    text: str
    """Raw user input text for this turn."""

    conversation_messages: tuple[ConversationMessage, ...]
    """Snapshot of recent CLI conversation as :class:`ConversationMessage`
    values, oldest first, capped to ``MAX_CONVERSATION_MESSAGES`` entries at
    assembly time."""

    configured_integrations: tuple[str, ...]
    """Integration names known to be configured at turn start."""

    configured_integrations_known: bool
    """Whether ``configured_integrations`` reflects real state (vs unknown)."""

    last_state: dict[str, Any] | None
    """Final ``AgentState`` from the most recent investigation (follow-up grounding)."""

    last_synthetic_observation_path: str | None
    """Path to latest synthetic-run observation file (failure explanation context)."""

    reasoning_effort: ReasoningEffortChoice | None
    """Session-scoped reasoning effort preference for LLM calls this turn."""

    @classmethod
    def from_session(cls, text: str, session: ReplSession) -> AgentContext:
        """Snapshot the relevant session fields for one turn.

        Call this once at the top of ``ShellTurnAgent.run_turn`` before any
        mutations happen, then pass the returned context downstream.
        """
        messages = session.agent.messages
        snapshot: tuple[ConversationMessage, ...] = tuple(
            ConversationMessage.from_role_content(role, content)
            for role, content in messages[-MAX_CONVERSATION_MESSAGES:]
            if isinstance(role, str) and isinstance(content, str)
        )
        return cls(
            text=text,
            conversation_messages=snapshot,
            configured_integrations=tuple(session.configured_integrations),
            configured_integrations_known=bool(session.configured_integrations_known),
            last_state=session.last_state,
            last_synthetic_observation_path=session.last_synthetic_observation_path,
            reasoning_effort=session.reasoning_effort,
        )


__all__ = ["AgentContext"]
