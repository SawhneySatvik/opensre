"""Shared agent state for runtime request assembly.

Owns the mutable per-session store and immutable read models used to assemble
runtime requests without exposing live mutable internals.
"""

from __future__ import annotations

from context.state.agent_state import (
    AgentContextInput,
    AgentMessageRole,
    AgentModelInfo,
    AgentRunStatus,
    AgentStateChange,
    AgentStateError,
    AgentStateSnapshot,
    MAX_CONVERSATION_MESSAGES,
    MAX_CONVERSATION_TURNS,
    MutableAgentState,
    create_mutable_agent_state,
)

__all__ = [
    "AgentContextInput",
    "AgentMessageRole",
    "AgentModelInfo",
    "AgentRunStatus",
    "AgentStateChange",
    "AgentStateError",
    "AgentStateSnapshot",
    "MAX_CONVERSATION_MESSAGES",
    "MAX_CONVERSATION_TURNS",
    "MutableAgentState",
    "create_mutable_agent_state",
]
