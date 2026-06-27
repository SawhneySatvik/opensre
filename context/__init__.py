"""Top-level context assembly and runtime-request package for OpenSRE.

This package owns shared prompt/context builders, session context, and the
``AgentContext`` request object consumed by runtime surfaces.
"""

from context.action_prompt import (
    build_action_system_prompt,
    build_action_user_message,
    connected_integrations_block,
    recent_conversation_block,
    sanitize_action_text,
)
from context.action_prompt_text import SYSTEM_PROMPT_BASE
from context.agent_context import AgentContext

__all__ = [
    "AgentContext",
    "SYSTEM_PROMPT_BASE",
    "build_action_system_prompt",
    "build_action_user_message",
    "connected_integrations_block",
    "recent_conversation_block",
    "sanitize_action_text",
]
