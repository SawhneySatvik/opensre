"""Interactive-shell LLM prompt context: typed models and prompt assembly.

Canonical export surface for the prompt layer. Only the runtime-free pieces are
re-exported here (typed models, the action-agent prompt builders, and the base
prompt text); ``interactive_shell.harness.llm_context.session`` is imported
widely, and every such import runs this package ``__init__``, so this module
deliberately stays free of the runtime / grounding / session stack.

The conversational assistant builder lives in ``assistant_prompt`` and is
imported from there directly by the runtime layer (it pulls in
``interactive_shell.runtime`` and the grounding caches).
"""

from __future__ import annotations

from interactive_shell.harness.llm_context.action_prompt import (
    build_action_system_prompt,
    build_action_user_message,
    connected_integrations_block,
    recent_conversation_block,
    sanitize_action_text,
)
from interactive_shell.harness.llm_context.action_prompt_text import SYSTEM_PROMPT_BASE
from interactive_shell.harness.llm_context.models import (
    CacheStats,
    ConversationMessage,
    PromptSection,
    coerce_messages,
    render_sections,
)

__all__ = [
    "SYSTEM_PROMPT_BASE",
    "CacheStats",
    "ConversationMessage",
    "PromptSection",
    "build_action_system_prompt",
    "build_action_user_message",
    "coerce_messages",
    "connected_integrations_block",
    "recent_conversation_block",
    "render_sections",
    "sanitize_action_text",
]
