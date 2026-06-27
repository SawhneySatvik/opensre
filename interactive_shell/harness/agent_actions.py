"""Shell action execution through the shared agent harness."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.markup import escape

from core.runtime.agent import Agent
from core.runtime.llm.agent_llm_client import AgentLLMResponse, ToolCall
from integrations.llm_cli.failure_explain import is_context_length_overflow
from interactive_shell.harness.llm_context import (
    build_action_system_prompt,
    build_action_user_message,
)
from interactive_shell.harness.state.conversation_history import MAX_CONVERSATION_MESSAGES
from interactive_shell.runtime import ReplSession
from interactive_shell.tools.tool_contracts import ToolContext
from interactive_shell.tools.tool_registry import REGISTRY
from interactive_shell.ui.action_rendering import ActionRenderObserver
from interactive_shell.ui.streaming import render_response_header
from interactive_shell.utils.error_handling.exception_reporting import report_exception

logger = logging.getLogger(__name__)

# Some hosted tool-calling models emit one tool call per assistant turn even when
# parallel tool calls are enabled. Keep the action loop bounded, but allow the
# shared AgentTool path to continue through a two-action compound request and a
# final no-tool response.
_MAX_ACTION_ITERATIONS = 3
_EXECUTED_HISTORY_TYPES = {
    "slash",
    "shell",
    "alert",
    "synthetic_test",
    "implementation",
    "cli_command",
}


@dataclass(frozen=True)
class TerminalActionExecutionResult:
    planned_count: int
    executed_count: int
    executed_success_count: int
    has_unhandled_clause: bool
    handled: bool
    response_text: str = ""


@dataclass(frozen=True)
class ActionExecutionDeps:
    """Optional dependency seams used by tests/harnesses."""

    llm_factory: Callable[[], Any] | None = None


class _StaticToolCallLLM:
    """Deterministic one-shot LLM used for explicit non-LLM shell commands."""

    def __init__(self, tool_calls: list[ToolCall]) -> None:
        self._tool_calls = tool_calls
        self._used = False

    def tool_schemas(self, _tools: list[Any]) -> list[dict[str, Any]]:
        return []

    def invoke(
        self,
        _messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AgentLLMResponse:
        _ = system
        _ = tools
        if self._used:
            return AgentLLMResponse(content="", tool_calls=[], raw_content=None)
        self._used = True
        return AgentLLMResponse(content="", tool_calls=self._tool_calls, raw_content=None)

    @staticmethod
    def build_assistant_message(content: str, tool_calls: list[ToolCall]) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.input} for tc in tool_calls
            ],
        }

    @staticmethod
    def build_tool_result_message(
        tool_calls: list[ToolCall],
        results: list[Any],
    ) -> dict[str, Any]:
        return {
            "role": "tool",
            "content": json.dumps(
                [
                    {"id": tc.id, "name": tc.name, "result": result}
                    for tc, result in zip(tool_calls, results)
                ],
                default=str,
            ),
        }


def _response_text_from_history_entries(entries: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for item in entries:
        response_text = item.get("response_text")
        if isinstance(response_text, str) and response_text.strip():
            chunks.append(response_text.strip())
    return "\n".join(chunks)


def _persist_action_agent_error(session: ReplSession, user_text: str, error_text: str) -> None:
    session.cli_agent_messages.append(("user", user_text))
    session.cli_agent_messages.append(("assistant", error_text))
    if len(session.cli_agent_messages) > MAX_CONVERSATION_MESSAGES:
        session.cli_agent_messages[:] = session.cli_agent_messages[-MAX_CONVERSATION_MESSAGES:]


def _render_action_agent_error(console: Console, message: str) -> None:
    console.print()
    render_response_header(console, "assistant")
    console.print(f"[yellow]{escape(message)}[/]")


def _bang_shell_command(message: str) -> str | None:
    # The only deterministic action bypass allowed in this module is the explicit
    # `!cmd` shell escape. Do NOT copy this pattern for `/slash` commands, bare
    # aliases, regex/keyword matches, or "obvious" natural-language intents.
    # Those must go through the action-agent LLM selecting first-class AgentTools.
    # Engineers have been fired before for reintroducing slash/regex shortcuts here.
    stripped = message.strip()
    if not stripped.startswith("!") or len(stripped) <= 1:
        return None
    cmd = " ".join(stripped[1:].split())
    return f"!{cmd}" if cmd else None


def _default_llm_factory() -> Any:
    from core.runtime.llm import agent_llm_client

    return agent_llm_client.get_agent_llm()


def _record_success_analytics(summary: TerminalActionExecutionResult) -> None:
    """Emit the planned/policy/executed analytics for a completed action turn."""
    from platform.analytics.cli import (
        capture_repl_execution_policy_decision,
        capture_terminal_actions_executed,
        capture_terminal_actions_planned,
    )

    capture_terminal_actions_planned(
        planned_count=summary.planned_count,
        has_unhandled_clause=False,
    )
    capture_repl_execution_policy_decision(
        {
            "policy_stage": "shell_action_agent",
            "policy_trace": "agent_tool_calls" if summary.planned_count else "assistant_handoff",
            "planned_count": summary.planned_count,
            "has_unhandled_clause": False,
        }
    )
    capture_terminal_actions_executed(
        planned_count=summary.planned_count,
        executed_count=summary.executed_count,
        executed_success_count=summary.executed_success_count,
    )


def _record_failure_analytics() -> None:
    """Emit the executed-analytics no-op used when a turn never ran any action."""
    from platform.analytics.cli import capture_terminal_actions_executed

    capture_terminal_actions_executed(
        planned_count=0,
        executed_count=0,
        executed_success_count=0,
    )


def execute_cli_actions(
    message: str,
    session: ReplSession,
    console: Console,
    *,
    confirm_fn: Callable[[str], str] | None = None,
    is_tty: bool | None = None,
    deps: ActionExecutionDeps | None = None,
) -> TerminalActionExecutionResult:
    """Run one shell action-selection turn through the shared agent harness."""
    history_start = len(session.history)
    ctx = ToolContext(
        session=session,
        console=console,
        confirm_fn=confirm_fn,
        is_tty=is_tty,
        action_already_listed=True,
    )
    tools = REGISTRY.agent_tools_for_context(ctx)
    observer = ActionRenderObserver(session=session, console=console, message=message)

    bang_command = _bang_shell_command(message)
    if bang_command is not None:
        # This is intentionally limited to the `!` shell escape. It is not a
        # general "deterministic command" fast path. In particular, do not add
        # `deterministic_command_text`, slash-command parsing, or regex intent
        # matching here. Slash execution still belongs to the `slash_invoke`
        # AgentTool selected by the action agent.

        def llm_factory() -> _StaticToolCallLLM:
            return _StaticToolCallLLM(
                [ToolCall(id="direct_shell_0", name="shell_run", input={"command": bang_command})]
            )

        user_message = message
        system_prompt = "Execute the explicit shell_run tool call."
    else:
        llm_factory = (
            deps.llm_factory if deps is not None and deps.llm_factory else _default_llm_factory
        )
        user_message = build_action_user_message(message)
        system_prompt = build_action_system_prompt(session)

    try:
        result = Agent(
            llm=llm_factory(),
            system=system_prompt,
            tools=tools,
            resolved_integrations={},
            max_iterations=_MAX_ACTION_ITERATIONS,
            on_event=observer,
        ).run([{"role": "user", "content": user_message}])
    except Exception as exc:
        if is_context_length_overflow(str(exc)):
            logger.debug(
                "shell action prompt overflow; falling through to assistant", exc_info=True
            )
            _record_failure_analytics()
            return TerminalActionExecutionResult(0, 0, 0, False, False)
        else:
            error_text = str(exc)
            report_exception(exc, context="interactive_shell.action_agent", expected=True)
            _render_action_agent_error(console, error_text)
            _persist_action_agent_error(session, message, error_text)
            session.record("cli_agent", message, ok=False)
        _record_failure_analytics()
        return TerminalActionExecutionResult(0, 0, 0, True, True, response_text=error_text)

    executed_entries = [
        item
        for item in session.history[history_start:]
        if item.get("type") in _EXECUTED_HISTORY_TYPES
    ]
    executed_count = len(executed_entries)
    executed_success_count = sum(1 for item in executed_entries if item.get("ok", True))
    planned_count = sum(1 for tc, _output in result.executed if tc.name != "assistant_handoff")
    handled = planned_count > 0
    response_text = _response_text_from_history_entries(executed_entries)
    if handled:
        console.print()

    summary = TerminalActionExecutionResult(
        planned_count,
        executed_count,
        executed_success_count,
        False,
        handled,
        response_text=response_text,
    )
    _record_success_analytics(summary)
    return summary


__all__ = [
    "ActionExecutionDeps",
    "TerminalActionExecutionResult",
    "execute_cli_actions",
]
