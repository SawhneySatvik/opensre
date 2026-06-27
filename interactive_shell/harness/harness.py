"""Public entrypoint for interactive-shell agent turns."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from rich.console import Console

from config.llm_reasoning_effort import apply_reasoning_effort
from interactive_shell.chat.cli_agent import answer_cli_agent
from interactive_shell.chat.tool_gathering import gather_tool_evidence
from interactive_shell.harness.agent_actions import (
    TerminalActionExecutionResult,
    execute_cli_actions,
)
from interactive_shell.session import ReplSession
from interactive_shell.utils.telemetry import LlmRunInfo, PromptRecorder
from platform.analytics.cli import capture_terminal_turn_summarized

ExecuteActions = Callable[..., TerminalActionExecutionResult]
GatherEvidence = Callable[..., str | None]
AnswerAgent = Callable[..., LlmRunInfo | None]


@dataclass(frozen=True)
class ShellTurnResult:
    final_intent: str
    action_result: TerminalActionExecutionResult
    assistant_response_text: str = ""
    answered: bool = False
    llm_run: LlmRunInfo | None = None


def handle_message_with_agent(
    text: str,
    session: ReplSession,
    console: Console,
    *,
    recorder: PromptRecorder | None,
    confirm_fn: Callable[[str], str] | None = None,
    is_tty: bool | None = None,
    execute_actions: ExecuteActions | None = None,
    gather_evidence: GatherEvidence | None = None,
    answer_agent: AnswerAgent | None = None,
) -> ShellTurnResult:
    """Run one interactive-shell turn through action, observe, gather, and answer phases."""
    execute_actions = execute_actions or execute_cli_actions
    gather_evidence = gather_evidence or gather_tool_evidence
    answer_agent = answer_agent or answer_cli_agent

    # Clear any observation left by a prior turn so only this turn's discovery
    # output can trigger a summary pass.
    session.last_command_observation = None

    turn = execute_actions(
        text,
        session,
        console,
        confirm_fn=confirm_fn,
        is_tty=is_tty,
    )

    fallback_to_llm = not turn.handled
    snapshot = session.record_terminal_turn(
        executed_count=turn.executed_count,
        executed_success_count=turn.executed_success_count,
        fallback_to_llm=fallback_to_llm,
    )
    capture_terminal_turn_summarized(
        planned_count=turn.planned_count,
        executed_count=turn.executed_count,
        executed_success_count=turn.executed_success_count,
        fallback_to_llm=fallback_to_llm,
        session_turn_index=snapshot.turn_index,
        session_fallback_count=snapshot.fallback_count,
        session_action_success_percent=snapshot.action_success_percent,
        session_fallback_rate_percent=snapshot.fallback_rate_percent,
    )

    command_observation = session.last_command_observation

    if turn.handled and (turn.has_unhandled_clause or turn.executed_count > 0):
        if (
            command_observation
            and not turn.has_unhandled_clause
            and turn.executed_success_count > 0
        ):
            with apply_reasoning_effort(session.reasoning_effort):
                run = answer_agent(
                    text,
                    session,
                    console,
                    confirm_fn=confirm_fn,
                    is_tty=is_tty,
                    tool_observation=command_observation,
                )
            assistant_text = run.response_text if run is not None and run.response_text else ""
            if recorder is not None:
                recorder.set_response(assistant_text, run)
                recorder.flush()
            session.record("cli_agent", text)
            final_intent = "cli_agent_summarized"
            session.last_assistant_intent = final_intent
            return ShellTurnResult(
                final_intent=final_intent,
                action_result=turn,
                assistant_response_text=assistant_text,
                answered=True,
                llm_run=run,
            )

        final_intent = "cli_agent_denied" if turn.has_unhandled_clause else "cli_agent_handled"
        if recorder is not None:
            recorder.set_response(turn.response_text)
            recorder.flush()
        session.last_assistant_intent = final_intent
        return ShellTurnResult(
            final_intent=final_intent,
            action_result=turn,
            assistant_response_text=turn.response_text,
            answered=False,
        )

    with apply_reasoning_effort(session.reasoning_effort):
        gathered = gather_evidence(text, session, console, is_tty=is_tty)
        if gathered:
            run = answer_agent(
                text,
                session,
                console,
                confirm_fn=confirm_fn,
                is_tty=is_tty,
                tool_observation=gathered,
                tool_observation_on_screen=False,
            )
        else:
            run = answer_agent(
                text,
                session,
                console,
                confirm_fn=confirm_fn,
                is_tty=is_tty,
                tool_observation=None,
            )

    assistant_text = run.response_text if run is not None and run.response_text else ""
    if recorder is not None:
        recorder.set_response(assistant_text, run)
        recorder.flush()
    session.record("cli_agent", text)
    final_intent = "cli_agent_handoff" if turn.handled else "cli_agent_fallback"
    session.last_assistant_intent = final_intent
    return ShellTurnResult(
        final_intent=final_intent,
        action_result=turn,
        assistant_response_text=assistant_text,
        answered=True,
        llm_run=run,
    )


__all__ = ["ShellTurnResult", "handle_message_with_agent"]
