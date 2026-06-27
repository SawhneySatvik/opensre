"""Turn routing for one interactive-shell turn.

Decides, from the tool-calling action result and any left-over discovery
observation, which of three paths a turn takes (summarize an observation,
finish without the LLM, or gather evidence and answer), then performs the
chosen path's effects. The path choice is the pure :func:`_route_turn`; this
module is the imperative shell around it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, assert_never

from rich.console import Console

from config.llm_reasoning_effort import apply_reasoning_effort
from interactive_shell.harness.response import generate_response
from interactive_shell.harness.tool_calling import run_tool_calling_turn
from interactive_shell.harness.turn_context import TurnContext
from interactive_shell.runtime import ReplSession
from interactive_shell.runtime.core.turn_accounting import (
    ShellTurnAccounting,
    ShellTurnResult,
    ToolCallingTurnResult,
)
from interactive_shell.tools.tool_gathering import gather_tool_evidence
from interactive_shell.utils.telemetry import LlmRunInfo, PromptRecorder

RunToolCallingTurn = Callable[..., ToolCallingTurnResult]
GatherEvidence = Callable[..., str | None]
ResponseGenerator = Callable[..., LlmRunInfo | None]


def _response_text(run: LlmRunInfo | None) -> str:
    return run.response_text if run is not None and run.response_text else ""


def _route_turn(
    action_result: ToolCallingTurnResult, observation: str | None
) -> Literal["summarize_observation", "handled_without_llm", "gather_and_answer"]:
    """Decide the turn path from the action result and any left-over observation."""
    if (
        action_result.handled
        and observation is not None
        and action_result.executed_success_count > 0
    ):
        return "summarize_observation"
    if action_result.handled:
        return "handled_without_llm"
    return "gather_and_answer"


def _gather_and_answer(
    *,
    text: str,
    session: ReplSession,
    console: Console,
    gather_evidence: GatherEvidence,
    response_generator: ResponseGenerator,
    confirm_fn: Callable[[str], str] | None,
    is_tty: bool | None,
    turn_ctx: TurnContext,
) -> LlmRunInfo | None:
    gathered = gather_evidence(text, session, console, is_tty=is_tty)

    # When evidence was gathered, mark it off-screen so the prompt builder
    # includes it. When nothing was gathered, omit the flag entirely so the
    # call shape matches the plain conversational (no-observation) path.
    on_screen: dict[str, bool] = {"tool_observation_on_screen": False} if gathered else {}

    return response_generator(
        text,
        session,
        console,
        confirm_fn=confirm_fn,
        is_tty=is_tty,
        tool_observation=gathered or None,
        turn_ctx=turn_ctx,
        **on_screen,
    )


def handle_message_with_agent(
    text: str,
    session: ReplSession,
    console: Console,
    *,
    recorder: PromptRecorder | None,
    confirm_fn: Callable[[str], str] | None = None,
    is_tty: bool | None = None,
    execute_actions: RunToolCallingTurn | None = None,
    gather_evidence: GatherEvidence | None = None,
    response_generator: ResponseGenerator | None = None,
) -> ShellTurnResult:
    """Run one interactive-shell turn through three paths, in order:

    1. ``summarize_observation`` — a successful action left discovery output, so
       summarize it into a direct answer.
    2. ``handled_without_llm`` — the action fully handled the turn; stop without the LLM.
    3. ``gather_and_answer`` — nothing was handled; gather evidence and answer.

    The path choice is the pure ``_route_turn``; this function is the imperative
    shell that performs the chosen path's effects.
    """
    execute_actions = execute_actions or run_tool_calling_turn
    gather_evidence = gather_evidence or gather_tool_evidence
    response_generator = response_generator or generate_response

    # Snapshot session state before any turn mutations. Both the action agent
    # and the conversational assistant read from this frozen context so their
    # prompts reflect a consistent turn-start view rather than live session state.
    turn_ctx = TurnContext.from_session(text, session)
    accounting = ShellTurnAccounting(session=session, text=text, recorder=recorder)

    # Clear any observation left by a prior turn so only this turn's discovery
    # output can trigger a summary pass.
    session.agent.reset_observation()

    action_result = execute_actions(
        text,
        session,
        console,
        confirm_fn=confirm_fn,
        is_tty=is_tty,
        turn_ctx=turn_ctx,
    )
    accounting.record_action_result(action_result)

    observation = session.agent.last_observation

    route = _route_turn(action_result, observation)
    match route:
        case "summarize_observation":
            with apply_reasoning_effort(turn_ctx.reasoning_effort):
                run = response_generator(
                    text,
                    session,
                    console,
                    confirm_fn=confirm_fn,
                    is_tty=is_tty,
                    tool_observation=observation,
                    turn_ctx=turn_ctx,
                )
            result = ShellTurnResult(
                final_intent="cli_agent_summarized",
                action_result=action_result,
                assistant_response_text=_response_text(run),
                llm_run=run,
            )

        case "handled_without_llm":
            result = ShellTurnResult(
                final_intent="cli_agent_handled",
                action_result=action_result,
                assistant_response_text=action_result.response_text,
            )

        case "gather_and_answer":
            with apply_reasoning_effort(turn_ctx.reasoning_effort):
                run = _gather_and_answer(
                    text=text,
                    session=session,
                    console=console,
                    gather_evidence=gather_evidence,
                    response_generator=response_generator,
                    confirm_fn=confirm_fn,
                    is_tty=is_tty,
                    turn_ctx=turn_ctx,
                )
            result = ShellTurnResult(
                final_intent="cli_agent_fallback",
                action_result=action_result,
                assistant_response_text=_response_text(run),
                llm_run=run,
            )

        case _:
            assert_never(route)

    return accounting.finalize(result)


__all__ = [
    "GatherEvidence",
    "ResponseGenerator",
    "RunToolCallingTurn",
    "handle_message_with_agent",
]
