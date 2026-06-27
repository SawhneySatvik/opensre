"""Turn routing for one interactive-shell turn.

Decides, from the tool-calling action result and any left-over discovery
observation, which of three paths a turn takes (summarize an observation,
finish without the LLM, or gather evidence and answer), then performs the
chosen path's effects. The path choice is the pure :func:`_route_turn`; this
module is the imperative shell around it.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import Literal, assert_never

from rich.console import Console

from config.llm_reasoning_effort import apply_reasoning_effort
from interactive_shell.harness.events import AgentEvent, AgentEventSink
from interactive_shell.harness.llm_context.session import ReplSession
from interactive_shell.harness.response import generate_response
from interactive_shell.harness.state import ConversationState
from interactive_shell.harness.tool_calling import run_tool_calling_turn
from interactive_shell.harness.turn_context import TurnContext
from interactive_shell.runtime.core.confirmation import DispatchCancelled
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


class ShellTurnAgent:
    """Stateful owner of the interactive-shell turn lifecycle.

    The shell agent owns shell/session state, turn snapshots, route selection,
    observation handling, accounting, and lifecycle events. It delegates the
    inner tool-calling loop to ``run_tool_calling_turn``, which may use
    ``core.runtime.agent.Agent`` as a disposable primitive.
    """

    def __init__(
        self,
        session: ReplSession,
        *,
        execute_actions: RunToolCallingTurn | None = None,
        gather_evidence: GatherEvidence | None = None,
        response_generator: ResponseGenerator | None = None,
        event_sink: AgentEventSink | None = None,
    ) -> None:
        self.session = session
        self._execute_actions = execute_actions or run_tool_calling_turn
        self._gather_evidence = gather_evidence or gather_tool_evidence
        self._response_generator = response_generator or generate_response
        self._event_sinks: list[AgentEventSink] = []
        if event_sink is not None:
            self.subscribe(event_sink)

    @property
    def state(self) -> ConversationState:
        """Return the shell-owned conversational state for this session."""
        return self.session.agent

    def subscribe(self, sink: AgentEventSink) -> Callable[[], None]:
        """Subscribe to lifecycle events and return an unsubscribe callback."""
        self._event_sinks.append(sink)

        def _unsubscribe() -> None:
            with suppress(ValueError):
                self._event_sinks.remove(sink)

        return _unsubscribe

    def run_turn(
        self,
        text: str,
        *,
        console: Console,
        recorder: PromptRecorder | None,
        confirm_fn: Callable[[str], str] | None = None,
        is_tty: bool | None = None,
    ) -> ShellTurnResult:
        """Run one interactive-shell turn through the shell agent lifecycle."""
        self._emit(AgentEvent(type="turn_start", text=text))
        try:
            return self._run_turn_body(
                text,
                console=console,
                recorder=recorder,
                confirm_fn=confirm_fn,
                is_tty=is_tty,
            )
        except DispatchCancelled:
            self._emit(AgentEvent(type="turn_interrupted"))
            raise
        except Exception as exc:
            self._emit(AgentEvent(type="turn_error", error=exc))
            raise
        finally:
            self._emit(AgentEvent(type="turn_end"))

    def _run_turn_body(
        self,
        text: str,
        *,
        console: Console,
        recorder: PromptRecorder | None,
        confirm_fn: Callable[[str], str] | None,
        is_tty: bool | None,
    ) -> ShellTurnResult:
        """Perform the chosen turn path after lifecycle setup."""
        # Snapshot session state before any turn mutations. Both the action
        # agent and the conversational assistant read from this frozen context
        # so prompts reflect a consistent turn-start view rather than live
        # session state.
        turn_ctx = TurnContext.from_session(text, self.session)
        accounting = ShellTurnAccounting(session=self.session, text=text, recorder=recorder)

        # Clear any observation left by a prior turn so only this turn's
        # discovery output can trigger a summary pass.
        self.session.agent.reset_observation()

        action_result = self._execute_actions(
            text,
            self.session,
            console,
            confirm_fn=confirm_fn,
            is_tty=is_tty,
            turn_ctx=turn_ctx,
        )
        accounting.record_action_result(action_result)

        observation = self.session.agent.last_observation

        route = _route_turn(action_result, observation)
        match route:
            case "summarize_observation":
                with apply_reasoning_effort(turn_ctx.reasoning_effort):
                    run = self._response_generator(
                        text,
                        self.session,
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
                        session=self.session,
                        console=console,
                        gather_evidence=self._gather_evidence,
                        response_generator=self._response_generator,
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

    def _emit(self, event: AgentEvent) -> None:
        for sink in tuple(self._event_sinks):
            sink(event)


__all__ = [
    "GatherEvidence",
    "ResponseGenerator",
    "RunToolCallingTurn",
    "ShellTurnAgent",
]
