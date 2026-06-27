"""Interactive shell turn handling and dispatch helpers."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Awaitable, Callable, Coroutine, Iterator
from dataclasses import dataclass
from typing import Any, Literal

from rich.console import Console
from rich.markup import escape

from config.llm_reasoning_effort import apply_reasoning_effort
from interactive_shell.chat.cli_agent import answer_cli_agent
from interactive_shell.harness.tool_calling import run_tool_calling_turn
from interactive_shell.runtime.background.workers import BackgroundTaskManager
from interactive_shell.runtime.core.state import (
    PROMPT_REFRESH_INTERVAL_S,
    ReplState,
    SpinnerState,
)
from interactive_shell.runtime.input import (
    PromptInputReader,
)
from interactive_shell.runtime.input.actions import (
    InputAction,
    ShellInputSnapshot,
    decide_input_action,
)
from interactive_shell.runtime.utils.input_policy import (
    turn_needs_exclusive_stdin,
    turn_should_show_spinner,
)
from interactive_shell.session import (
    ReplSession,
)
from interactive_shell.tools.tool_gathering import gather_tool_evidence
from interactive_shell.turn_accounting import (
    ShellTurnAccounting,
    ShellTurnResult,
    ToolCallingTurnResult,
)
from interactive_shell.ui import ERROR, WARNING
from interactive_shell.ui.components.cpr_stdin import drain_stale_cpr_bytes
from interactive_shell.ui.output.repl_progress import repl_safe_progress_scope
from interactive_shell.ui.streaming.console import StreamingConsole
from interactive_shell.utils.error_handling.exception_reporting import report_exception
from interactive_shell.utils.telemetry import LlmRunInfo, PromptRecorder
from platform.analytics.repl_context import bind_cli_session_id, reset_cli_session_id

log = logging.getLogger(__name__)

_AGENT_TURN_KIND = "agent"

RunToolCallingTurn = Callable[..., ToolCallingTurnResult]
GatherEvidence = Callable[..., str | None]
AnswerAgent = Callable[..., LlmRunInfo | None]


def _response_text(run: LlmRunInfo | None) -> str:
    return run.response_text if run is not None and run.response_text else ""


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
    answer_agent: AnswerAgent | None = None,
) -> ShellTurnResult:
    """Run one interactive-shell turn through three paths, in order:

    1. ``answer_from_observation`` — a successful action left discovery output, so
       summarize it into a direct answer.
    2. ``action_handled`` — the action fully handled the turn; stop without the LLM.
    3. ``gather_and_answer`` — nothing was handled; gather evidence and answer.
    """
    execute_actions = execute_actions or run_tool_calling_turn
    gather_evidence = gather_evidence or gather_tool_evidence
    answer_agent = answer_agent or answer_cli_agent

    accounting = ShellTurnAccounting(session=session, text=text, recorder=recorder)

    # Clear any observation left by a prior turn so only this turn's discovery
    # output can trigger a summary pass.
    session.last_command_observation = None

    action_result = execute_actions(
        text,
        session,
        console,
        confirm_fn=confirm_fn,
        is_tty=is_tty,
    )
    accounting.record_action_result(action_result)

    observation = session.last_command_observation

    if (
        action_result.handled
        and observation is not None
        and action_result.executed_success_count > 0
    ):
        # Path 1: a successful terminal action left discovery output worth summarizing.
        with apply_reasoning_effort(session.reasoning_effort):
            run = answer_agent(
                text,
                session,
                console,
                confirm_fn=confirm_fn,
                is_tty=is_tty,
                tool_observation=observation,
            )
        result = ShellTurnResult(
            final_intent="cli_agent_summarized",
            action_result=action_result,
            assistant_response_text=_response_text(run),
            llm_run=run,
        )
    elif action_result.handled:
        # Path 2: the action fully handled the turn; stop without the LLM.
        result = ShellTurnResult(
            final_intent="cli_agent_handled",
            action_result=action_result,
            assistant_response_text=action_result.response_text,
        )
    else:
        # Path 3: nothing was handled; gather evidence and answer.
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
        result = ShellTurnResult(
            final_intent="cli_agent_fallback",
            action_result=action_result,
            assistant_response_text=_response_text(run),
            llm_run=run,
        )

    return accounting.finalize(result)


@dataclass(frozen=True)
class AgentEvent:
    """Agent lifecycle event emitted during one submitted shell turn."""

    type: Literal["turn_start", "turn_interrupted", "turn_error", "turn_end"]
    text: str | None = None
    error: Exception | None = None


AgentEventSink = Callable[[AgentEvent], Awaitable[None]]


class DispatchCancelled(Exception):
    """Raised when in-flight dispatch is cancelled during confirmation."""


@contextlib.contextmanager
def _bound_cli_session(session_id: str) -> Iterator[None]:
    token = bind_cli_session_id(session_id)
    try:
        yield
    finally:
        reset_cli_session_id(token)


class ConsoleAgentEventSink:
    """Render agent lifecycle events to the terminal console."""

    def __init__(
        self,
        *,
        session: ReplSession,
        spinner: SpinnerState,
        console: StreamingConsole,
    ) -> None:
        self.session = session
        self.spinner = spinner
        self.console = console
        self.show_spinner = False

    async def __call__(self, event: AgentEvent) -> None:
        match event.type:
            case "turn_start":
                await self._turn_start(event)
            case "turn_interrupted":
                await self._turn_interrupted()
            case "turn_error":
                await self._turn_error(event)
            case "turn_end":
                await self._turn_end()
            case _:
                raise ValueError(f"Unknown agent event type: {event.type!r}")

    async def _turn_start(self, event: AgentEvent) -> None:
        from interactive_shell.ui.output import set_prompt_suppress_fn

        text = event.text or ""
        self.show_spinner = turn_should_show_spinner(text, self.session)
        if self.show_spinner:
            self.spinner.start()
            set_prompt_suppress_fn(self.console.suppress_prompt_spinner)

    async def _turn_interrupted(self) -> None:
        self.console.print(f"[{WARNING}]· interrupted[/]")

    async def _turn_error(self, event: AgentEvent) -> None:
        exc = event.error
        if exc is None:
            raise ValueError("turn_error event requires an error")
        self.console.print(f"[{ERROR}]turn error:[/] {escape(str(exc))}")

    async def _turn_end(self) -> None:
        from interactive_shell.ui.output import set_prompt_suppress_fn

        set_prompt_suppress_fn(None)
        if self.show_spinner:
            self.spinner.stop()
        await asyncio.sleep(0.05)
        drain_stale_cpr_bytes()


class AgentTurnRunner:
    """Run one submitted shell turn through the agent harness."""

    def __init__(
        self,
        *,
        session: ReplSession,
        state: ReplState,
        spinner: SpinnerState,
        invalidate_prompt: Callable[[], None],
    ) -> None:
        self.session = session
        self.state = state
        self.spinner = spinner
        self.invalidate_prompt = invalidate_prompt

    async def run_agent_turn(self, text: str) -> None:
        """Set up shell presentation for one turn and drive its lifecycle."""
        dispatch_cancel = threading.Event()
        console = StreamingConsole(
            self.spinner,
            dispatch_cancel,
            prompt_invalidator=self.invalidate_prompt,
            highlight=False,
            force_terminal=True,
            color_system="truecolor",
            legacy_windows=False,
        )
        emit = ConsoleAgentEventSink(
            session=self.session,
            spinner=self.spinner,
            console=console,
        )
        recorder = PromptRecorder.start(
            session=self.session,
            text=text,
            turn_kind=_AGENT_TURN_KIND,
        )
        progress_scope = (
            contextlib.nullcontext()
            if turn_needs_exclusive_stdin(text, self.session)
            else repl_safe_progress_scope()
        )
        with progress_scope:
            await self._run_loop(
                text=text,
                output=console,
                recorder=recorder,
                confirm=lambda prompt: request_confirmation_via_prompt(self.state, prompt),
                emit=emit,
                dispatch_cancel=dispatch_cancel,
            )

    async def _run_loop(
        self,
        *,
        text: str,
        output: StreamingConsole,
        recorder: PromptRecorder | None,
        confirm: Callable[[str], str],
        emit: AgentEventSink,
        dispatch_cancel: threading.Event,
    ) -> None:
        current_task = asyncio.current_task()
        if current_task is not None:
            self.state.start_dispatch(task=current_task, cancel_event=dispatch_cancel)
        else:
            self.state.attach_cancel_event(dispatch_cancel)

        await emit(AgentEvent(type="turn_start", text=text))
        try:
            await self._execute_agent_turn(
                text=text,
                output=output,
                recorder=recorder,
                confirm=confirm,
            )
        except asyncio.CancelledError:
            await emit(AgentEvent(type="turn_interrupted"))
            raise
        except DispatchCancelled:
            await emit(AgentEvent(type="turn_interrupted"))
        except Exception as exc:
            report_exception(exc, context="interactive_shell.turn")
            await emit(AgentEvent(type="turn_error", error=exc))
        finally:
            self.state.finish_dispatch(dispatch_cancel)
            await emit(AgentEvent(type="turn_end"))

    async def _execute_agent_turn(
        self,
        *,
        text: str,
        output: StreamingConsole,
        recorder: PromptRecorder | None,
        confirm: Callable[[str], str],
    ) -> None:
        with _bound_cli_session(self.session.session_id):
            await asyncio.to_thread(
                handle_message_with_agent,
                text,
                self.session,
                output,
                recorder=recorder,
                confirm_fn=confirm,
                is_tty=None,
            )


async def run_input_loop(
    *,
    state: ReplState,
    session: ReplSession,
    background: BackgroundTaskManager | None,
    input_reader: PromptInputReader,
    echo_console: Console,
    handle_input_action: Callable[[InputAction], Awaitable[bool]],
) -> None:
    """Read input events and dispatch them until exit or close is requested."""
    while not state.exit_requested:
        if background is not None:
            background.drain_turn_start_output(echo_console)
        event = await input_reader.read()
        action = decide_input_action(
            event,
            ShellInputSnapshot(
                exit_requested=state.exit_requested,
                dispatch_running=state.is_dispatch_running(),
                awaiting_confirmation=state.is_awaiting_confirmation(),
            ),
            needs_exclusive_stdin=lambda text: turn_needs_exclusive_stdin(
                text,
                session,
            ),
        )
        should_continue = await handle_input_action(action)
        if not should_continue:
            return


async def run_agent_turn_queue(
    *,
    state: ReplState,
    run_turn: Callable[[str], Coroutine[Any, Any, None]],
) -> None:
    """Consume queued turns and run each one until exit."""
    while not state.exit_requested:
        try:
            text = await state.queue.get()
        except asyncio.CancelledError:
            return
        if state.exit_requested:
            state.queue.task_done()
            return

        turn_task = asyncio.create_task(run_turn(text))
        state.attach_turn_task(turn_task)
        try:
            await turn_task
        except asyncio.CancelledError:
            log.debug("Queued turn task was cancelled")
        except Exception as exc:
            log.debug("Queued turn task ended with exception: %s", exc)
        finally:
            state.clear_current_task()
            state.queue.task_done()


def request_confirmation_via_prompt(state: ReplState, prompt_text: str) -> str:
    response_event = threading.Event()
    state.begin_confirmation(response_event, prompt_text)
    try:
        while not response_event.is_set():
            cancel = state.current_cancel_event
            if cancel is not None and cancel.is_set():
                raise DispatchCancelled("cancelled while awaiting confirmation")
            response_event.wait(timeout=PROMPT_REFRESH_INTERVAL_S)
        if not state.confirm_response:
            raise DispatchCancelled("cancelled while awaiting confirmation")
        return state.confirm_response[0]
    finally:
        state.clear_confirmation()


__all__ = [
    "AgentEvent",
    "AgentEventSink",
    "AgentTurnRunner",
    "DispatchCancelled",
    "handle_message_with_agent",
    "request_confirmation_via_prompt",
    "run_agent_turn_queue",
    "run_input_loop",
]
