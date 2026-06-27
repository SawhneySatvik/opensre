"""Runtime driver for interactive OpenSRE shell turns.

This module is the stateful driver around one shell turn: it wires terminal
presentation, drives the turn lifecycle and cancellation, and consumes the
queued-turn and input-event loops.

The turn's actual work is delegated:

- turn routing and the three answer paths live in
  :mod:`interactive_shell.harness.turn`,
- final response generation in :mod:`interactive_shell.harness.response`,
- action-plan parsing/execution in
  :mod:`interactive_shell.harness.action_plan` and
  :mod:`interactive_shell.harness.action_exec`,
- prompt construction in ``harness/llm_context/assistant_prompt.py``,
- terminal presentation in ``runtime/agent_presentation.py``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Awaitable, Callable, Coroutine, Iterator
from typing import Any

from rich.console import Console

from interactive_shell.harness.turn import handle_message_with_agent
from interactive_shell.runtime import ReplSession
from interactive_shell.runtime.agent_presentation import (
    AgentEvent,
    AgentEventSink,
    ConsoleAgentEventSink,
)
from interactive_shell.runtime.background.workers import BackgroundTaskManager
from interactive_shell.runtime.core.confirmation import (
    DispatchCancelled,
    request_confirmation_via_prompt,
)
from interactive_shell.runtime.core.state import (
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
)
from interactive_shell.ui.output.repl_progress import repl_safe_progress_scope
from interactive_shell.ui.streaming.console import StreamingConsole
from interactive_shell.utils.error_handling.exception_reporting import report_exception
from interactive_shell.utils.telemetry import PromptRecorder
from platform.analytics.repl_context import bind_cli_session_id, reset_cli_session_id

_logger = logging.getLogger(__name__)

_AGENT_TURN_KIND = "agent"


@contextlib.contextmanager
def _bound_cli_session(session_id: str) -> Iterator[None]:
    token = bind_cli_session_id(session_id)
    try:
        yield
    finally:
        reset_cli_session_id(token)


class AgentTurnRunner:
    """Drives one submitted shell turn: presentation setup and lifecycle."""

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
            await _run_agent_turn_loop(
                session=self.session,
                state=self.state,
                text=text,
                output=console,
                recorder=recorder,
                confirm=lambda prompt: request_confirmation_via_prompt(self.state, prompt),
                emit=emit,
                dispatch_cancel=dispatch_cancel,
            )


async def _run_agent_turn_loop(
    *,
    session: ReplSession,
    state: ReplState,
    text: str,
    output: StreamingConsole,
    recorder: PromptRecorder | None,
    confirm: Callable[[str], str],
    emit: AgentEventSink,
    dispatch_cancel: threading.Event,
) -> None:
    current_task = asyncio.current_task()
    if current_task is not None:
        state.start_dispatch(task=current_task, cancel_event=dispatch_cancel)
    else:
        state.attach_cancel_event(dispatch_cancel)

    await emit(AgentEvent(type="turn_start", text=text))
    try:
        await _execute_agent_turn(
            session=session,
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
        state.finish_dispatch(dispatch_cancel)
        await emit(AgentEvent(type="turn_end"))


async def _execute_agent_turn(
    *,
    session: ReplSession,
    text: str,
    output: StreamingConsole,
    recorder: PromptRecorder | None,
    confirm: Callable[[str], str],
) -> None:
    with _bound_cli_session(session.session_id):
        await asyncio.to_thread(
            handle_message_with_agent,
            text,
            session,
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
            _logger.debug("Queued turn task was cancelled")
        except Exception as exc:
            _logger.debug("Queued turn task ended with exception: %s", exc)
        finally:
            state.clear_current_task()
            state.queue.task_done()


__all__ = [
    "AgentEvent",
    "AgentEventSink",
    "AgentTurnRunner",
    "run_agent_turn_queue",
    "run_input_loop",
]
