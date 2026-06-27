"""Runtime host for interactive OpenSRE shell turns."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Awaitable, Callable, Coroutine, Iterator
from concurrent.futures import Future
from typing import Any

from rich.console import Console

from interactive_shell.harness.agent import ShellTurnAgent
from interactive_shell.harness.events import (
    AgentEvent,
    AgentEventSink,
    AgentEventType,
    AsyncAgentEventSink,
)
from interactive_shell.harness.llm_context.session import ReplSession
from interactive_shell.runtime.agent_presentation import ConsoleAgentEventSink
from interactive_shell.runtime.background.workers import BackgroundTaskManager
from interactive_shell.runtime.core.confirmation import (
    DispatchCancelled,
    request_confirmation_via_prompt,
)
from interactive_shell.runtime.core.state import ReplState, SpinnerState
from interactive_shell.runtime.input import PromptInputReader
from interactive_shell.runtime.input.actions import (
    InputAction,
    ShellInputSnapshot,
    decide_input_action,
)
from interactive_shell.runtime.utils.input_policy import turn_needs_exclusive_stdin
from interactive_shell.ui.output.repl_progress import repl_safe_progress_scope
from interactive_shell.ui.streaming.console import StreamingConsole
from interactive_shell.utils.error_handling.exception_reporting import report_exception
from interactive_shell.utils.telemetry import PromptRecorder
from platform.analytics.repl_context import bind_cli_session_id, reset_cli_session_id

_logger = logging.getLogger(__name__)
_AGENT_TURN_KIND = "agent"
_ONCE_EVENTS: frozenset[AgentEventType] = frozenset(
    {"turn_start", "turn_interrupted", "turn_error", "turn_end"}
)


@contextlib.contextmanager
def _bound_cli_session(session_id: str) -> Iterator[None]:
    """Temporarily bind the CLI session ID for the current turn."""
    token = bind_cli_session_id(session_id)
    try:
        yield
    finally:
        reset_cli_session_id(token)


class _ThreadedAgentEventBridge:
    """Bridge sync shell-agent events to the async terminal presentation sink."""

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        sink: AsyncAgentEventSink,
    ) -> None:
        self._loop = loop
        self._sink = sink
        self._lock = threading.Lock()
        self._emitted: set[AgentEventType] = set()

    def __call__(self, event: AgentEvent) -> None:
        if not self._claim(event):
            return
        future: Future[None] = asyncio.run_coroutine_threadsafe(self._sink(event), self._loop)
        future.result()

    async def emit_async(self, event: AgentEvent) -> None:
        if not self._claim(event):
            return
        await self._sink(event)

    def _claim(self, event: AgentEvent) -> bool:
        if event.type not in _ONCE_EVENTS:
            return True
        with self._lock:
            if event.type in self._emitted:
                return False
            self._emitted.add(event.type)
            return True


def _setup_turn_presentation(
    runner: ShellTurnHost, user_input: str
) -> tuple[StreamingConsole, AsyncAgentEventSink, PromptRecorder | None, threading.Event]:
    """Create console, event sink, recorder, and cancellation primitive for a turn."""
    cancel_event = threading.Event()

    console = StreamingConsole(
        runner.spinner,
        cancel_event,
        prompt_invalidator=runner.invalidate_prompt,
        highlight=False,
        force_terminal=True,
        color_system="truecolor",
        legacy_windows=False,
    )

    event_sink = ConsoleAgentEventSink(
        session=runner.session,
        spinner=runner.spinner,
        console=console,
    )

    recorder = PromptRecorder.start(
        session=runner.session,
        text=user_input,
        turn_kind=_AGENT_TURN_KIND,
    )

    return console, event_sink, recorder, cancel_event


class ShellTurnHost:
    """Terminal/runtime host for a stateful shell turn agent."""

    def __init__(
        self,
        *,
        session: ReplSession,
        state: ReplState,
        spinner: SpinnerState,
        invalidate_prompt: Callable[[], None],
        agent: ShellTurnAgent | None = None,
    ) -> None:
        self.session = session
        self.state = state
        self.spinner = spinner
        self.invalidate_prompt = invalidate_prompt
        self.agent = agent or ShellTurnAgent(session)

    async def run_turn(self, user_input: str) -> None:
        """Execute a complete agent turn with presentation and runtime state."""
        console, event_sink, recorder, cancel_event = _setup_turn_presentation(self, user_input)

        progress_scope = (
            contextlib.nullcontext()
            if turn_needs_exclusive_stdin(user_input, self.session)
            else repl_safe_progress_scope()
        )

        with progress_scope:
            await self._execute_turn_lifecycle(
                user_input=user_input,
                console=console,
                recorder=recorder,
                event_sink=event_sink,
                cancel_event=cancel_event,
            )

    async def _execute_turn_lifecycle(
        self,
        user_input: str,
        console: StreamingConsole,
        recorder: PromptRecorder | None,
        event_sink: AsyncAgentEventSink,
        cancel_event: threading.Event,
    ) -> None:
        """Manage dispatch tracking around the shell-owned turn agent."""
        task = asyncio.current_task()
        if task is not None:
            self.state.start_dispatch(task=task, cancel_event=cancel_event)
        else:
            self.state.attach_cancel_event(cancel_event)

        loop = asyncio.get_running_loop()
        event_bridge = _ThreadedAgentEventBridge(loop=loop, sink=event_sink)
        unsubscribe = self.agent.subscribe(event_bridge)

        try:
            await self._run_agent_turn(user_input, console, recorder)
        except asyncio.CancelledError:
            await event_bridge.emit_async(AgentEvent(type="turn_interrupted"))
            raise
        except DispatchCancelled:
            await event_bridge.emit_async(AgentEvent(type="turn_interrupted"))
        except Exception as exc:
            report_exception(exc, context="interactive_shell.turn")
            await event_bridge.emit_async(AgentEvent(type="turn_error", error=exc))
        finally:
            self.state.finish_dispatch(cancel_event)
            await event_bridge.emit_async(AgentEvent(type="turn_end"))
            unsubscribe()

    async def _run_agent_turn(
        self, user_input: str, output: StreamingConsole, recorder: PromptRecorder | None
    ) -> None:
        """Execute the shell agent in a thread with proper session context."""

        def confirm_fn(prompt: str) -> str:
            return request_confirmation_via_prompt(self.state, prompt)

        with _bound_cli_session(self.session.session_id):
            await asyncio.to_thread(
                self.agent.run_turn,
                user_input,
                console=output,
                recorder=recorder,
                confirm_fn=confirm_fn,
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
    """Continuously read and process user input events until exit."""
    while not state.exit_requested:
        if background:
            background.drain_turn_start_output(echo_console)

        event = await input_reader.read()

        action = decide_input_action(
            event,
            ShellInputSnapshot(
                exit_requested=state.exit_requested,
                dispatch_running=state.is_dispatch_running(),
                awaiting_confirmation=state.is_awaiting_confirmation(),
            ),
            needs_exclusive_stdin=lambda text: turn_needs_exclusive_stdin(text, session),
        )

        if not await handle_input_action(action):
            return


async def run_agent_turn_queue(
    *,
    state: ReplState,
    run_turn: Callable[[str], Coroutine[Any, Any, None]],
) -> None:
    """Process turns from the queue until the REPL is shutting down."""
    while not state.exit_requested:
        try:
            user_input = await state.queue.get()
        except asyncio.CancelledError:
            return

        if state.exit_requested:
            state.queue.task_done()
            return

        turn_task = asyncio.create_task(run_turn(user_input))
        state.attach_turn_task(turn_task)

        try:
            await turn_task
        except asyncio.CancelledError:
            _logger.debug("Queued agent turn was cancelled")
        except Exception as exc:
            _logger.debug("Queued agent turn failed: %s", exc)
        finally:
            state.clear_current_task()
            state.queue.task_done()


__all__ = [
    "AgentEvent",
    "AgentEventSink",
    "ShellTurnHost",
    "run_agent_turn_queue",
    "run_input_loop",
]
