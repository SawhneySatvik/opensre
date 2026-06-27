"""Interactive shell runtime controller, action execution, and turn handling."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
from collections.abc import Awaitable, Callable, Coroutine, Iterator
from dataclasses import dataclass
from typing import Any, Literal

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.markup import escape

from config.llm_reasoning_effort import apply_reasoning_effort
from core.domain.alerts import inbox as _alert_inbox
from core.runtime.agent import Agent
from core.runtime.llm.agent_llm_client import AgentLLMResponse, ToolCall
from integrations.llm_cli.failure_explain import is_context_length_overflow
from interactive_shell.chat.cli_agent import answer_cli_agent
from interactive_shell.chat.tool_gathering import gather_tool_evidence
from interactive_shell.harness.llm_context import (
    build_action_system_prompt,
    build_action_user_message,
)
from interactive_shell.harness.state.conversation_history import MAX_CONVERSATION_MESSAGES
from interactive_shell.runtime.background.workers import BackgroundTaskManager
from interactive_shell.runtime.core.prompt_manager import PromptManager
from interactive_shell.runtime.core.state import (
    PROMPT_REFRESH_INTERVAL_S,
    ReplState,
    SpinnerState,
)
from interactive_shell.runtime.input import (
    PromptInputReader,
)
from interactive_shell.runtime.input.actions import (
    CancelTurn,
    CloseShell,
    DeliverConfirmation,
    IgnoreInput,
    InputAction,
    ShellInputSnapshot,
    SubmitTurn,
    decide_input_action,
)
from interactive_shell.runtime.utils.input_policy import (
    turn_needs_exclusive_stdin,
    turn_should_show_spinner,
)
from interactive_shell.session import (
    ReplRuntimeContext,
    ReplSession,
    create_repl_runtime_context,
)
from interactive_shell.tools.tool_contracts import ToolContext
from interactive_shell.tools.tool_registry import REGISTRY
from interactive_shell.ui import ERROR, WARNING
from interactive_shell.ui.action_rendering import ActionRenderObserver
from interactive_shell.ui.components.cpr_stdin import drain_stale_cpr_bytes
from interactive_shell.ui.output.repl_progress import repl_safe_progress_scope
from interactive_shell.ui.streaming import render_response_header
from interactive_shell.ui.streaming.console import StreamingConsole
from interactive_shell.utils.error_handling.exception_reporting import report_exception
from interactive_shell.utils.telemetry import LlmRunInfo, PromptRecorder
from platform.analytics.cli import capture_terminal_turn_summarized
from platform.analytics.repl_context import bind_cli_session_id, reset_cli_session_id

log = logging.getLogger(__name__)

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
            log.debug(
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


_AGENT_TURN_KIND = "agent"

ExecuteActions = Callable[..., TerminalActionExecutionResult]
GatherEvidence = Callable[..., str | None]
AnswerAgent = Callable[..., LlmRunInfo | None]


@dataclass(frozen=True)
class ShellTurnResult:
    final_intent: str
    action_result: TerminalActionExecutionResult
    assistant_response_text: str = ""
    llm_run: LlmRunInfo | None = None

    @property
    def answered(self) -> bool:
        """A turn is "answered" exactly when the conversational LLM produced a run."""
        return self.llm_run is not None


def _response_text(run: LlmRunInfo | None) -> str:
    return run.response_text if run is not None and run.response_text else ""


def _record_terminal_turn_telemetry(
    action_result: TerminalActionExecutionResult,
    session: ReplSession,
) -> None:
    """Update session terminal aggregates and emit the turn-summary analytics event."""
    fallback_to_llm = not action_result.handled
    snapshot = session.record_terminal_turn(
        executed_count=action_result.executed_count,
        executed_success_count=action_result.executed_success_count,
        fallback_to_llm=fallback_to_llm,
    )
    capture_terminal_turn_summarized(
        planned_count=action_result.planned_count,
        executed_count=action_result.executed_count,
        executed_success_count=action_result.executed_success_count,
        fallback_to_llm=fallback_to_llm,
        session_turn_index=snapshot.turn_index,
        session_fallback_count=snapshot.fallback_count,
        session_action_success_percent=snapshot.action_success_percent,
        session_fallback_rate_percent=snapshot.fallback_rate_percent,
    )


def _finalize_turn(
    session: ReplSession,
    recorder: PromptRecorder | None,
    text: str,
    result: ShellTurnResult,
) -> ShellTurnResult:
    """Flush the recorder, persist the turn, and stamp the session intent in one place."""
    if recorder is not None:
        recorder.set_response(result.assistant_response_text, result.llm_run)
        recorder.flush()
    if result.llm_run is not None:
        session.record("cli_agent", text)
    session.last_assistant_intent = result.final_intent
    return result


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
    """Run one interactive-shell turn through three paths, in order:

    1. ``answer_from_observation`` — a successful action left discovery output, so
       summarize it into a direct answer.
    2. ``action_handled`` — the action fully handled the turn; stop without the LLM.
    3. ``gather_and_answer`` — nothing was handled; gather evidence and answer.
    """
    execute_actions = execute_actions or execute_cli_actions
    gather_evidence = gather_evidence or gather_tool_evidence
    answer_agent = answer_agent or answer_cli_agent

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
    _record_terminal_turn_telemetry(action_result, session)

    observation = session.last_command_observation

    # Path 1: a successful terminal action left discovery output worth summarizing.
    if (
        action_result.handled
        and observation is not None
        and action_result.executed_success_count > 0
    ):
        with apply_reasoning_effort(session.reasoning_effort):
            run = answer_agent(
                text,
                session,
                console,
                confirm_fn=confirm_fn,
                is_tty=is_tty,
                tool_observation=observation,
            )
        return _finalize_turn(
            session,
            recorder,
            text,
            ShellTurnResult(
                final_intent="cli_agent_summarized",
                action_result=action_result,
                assistant_response_text=_response_text(run),
                llm_run=run,
            ),
        )

    if action_result.handled:
        return _finalize_turn(
            session,
            recorder,
            text,
            ShellTurnResult(
                final_intent="cli_agent_handled",
                action_result=action_result,
                assistant_response_text=action_result.response_text,
            ),
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
    return _finalize_turn(
        session,
        recorder,
        text,
        ShellTurnResult(
            final_intent="cli_agent_fallback",
            action_result=action_result,
            assistant_response_text=_response_text(run),
            llm_run=run,
        ),
    )


@dataclass(frozen=True)
class AgentEvent:
    """Agent lifecycle event emitted during one submitted shell turn."""

    type: Literal["turn_start", "turn_interrupted", "turn_error", "turn_end"]
    text: str | None = None
    error: Exception | None = None


AgentEventSink = Callable[[AgentEvent], Awaitable[None]]


class DispatchCancelled(Exception):
    """Raised when in-flight dispatch is cancelled during confirmation."""


def _resolve_runtime_context(
    session: ReplSession | ReplRuntimeContext | None,
    *,
    state: ReplState | None,
    spinner: SpinnerState | None,
    pt_session: PromptSession[str] | None,
    inbox: _alert_inbox.AlertInbox | None,
) -> ReplRuntimeContext:
    if isinstance(session, ReplRuntimeContext):
        if state is None and spinner is None and pt_session is None and inbox is None:
            return session
        return ReplRuntimeContext(
            session=session.session,
            state=state or session.state,
            spinner=spinner or session.spinner,
            pt_session=pt_session if pt_session is not None else session.pt_session,
            inbox=inbox if inbox is not None else session.inbox,
        )
    return create_repl_runtime_context(
        session,
        state=state,
        spinner=spinner,
        pt_session=pt_session,
        inbox=inbox,
    )


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
            self.state.current_cancel_event = dispatch_cancel

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
        state.current_task = turn_task
        try:
            await turn_task
        except asyncio.CancelledError:
            log.debug("Queued turn task was cancelled")
        except Exception as exc:
            log.debug("Queued turn task ended with exception: %s", exc)
        finally:
            state.clear_current_task()
            state.queue.task_done()


class InteractiveShellController:
    """Coordinate prompt input, queued dispatch, background workers, and shutdown."""

    def __init__(
        self,
        session: ReplSession | ReplRuntimeContext | None = None,
        *,
        state: ReplState | None = None,
        spinner: SpinnerState | None = None,
        pt_session: PromptSession[str] | None = None,
        inbox: _alert_inbox.AlertInbox | None = None,
    ) -> None:
        self.runtime_context = _resolve_runtime_context(
            session,
            state=state,
            spinner=spinner,
            pt_session=pt_session,
            inbox=inbox,
        )
        self.session = self.runtime_context.session
        self.inbox = self.runtime_context.inbox
        self.state = self.runtime_context.state
        self.spinner = self.runtime_context.spinner
        self.prompt = PromptManager(
            self.session,
            self.state,
            self.spinner,
            self.runtime_context.pt_session,
        )
        self.turn_runner = AgentTurnRunner(
            session=self.session,
            state=self.state,
            spinner=self.spinner,
            invalidate_prompt=self.prompt.invalidate_prompt,
        )
        self.echo_console = Console(highlight=False, force_terminal=True, color_system="truecolor")
        self.input_reader = PromptInputReader(
            self.prompt,
            self.state,
            self.session,
            self.echo_console,
        )
        self.background: BackgroundTaskManager | None = None
        self.tasks: list[tuple[str, asyncio.Task[None]]] = []

    async def start_interactive_shell(self) -> None:
        self.session.schedule_warm_resolved_integrations()
        self._start_runtime_services()
        try:
            with patch_stdout(raw=True):
                await run_input_loop(
                    state=self.state,
                    session=self.session,
                    background=self.background,
                    input_reader=self.input_reader,
                    echo_console=self.echo_console,
                    handle_input_action=self._handle_input_action,
                )
        finally:
            await self._shutdown_runtime()

    def _start_runtime_services(self) -> None:
        self.prompt.setup()
        self.background = BackgroundTaskManager(
            self.session,
            self.state,
            self.spinner,
            self.inbox,
            self.prompt.invalidate_prompt,
        )
        self.tasks = self.background.start_all(
            lambda: run_agent_turn_queue(
                state=self.state,
                run_turn=self.turn_runner.run_agent_turn,
            )
        )

    async def _handle_input_action(self, action: InputAction) -> bool:
        match action:
            case IgnoreInput():
                return True
            case CloseShell():
                return False
            case CancelTurn(submitted_text=text):
                if text:
                    self.prompt.render_submitted_prompt(self.echo_console, text)
                self.state.cancel_current_dispatch()
                return True
            case DeliverConfirmation(text=text):
                self.state.deliver_confirmation(text)
                return True
            case SubmitTurn(text=text, wait_until_idle=wait, warning=warning):
                if warning:
                    self.echo_console.print(warning)
                self.prompt.render_submitted_prompt(self.echo_console, text)
                await self.state.queue.put(text)
                if wait:
                    await self.state.queue.join()
                return True
        raise AssertionError(f"Unhandled input action: {action!r}")

    async def _shutdown_runtime(self) -> None:
        self.state.request_exit()
        self.state.cancel_current_dispatch()

        for _label, task in self.tasks:
            task.cancel()

        shutdown_results = await asyncio.gather(
            *(task for _label, task in self.tasks),
            return_exceptions=True,
        )
        for (label, _task), result in zip(self.tasks, shutdown_results, strict=True):
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                log.debug("%s task shutdown raised exception: %s", label, result)


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
    "ActionExecutionDeps",
    "AgentEvent",
    "AgentEventSink",
    "AgentTurnRunner",
    "DispatchCancelled",
    "InteractiveShellController",
    "ShellTurnResult",
    "TerminalActionExecutionResult",
    "execute_cli_actions",
    "handle_message_with_agent",
    "request_confirmation_via_prompt",
    "run_agent_turn_queue",
    "run_input_loop",
]
