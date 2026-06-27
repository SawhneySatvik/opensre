"""Terminal assistant and turn handling for the interactive OpenSRE shell.

This module owns:

- the conversational assistant turn (``answer_cli_agent``), grounding, and the
  JSON action-plan parsing / capability validation / execution path;
- the interactive-shell turn dispatch (``handle_message_with_agent``), the
  per-turn agent lifecycle (``AgentTurnRunner``, ``ConsoleAgentEventSink``), and
  the input / queue loops that drive submitted turns.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
from collections.abc import Awaitable, Callable, Coroutine, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape

from config.llm_reasoning_effort import apply_reasoning_effort
from integrations.llm_cli.errors import CLITimeoutError
from interactive_shell.harness.llm_context.assistant_system_prompt import (
    _build_environment_block,
    _build_observation_block,
    _build_system_prompt,
)
from interactive_shell.harness.llm_context.grounding.agents_md_reference import (
    build_agents_md_reference_text,
)
from interactive_shell.harness.llm_context.grounding.cli_reference import (
    build_cli_reference_text,
)
from interactive_shell.harness.llm_context.grounding.grounding_diagnostics import (
    log_grounding_cache_diagnostics,
)
from interactive_shell.harness.llm_context.grounding.investigation_flow_reference import (
    build_investigation_flow_reference_text,
)
from interactive_shell.harness.state.conversation_history import (
    MAX_CONVERSATION_MESSAGES,
    format_recent_conversation,
)
from interactive_shell.harness.tool_calling import run_tool_calling_turn
from interactive_shell.harness.turn_context import TurnContext
from interactive_shell.runtime import ReplSession
from interactive_shell.runtime.background.workers import BackgroundTaskManager
from interactive_shell.runtime.core.state import (
    PROMPT_REFRESH_INTERVAL_S,
    ReplState,
    SpinnerState,
)
from interactive_shell.runtime.core.token_accounting import build_llm_run_info
from interactive_shell.runtime.core.turn_accounting import (
    ShellTurnAccounting,
    ShellTurnResult,
    ToolCallingTurnResult,
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
from interactive_shell.session import SUGGESTED_PROMPT_AFTER_FAILED_SYNTHETIC_TEST
from interactive_shell.tools.tool_gathering import gather_tool_evidence
from interactive_shell.ui import (
    BOLD_BRAND,
    DIM,
    ERROR,
    MARKDOWN_THEME,
    STREAM_LABEL_ASSISTANT,
    WARNING,
    stream_to_console,
)
from interactive_shell.ui.components.cpr_stdin import drain_stale_cpr_bytes
from interactive_shell.ui.output.repl_progress import repl_safe_progress_scope
from interactive_shell.ui.streaming.console import StreamingConsole
from interactive_shell.utils.error_handling.exception_reporting import report_exception
from interactive_shell.utils.telemetry import LlmRunInfo, PromptRecorder
from platform.analytics.repl_context import bind_cli_session_id, reset_cli_session_id

_logger = logging.getLogger(__name__)

_MAX_SYNTHETIC_OBSERVATION_PROMPT_CHARS = 120_000

_AGENT_TURN_KIND = "agent"

RunToolCallingTurn = Callable[..., ToolCallingTurnResult]
GatherEvidence = Callable[..., str | None]
AnswerAgent = Callable[..., LlmRunInfo | None]


# ---------------------------------------------------------------------------
# Action plan parsing and capability validation
# ---------------------------------------------------------------------------


# `run_interactive` is not a narrow feature allowlist. It is the bridge from an
# agent-planned action back into the OpenSRE interactive shell. Any command that
# is registered in the slash-command registry is already an OpenSRE command and
# must stay eligible here.
#
# Keep this registry-backed instead of listing subcommands like
# `/integrations setup` or `/integrations remove`: duplicating subcommand lists
# here drifts from the actual dispatcher and causes valid OpenSRE commands to be
# rejected before the normal policy/confirmation flow can evaluate them. The
# dispatcher remains the source of truth for argument validation, execution tier,
# confirmation, exclusive-stdin handling, and the command's side effects.
#
# The only thing this gate should reject is non-OpenSRE input: empty strings,
# shell snippets, arbitrary text, or unknown slash commands. Do not reintroduce
# a per-command allowlist in this file.
def _registered_interactive_command(command: str) -> bool:
    parts = command.strip().split()
    if not parts:
        return False
    name = parts[0].lower()
    if name == "/":
        return True
    if not name.startswith("/"):
        return False

    from interactive_shell.command_registry import SLASH_COMMANDS

    return name in SLASH_COMMANDS


_ALLOWED_SLASH_ACTIONS = frozenset(
    {
        "/model show",
        "/health",
        "/doctor",
        "/version",
    }
)

# Conversational action kinds map onto the same capability gates the action
# planner uses, so a session that explicitly disables a surface cannot actuate
# it from the chat answer path either.
_ACTION_CAPABILITY: dict[str, str] = {
    "switch_llm_provider": "llm_provider",
    "switch_toolcall_model": "llm_provider",
    "slash": "slash_commands",
    "run_interactive": "slash_commands",
    "run_cli_command": "cli_commands",
}


def _as_text(value: object) -> str:
    return str(value or "").strip()


@dataclass(frozen=True)
class ActionPlanAction:
    """Typed representation of a single action emitted by the CLI agent."""

    kind: str
    provider: str = ""
    model: str = ""
    toolcall_model: str = ""
    command: str = ""
    args: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> ActionPlanAction | None:
        kind = _as_text(payload.get("action"))

        if not kind and _as_text(payload.get("provider")):
            kind = "switch_llm_provider"

        if not kind and _as_text(payload.get("command")):
            kind = "slash"

        if not kind:
            return None

        return cls(
            kind=kind,
            provider=_as_text(payload.get("provider")),
            model=_as_text(payload.get("model")),
            toolcall_model=_as_text(payload.get("toolcall_model")),
            command=_as_text(payload.get("command")),
            args=_as_text(payload.get("args")),
        )

    @property
    def capability(self) -> str | None:
        return _ACTION_CAPABILITY.get(self.kind)

    @property
    def label(self) -> str:
        if self.kind == "switch_llm_provider":
            text = f"switch LLM provider to {self.provider}"
            if self.model:
                text += f" ({self.model})"
            if self.toolcall_model:
                text += f" + toolcall {self.toolcall_model}"
            return text

        if self.kind == "switch_toolcall_model":
            return (
                f"switch toolcall model to {self.model}" if self.model else "switch toolcall model"
            )

        if self.kind == "slash":
            return self.command

        if self.kind == "run_cli_command":
            return f"opensre {self.args}" if self.args else "opensre"

        if self.kind == "run_interactive":
            return self.command or "interactive command"

        return f"unsupported action: {self.kind or '?'}"


def _actions_allowed_by_capabilities(
    actions: list[ActionPlanAction], session: ReplSession
) -> list[ActionPlanAction]:
    """Drop actions whose capability surface is explicitly disabled for *session*."""
    from interactive_shell.tools.tool_contracts import (
        capability_not_explicitly_disabled,
    )

    return [
        action
        for action in actions
        if action.capability is None
        or capability_not_explicitly_disabled(session, action.capability)
    ]


def _opensre_integration_command_blocked(payload: str, session: ReplSession) -> bool:
    """Block integration-management CLI runs when the session has none configured."""
    if not session.configured_integrations_known or session.configured_integrations:
        return False
    lowered = payload.strip().lower()
    return lowered.startswith("integrations") or "integration" in lowered


def _extract_json_object(text: str) -> dict[str, object] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _normalize_action(action: dict[str, object]) -> ActionPlanAction | None:
    return ActionPlanAction.from_payload(action)


def _parse_action_plan(text: str) -> list[ActionPlanAction]:
    payload = _extract_json_object(text)
    if payload is None:
        return []

    actions = payload.get("actions")
    if not isinstance(actions, list):
        normalized = _normalize_action(payload)
        return [normalized] if normalized is not None else []

    return [
        normalized
        for action in actions
        if isinstance(action, dict)
        for normalized in [_normalize_action(action)]
        if normalized is not None
    ]


def _execute_action_plan(
    actions: list[ActionPlanAction],
    session: ReplSession,
    console: Console,
    *,
    confirm_fn: Callable[[str], str] | None = None,
    is_tty: bool | None = None,
) -> bool:
    return _ActionPlanExecutor(
        session=session,
        console=console,
        confirm_fn=confirm_fn,
        is_tty=is_tty,
    ).execute(actions)


class _ActionPlanExecutor:
    def __init__(
        self,
        *,
        session: ReplSession,
        console: Console,
        confirm_fn: Callable[[str], str] | None,
        is_tty: bool | None,
    ) -> None:
        self.session = session
        self.console = console
        self.confirm_fn = confirm_fn
        self.is_tty = is_tty

    def execute(self, actions: list[ActionPlanAction]) -> bool:
        if not actions:
            return False

        actions = _actions_allowed_by_capabilities(actions, self.session)
        if not actions:
            return False

        self._print_requested_actions(actions)

        for action in actions:
            self.console.print()
            self._execute_one(action)

        self.console.print()
        return True

    def _execute_one(self, action: ActionPlanAction) -> None:
        handlers: dict[str, Callable[[ActionPlanAction], None]] = {
            "switch_llm_provider": self._switch_llm_provider,
            "switch_toolcall_model": self._switch_toolcall_model,
            "slash": self._slash,
            "run_cli_command": self._run_cli_command,
            "run_interactive": self._run_interactive,
        }

        handler = handlers.get(action.kind)
        if handler is None:
            self._error(f"unsupported action: {action.kind or '?'}")
            return

        handler(action)

    def _print_requested_actions(self, actions: list[ActionPlanAction]) -> None:
        self.console.print()
        self.console.print(f"[{BOLD_BRAND}]{STREAM_LABEL_ASSISTANT}:[/]")
        self.console.print(f"[{DIM}]Requested actions:[/]")

        for index, action in enumerate(actions, start=1):
            self.console.print(f"[{DIM}]{index}.[/] [{BOLD_BRAND}]{escape(action.label)}[/]")

        self.console.print()

    def _switch_llm_provider(self, action: ActionPlanAction) -> None:
        from interactive_shell.command_registry import switch_llm_provider
        from interactive_shell.tools.shared import allow_tool
        from interactive_shell.ui.execution_confirm import execution_allowed

        if not action.provider:
            self._error("missing provider for switch_llm_provider action")
            return

        slash_label = f"/model set {action.provider}"
        if action.model:
            slash_label += f" {action.model}"
        if action.toolcall_model:
            slash_label += f" --toolcall-model {action.toolcall_model}"

        policy = allow_tool("switch_llm_provider")
        if not execution_allowed(
            policy,
            session=self.session,
            console=self.console,
            action_summary=slash_label,
            confirm_fn=self.confirm_fn,
            is_tty=self.is_tty,
            action_already_listed=True,
        ):
            return

        self.console.print(f"[bold]$ {escape(slash_label)}[/bold]")
        switch_llm_provider(
            action.provider,
            self.console,
            model=action.model or None,
            toolcall_model=action.toolcall_model or None,
        )
        self.session.record("slash", slash_label)

    def _switch_toolcall_model(self, action: ActionPlanAction) -> None:
        from interactive_shell.command_registry import switch_toolcall_model
        from interactive_shell.tools.shared import allow_tool
        from interactive_shell.ui.execution_confirm import execution_allowed

        if not action.model:
            self._error("missing model for switch_toolcall_model action")
            return

        slash_label = f"/model toolcall set {action.model}"

        policy = allow_tool("switch_toolcall_model")
        if not execution_allowed(
            policy,
            session=self.session,
            console=self.console,
            action_summary=slash_label,
            confirm_fn=self.confirm_fn,
            is_tty=self.is_tty,
            action_already_listed=True,
        ):
            return

        self.console.print(f"[bold]$ {escape(slash_label)}[/bold]")
        switch_toolcall_model(action.model, self.console)
        self.session.record("slash", slash_label)

    def _slash(self, action: ActionPlanAction) -> None:
        from interactive_shell.command_registry import SLASH_COMMANDS, dispatch_slash
        from interactive_shell.tools.shared import allow_tool
        from interactive_shell.ui.execution_confirm import execution_allowed

        command = action.command
        if command not in _ALLOWED_SLASH_ACTIONS:
            self._error(f"unsupported action command: {command}")
            return

        stripped = command.strip()
        parts = stripped.split()
        name = parts[0].lower()
        cmd_slash = SLASH_COMMANDS.get(name)

        if cmd_slash is None:
            dispatch_slash(
                command,
                self.session,
                self.console,
                confirm_fn=self.confirm_fn,
                is_tty=self.is_tty,
            )
            return

        policy = allow_tool("slash")
        if not execution_allowed(
            policy,
            session=self.session,
            console=self.console,
            action_summary=stripped,
            confirm_fn=self.confirm_fn,
            is_tty=self.is_tty,
            action_already_listed=True,
        ):
            self.session.record("slash", stripped, ok=False)
            return

        self.console.print(f"[bold]$ {escape(command)}[/bold]")
        dispatch_slash(
            command,
            self.session,
            self.console,
            confirm_fn=self.confirm_fn,
            is_tty=self.is_tty,
            policy_precleared=True,
        )

    def _run_cli_command(self, action: ActionPlanAction) -> None:
        if not action.args:
            self._error("missing args for run_cli_command action")
            return

        if _opensre_integration_command_blocked(action.args, self.session):
            self.console.print(
                f"[{WARNING}]integration command blocked: no integrations are configured "
                "in this session.[/]"
            )
            return

        from interactive_shell.runtime.subprocess_runner import run_opensre_cli_command

        run_opensre_cli_command(
            action.args,
            self.session,
            self.console,
            confirm_fn=self.confirm_fn,
            is_tty=self.is_tty,
        )

    def _run_interactive(self, action: ActionPlanAction) -> None:
        command = action.command

        if not _registered_interactive_command(command):
            self._error(f"unsupported interactive command: {command}")
            return

        from interactive_shell.ui.components.choice_menu import repl_tty_interactive

        if not repl_tty_interactive():
            self.console.print(
                f"Run [bold]{escape(command)}[/bold] in the interactive shell to continue."
            )
            return

        self.console.print(f"[{DIM}]Launching[/] [{BOLD_BRAND}]{escape(command)}[/]…")
        self.session.queue_auto_command(command)

    def _error(self, message: str) -> None:
        self.console.print(f"[{ERROR}]{escape(message)}[/]")


# ---------------------------------------------------------------------------
# Grounding helpers
# ---------------------------------------------------------------------------


def _summarize_evidence(evidence: Any) -> list[str]:
    """Render a short evidence preview for the prior-investigation grounding block.

    ``AgentState.evidence`` is a ``dict[str, Any]`` keyed by evidence id, but
    we accept list/other shapes defensively so an unexpected value doesn't
    silently drop all grounding context.
    """
    if isinstance(evidence, dict):
        sample_keys = list(evidence)[:3]
        sample = {key: evidence[key] for key in sample_keys}
        return [
            f"Evidence items: {len(evidence)}",
            "Evidence keys: " + ", ".join(map(str, sample_keys)),
            "Sample evidence:\n" + json.dumps(sample, indent=2, default=str)[:1500],
        ]
    if isinstance(evidence, list):
        return [
            f"Evidence items: {len(evidence)}",
            "Sample evidence:\n" + json.dumps(evidence[:3], indent=2, default=str)[:1500],
        ]
    return [
        f"Evidence type: {type(evidence).__name__}",
        f"Evidence summary:\n{str(evidence)[:1500]}",
    ]


def _summarize_last_state(state: dict[str, Any]) -> str:
    """Produce a compact text summary of the previous investigation for grounding."""
    parts: list[str] = []
    alert_name = state.get("alert_name")
    if alert_name:
        parts.append(f"Alert: {alert_name}")
    root_cause = state.get("root_cause")
    if root_cause:
        parts.append(f"Root cause: {root_cause}")
    problem_md = state.get("problem_md") or ""
    if problem_md:
        parts.append(f"Problem summary:\n{problem_md[:2000]}")
    slack_message = state.get("slack_message") or ""
    if slack_message:
        parts.append(f"Report:\n{slack_message[:2000]}")
    evidence = state.get("evidence")
    if evidence:
        try:
            parts.extend(_summarize_evidence(evidence))
        except (TypeError, ValueError) as exc:
            # Serialization can fail on exotic evidence values; tell the LLM
            # the context was withheld rather than silently dropping it.
            _logger.warning("could not serialize evidence for grounding: %s", exc)
            parts.append("(evidence present but could not be serialized for grounding)")
    return "\n\n".join(parts) or "(no prior investigation details available)"


def _user_message_requests_synthetic_failure_explanation(message: str) -> bool:
    """True when the user is likely asking about a failed synthetic benchmark."""
    m = message.strip().lower()
    if not m:
        return False
    suggested = SUGGESTED_PROMPT_AFTER_FAILED_SYNTHETIC_TEST.lower().rstrip("?")
    if m.rstrip("?") == suggested:
        return True
    if "why" in m and "fail" in m:
        return True
    return "what went wrong" in m


def _load_synthetic_observation_text(
    path_str: str, *, max_chars: int = _MAX_SYNTHETIC_OBSERVATION_PROMPT_CHARS
) -> str:
    try:
        raw = Path(path_str).read_text(encoding="utf-8")
    except OSError:
        return ""
    if len(raw) > max_chars:
        return (
            raw[:max_chars]
            + f"\n… [truncated for prompt size; observation is {len(raw)} characters total]"
        )
    return raw


# ---------------------------------------------------------------------------
# CLI agent answer
# ---------------------------------------------------------------------------


def _load_reasoning_client(console: Console) -> Any | None:
    try:
        from core.runtime.llm.llm_client import get_llm_for_reasoning
    except Exception as exc:
        report_exception(exc, context="interactive_shell.cli_agent.import")
        console.print(f"[{ERROR}]LLM client unavailable:[/] {escape(str(exc))}")
        return None

    return get_llm_for_reasoning()


def _build_integration_guard(ctx: TurnContext) -> str:
    if not (ctx.configured_integrations_known and not ctx.configured_integrations):
        return ""

    return (
        "No integrations are configured in this session. You may still help the user "
        "configure one: when they ask to set up, connect, or add an integration, emit a "
        "run_interactive action for `/integrations setup <service>` (or `/mcp connect "
        "<server>`). Do NOT emit run_cli_command or slash actions to show/verify/remove "
        "integrations that are not configured; for those, answer with guidance only.\n\n"
    )


def _build_synthetic_failure_block(ctx: TurnContext) -> str:
    obs_path = ctx.last_synthetic_observation_path
    if not obs_path:
        return ""

    if not _user_message_requests_synthetic_failure_explanation(ctx.text):
        return ""

    obs_text = _load_synthetic_observation_text(obs_path)
    if not obs_text:
        return ""

    return (
        "The user is asking about a failed `opensre tests synthetic` run "
        "in this checkout. The JSON below is the saved observation "
        f"(scores, gates, stderr summary). Path: {obs_path}\n"
        "Use it to explain validation failures. Do not say nothing ran or "
        "that you lack context — the run completed and this file was written.\n\n"
        f"--- observation_json ---\n{obs_text}\n\n"
    )


def _build_cli_agent_prompt(
    *,
    message: str,
    session: ReplSession,
    tool_observation: str | None,
    tool_observation_on_screen: bool,
    turn_ctx: TurnContext,
) -> str:
    reference = build_cli_reference_text()
    agents_md = build_agents_md_reference_text()
    investigation_flow = build_investigation_flow_reference_text()
    log_grounding_cache_diagnostics("cli_agent_grounding")

    history = format_recent_conversation(list(turn_ctx.conversation_messages))
    prior_investigation = (
        _summarize_last_state(turn_ctx.last_state) if turn_ctx.last_state is not None else ""
    )

    system = _build_system_prompt(
        reference,
        history,
        agents_md=agents_md,
        investigation_flow=investigation_flow,
        prior_investigation=prior_investigation,
        environment=_build_environment_block(session),
    )

    return (
        f"{system}\n"
        f"{_build_integration_guard(turn_ctx)}"
        f"{_build_observation_block(tool_observation, on_screen=tool_observation_on_screen)}"
        f"{_build_synthetic_failure_block(turn_ctx)}"
        f"--- User message ---\n{message}"
    )


def _stream_cli_agent_response(
    *,
    client: Any,
    prompt: str,
    session: ReplSession,
    console: Console,
) -> LlmRunInfo | None:
    try:
        started = time.monotonic()
        text_str = stream_to_console(
            console,
            label=STREAM_LABEL_ASSISTANT,
            chunks=client.invoke_stream(prompt),
            suppress_if_starts_with="{",
        )
    except KeyboardInterrupt:
        console.print(f"[{DIM}]· cancelled[/]")
        return None
    except Exception as exc:
        report_exception(
            exc,
            context="interactive_shell.cli_agent.stream",
            expected=isinstance(exc, CLITimeoutError),
        )
        console.print(f"[{ERROR}]assistant failed:[/] {escape(str(exc))}")
        return None

    return build_llm_run_info(
        session=session,
        prompt=prompt,
        response_text=text_str,
        started=started,
        client=client,
    )


def _render_json_like_response(console: Console, text: str) -> None:
    if not text.lstrip().startswith("{") or not text.strip():
        return

    console.print()
    console.print(f"[{BOLD_BRAND}]{STREAM_LABEL_ASSISTANT}:[/]")
    with console.use_theme(MARKDOWN_THEME):
        console.print(Markdown(text, code_theme="ansi_dark"))
    console.print()


def _record_cli_agent_turn(session: ReplSession, message: str, assistant_text: str) -> None:
    session.cli_agent_messages.append(("user", message))
    session.cli_agent_messages.append(("assistant", assistant_text))
    if len(session.cli_agent_messages) > MAX_CONVERSATION_MESSAGES:
        session.cli_agent_messages[:] = session.cli_agent_messages[-MAX_CONVERSATION_MESSAGES:]


def answer_cli_agent(
    message: str,
    session: ReplSession,
    console: Console,
    *,
    confirm_fn: Callable[[str], str] | None = None,
    is_tty: bool | None = None,
    tool_observation: str | None = None,
    tool_observation_on_screen: bool = True,
    turn_ctx: TurnContext | None = None,
) -> LlmRunInfo | None:
    """Run one turn of the terminal assistant (guidance only; no investigation run).

    ``turn_ctx`` is the immutable per-turn snapshot assembled at turn start.
    When present, snapshot fields (conversation history, integration state,
    prior investigation, synthetic-run path) are read from it rather than from
    the live session, so prompt construction reflects a stable turn-start view.
    """
    client = _load_reasoning_client(console)
    if client is None:
        return None

    ctx = turn_ctx or TurnContext.from_session(message, session)

    prompt = _build_cli_agent_prompt(
        message=message,
        session=session,
        tool_observation=tool_observation,
        tool_observation_on_screen=tool_observation_on_screen,
        turn_ctx=ctx,
    )

    run_info = _stream_cli_agent_response(
        client=client,
        prompt=prompt,
        session=session,
        console=console,
    )
    if run_info is None:
        return None

    text_str = run_info.response_text or ""
    actions = _parse_action_plan(text_str)

    handled = _execute_action_plan(
        actions,
        session,
        console,
        confirm_fn=confirm_fn,
        is_tty=is_tty,
    )

    _record_cli_agent_turn(session, message, text_str)

    if not handled:
        _render_json_like_response(console, text_str)

    return run_info


def _response_text(run: LlmRunInfo | None) -> str:
    return run.response_text if run is not None and run.response_text else ""


# ---------------------------------------------------------------------------
# Turn dispatch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TurnDependencies:
    execute_actions: RunToolCallingTurn
    gather_evidence: GatherEvidence
    answer_agent: AnswerAgent

    @classmethod
    def from_optional(
        cls,
        *,
        execute_actions: RunToolCallingTurn | None,
        gather_evidence: GatherEvidence | None,
        answer_agent: AnswerAgent | None,
    ) -> _TurnDependencies:
        return cls(
            execute_actions=execute_actions or run_tool_calling_turn,
            gather_evidence=gather_evidence or gather_tool_evidence,
            answer_agent=answer_agent or answer_cli_agent,
        )


def _should_summarize_action_observation(
    action_result: ToolCallingTurnResult,
    observation: str | None,
) -> bool:
    return (
        action_result.handled
        and observation is not None
        and action_result.executed_success_count > 0
    )


def _gather_and_answer(
    *,
    text: str,
    session: ReplSession,
    console: Console,
    deps: _TurnDependencies,
    confirm_fn: Callable[[str], str] | None,
    is_tty: bool | None,
    turn_ctx: TurnContext,
) -> LlmRunInfo | None:
    gathered = deps.gather_evidence(text, session, console, is_tty=is_tty)

    # Only pass tool_observation_on_screen when there is gathered content to
    # display (False = output is off-screen; omit = nothing was gathered).
    extra: dict[str, object] = {"turn_ctx": turn_ctx}
    if gathered:
        extra["tool_observation_on_screen"] = False

    return deps.answer_agent(
        text,
        session,
        console,
        confirm_fn=confirm_fn,
        is_tty=is_tty,
        tool_observation=gathered or None,
        **extra,  # type: ignore[arg-type]
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
    answer_agent: AnswerAgent | None = None,
) -> ShellTurnResult:
    """Run one interactive-shell turn through three paths, in order:

    1. ``answer_from_observation`` — a successful action left discovery output, so
       summarize it into a direct answer.
    2. ``action_handled`` — the action fully handled the turn; stop without the LLM.
    3. ``gather_and_answer`` — nothing was handled; gather evidence and answer.
    """
    # Snapshot session state before any turn mutations. Both the action agent
    # and the conversational assistant read from this frozen context so their
    # prompts reflect a consistent turn-start view rather than live session state.
    turn_ctx = TurnContext.from_session(text, session)

    deps = _TurnDependencies.from_optional(
        execute_actions=execute_actions,
        gather_evidence=gather_evidence,
        answer_agent=answer_agent,
    )
    accounting = ShellTurnAccounting(session=session, text=text, recorder=recorder)

    # Clear any observation left by a prior turn so only this turn's discovery
    # output can trigger a summary pass.
    session.last_command_observation = None

    action_result = deps.execute_actions(
        text,
        session,
        console,
        confirm_fn=confirm_fn,
        is_tty=is_tty,
        turn_ctx=turn_ctx,
    )
    accounting.record_action_result(action_result)

    observation = session.last_command_observation

    if _should_summarize_action_observation(action_result, observation):
        # Path 1: a successful terminal action left discovery output worth summarizing.
        with apply_reasoning_effort(turn_ctx.reasoning_effort):
            run = deps.answer_agent(
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

    elif action_result.handled:
        # Path 2: the action fully handled the turn; stop without the LLM.
        result = ShellTurnResult(
            final_intent="cli_agent_handled",
            action_result=action_result,
            assistant_response_text=action_result.response_text,
        )

    else:
        # Path 3: nothing was handled; gather evidence and answer.
        with apply_reasoning_effort(turn_ctx.reasoning_effort):
            run = _gather_and_answer(
                text=text,
                session=session,
                console=console,
                deps=deps,
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
            _logger.debug("Queued turn task was cancelled")
        except Exception as exc:
            _logger.debug("Queued turn task ended with exception: %s", exc)
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
    "answer_cli_agent",
    "handle_message_with_agent",
    "request_confirmation_via_prompt",
    "run_agent_turn_queue",
    "run_input_loop",
]
