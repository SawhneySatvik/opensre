"""Terminal assistant and turn handling for the interactive OpenSRE shell.

This module is structured as a **functional core wrapped by a thin effectful
interpreter**. Each function falls into one of three layers:

- **Pure** — deterministic, no IO, no mutation. They *decide* (parse the action
  plan, filter by capabilities, plan harness effects, route the turn, render the
  prompt, reduce presentation state). These are trivially unit-testable with
  plain values and no mocks.
- **Snapshot** — read the mutable world (session, registries, grounding caches,
  files, TTY) exactly once and return immutable data
  (``_read_action_planning_env``, ``_collect_cli_agent_prompt_context``,
  ``_routing_input_from_result``).
- **Interpreter** — perform the effects described by the pure layer against the
  real console / session / subprocess boundary (``_interpret_instruction`` and
  friends, ``_stream_cli_agent_response``, ``_render_agent_presentation_transition``).

``ConsoleAgentEventSink`` and ``AgentTurnRunner`` remain as stable
imperative-shell classes layered over the functional core.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
from collections.abc import Awaitable, Callable, Coroutine, Iterator, Mapping
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
from interactive_shell.harness.llm_context.grounding.investigation_flow_reference import (
    build_investigation_flow_reference_text,
)
from interactive_shell.harness.llm_context.conversation_history import (
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
# Action plan model (pure)
# ---------------------------------------------------------------------------


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
    def from_payload(cls, payload: Mapping[str, object]) -> ActionPlanAction | None:
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


def _extract_json_object(text: str) -> dict[str, object] | None:
    """Find the first top-level JSON object embedded in *text* (pure)."""
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


def _parse_action_plan(text: str) -> tuple[ActionPlanAction, ...]:
    """Parse raw model output into an immutable action plan (pure)."""
    payload = _extract_json_object(text)
    if payload is None:
        return ()

    actions = payload.get("actions")
    if not isinstance(actions, list):
        single = ActionPlanAction.from_payload(payload)
        return (single,) if single is not None else ()

    return tuple(
        action
        for raw in actions
        if isinstance(raw, dict)
        for action in (ActionPlanAction.from_payload(raw),)
        if action is not None
    )


# ---------------------------------------------------------------------------
# Capability filtering (pure core + snapshot adapter)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapabilitySnapshot:
    """Immutable view of which capability surfaces are explicitly disabled."""

    disabled_capabilities: frozenset[str]


def _filter_actions_by_capabilities(
    actions: tuple[ActionPlanAction, ...], capabilities: CapabilitySnapshot
) -> tuple[ActionPlanAction, ...]:
    """Drop actions whose capability surface is explicitly disabled (pure)."""
    return tuple(
        action
        for action in actions
        if action.capability is None or action.capability not in capabilities.disabled_capabilities
    )


def _read_capability_snapshot(session: ReplSession) -> CapabilitySnapshot:
    """Snapshot the session's disabled capability surfaces once."""
    from interactive_shell.tools.tool_contracts import capability_not_explicitly_disabled

    disabled = frozenset(
        capability
        for capability in frozenset(_ACTION_CAPABILITY.values())
        if not capability_not_explicitly_disabled(session, capability)
    )
    return CapabilitySnapshot(disabled_capabilities=disabled)


# ---------------------------------------------------------------------------
# Action planning environment (snapshot) + pure predicates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionPlanningEnv:
    """Immutable snapshot of everything the pure action planner needs."""

    allowed_slash_actions: frozenset[str]
    registered_slash_commands: frozenset[str]
    configured_integrations_known: bool
    configured_integrations_count: int
    capabilities: CapabilitySnapshot
    repl_tty_interactive: bool


def _read_action_planning_env(session: ReplSession) -> ActionPlanningEnv:
    """Read the live world once into a frozen planning environment."""
    from interactive_shell.command_registry import SLASH_COMMANDS
    from interactive_shell.ui.components.choice_menu import repl_tty_interactive

    return ActionPlanningEnv(
        allowed_slash_actions=_ALLOWED_SLASH_ACTIONS,
        registered_slash_commands=frozenset(SLASH_COMMANDS),
        configured_integrations_known=session.configured_integrations_known,
        configured_integrations_count=len(session.configured_integrations),
        capabilities=_read_capability_snapshot(session),
        repl_tty_interactive=repl_tty_interactive(),
    )


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
def _registered_interactive_command(command: str, registered: frozenset[str]) -> bool:
    """True when *command* names a registered OpenSRE slash command (pure)."""
    parts = command.strip().split()
    if not parts:
        return False
    name = parts[0].lower()
    if name == "/":
        return True
    if not name.startswith("/"):
        return False
    return name in registered


def _integration_command_blocked(payload: str, env: ActionPlanningEnv) -> bool:
    """Block integration-management CLI runs when none are configured (pure)."""
    if not env.configured_integrations_known or env.configured_integrations_count:
        return False
    lowered = payload.strip().lower()
    return lowered.startswith("integrations") or "integration" in lowered


# ---------------------------------------------------------------------------
# Harness effects + pure action planners
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HarnessEffect:
    """A single, fully-described side effect for the interpreter to perform."""

    type: Literal[
        "print",
        "switch_llm_provider",
        "switch_toolcall_model",
        "dispatch_slash",
        "run_opensre_cli",
        "queue_interactive_command",
        "record_session",
    ]
    message: str = ""
    command: str = ""
    action: ActionPlanAction | None = None
    policy_precleared: bool = False
    ok: bool = True


@dataclass(frozen=True)
class ConfirmedEffects:
    """Effects gated behind an execution-policy confirmation.

    The interpreter runs ``on_allowed`` if the policy/confirmation clears, and
    ``on_denied`` otherwise. Modeling the branch explicitly keeps confirmation
    control flow visible and testable instead of buried in imperative returns.
    """

    policy_tool: str
    summary: str
    on_allowed: tuple[HarnessEffect, ...]
    on_denied: tuple[HarnessEffect, ...] = ()


HarnessInstruction = HarnessEffect | ConfirmedEffects


def _print(message: str) -> HarnessEffect:
    return HarnessEffect(type="print", message=message)


def _print_error(message: str) -> HarnessEffect:
    return HarnessEffect(type="print", message=f"[{ERROR}]{escape(message)}[/]")


def _plan_switch_llm_provider(action: ActionPlanAction) -> tuple[HarnessInstruction, ...]:
    if not action.provider:
        return (_print_error("missing provider for switch_llm_provider action"),)

    slash_label = f"/model set {action.provider}"
    if action.model:
        slash_label += f" {action.model}"
    if action.toolcall_model:
        slash_label += f" --toolcall-model {action.toolcall_model}"

    return (
        ConfirmedEffects(
            policy_tool="switch_llm_provider",
            summary=slash_label,
            on_allowed=(
                _print(f"[bold]$ {escape(slash_label)}[/bold]"),
                HarnessEffect(type="switch_llm_provider", action=action),
                HarnessEffect(type="record_session", command=slash_label, ok=True),
            ),
        ),
    )


def _plan_switch_toolcall_model(action: ActionPlanAction) -> tuple[HarnessInstruction, ...]:
    if not action.model:
        return (_print_error("missing model for switch_toolcall_model action"),)

    command = f"/model toolcall set {action.model}"

    return (
        ConfirmedEffects(
            policy_tool="switch_toolcall_model",
            summary=command,
            on_allowed=(
                _print(f"[bold]$ {escape(command)}[/bold]"),
                HarnessEffect(type="switch_toolcall_model", action=action),
                HarnessEffect(type="record_session", command=command, ok=True),
            ),
        ),
    )


def _plan_slash_action(
    action: ActionPlanAction, env: ActionPlanningEnv
) -> tuple[HarnessInstruction, ...]:
    command = action.command
    if command not in env.allowed_slash_actions:
        return (_print_error(f"unsupported action command: {command}"),)

    stripped = command.strip()
    name = stripped.split()[0].lower()

    # Unknown to the dispatcher: hand straight to dispatch_slash, which renders
    # its own "unknown command" feedback (no policy preclear).
    if name not in env.registered_slash_commands:
        return (HarnessEffect(type="dispatch_slash", command=command, policy_precleared=False),)

    return (
        ConfirmedEffects(
            policy_tool="slash",
            summary=stripped,
            on_allowed=(
                _print(f"[bold]$ {escape(command)}[/bold]"),
                HarnessEffect(type="dispatch_slash", command=command, policy_precleared=True),
            ),
            on_denied=(HarnessEffect(type="record_session", command=stripped, ok=False),),
        ),
    )


def _plan_cli_command(
    action: ActionPlanAction, env: ActionPlanningEnv
) -> tuple[HarnessInstruction, ...]:
    if not action.args:
        return (_print_error("missing args for run_cli_command action"),)

    if _integration_command_blocked(action.args, env):
        return (
            _print(
                f"[{WARNING}]integration command blocked: no integrations are configured "
                "in this session.[/]"
            ),
        )

    return (HarnessEffect(type="run_opensre_cli", command=action.args),)


def _plan_interactive_command(
    action: ActionPlanAction, env: ActionPlanningEnv
) -> tuple[HarnessInstruction, ...]:
    command = action.command

    if not _registered_interactive_command(command, env.registered_slash_commands):
        return (_print_error(f"unsupported interactive command: {command}"),)

    if not env.repl_tty_interactive:
        return (
            _print(f"Run [bold]{escape(command)}[/bold] in the interactive shell to continue."),
        )

    return (
        _print(f"[{DIM}]Launching[/] [{BOLD_BRAND}]{escape(command)}[/]…"),
        HarnessEffect(type="queue_interactive_command", command=command),
    )


def _plan_action_effects(
    action: ActionPlanAction, env: ActionPlanningEnv
) -> tuple[HarnessInstruction, ...]:
    """Translate one action into the instructions that realize it (pure)."""
    match action.kind:
        case "switch_llm_provider":
            return _plan_switch_llm_provider(action)
        case "switch_toolcall_model":
            return _plan_switch_toolcall_model(action)
        case "slash":
            return _plan_slash_action(action, env)
        case "run_cli_command":
            return _plan_cli_command(action, env)
        case "run_interactive":
            return _plan_interactive_command(action, env)
        case _:
            return (_print_error(f"unsupported action: {action.kind or '?'}"),)


def _plan_requested_actions_header(
    actions: tuple[ActionPlanAction, ...],
) -> tuple[HarnessInstruction, ...]:
    numbered = [
        _print(f"[{DIM}]{index}.[/] [{BOLD_BRAND}]{escape(action.label)}[/]")
        for index, action in enumerate(actions, start=1)
    ]
    return (
        _print(""),
        _print(f"[{BOLD_BRAND}]{STREAM_LABEL_ASSISTANT}:[/]"),
        _print(f"[{DIM}]Requested actions:[/]"),
        *numbered,
        _print(""),
    )


def _plan_action_plan_effects(
    actions: tuple[ActionPlanAction, ...], env: ActionPlanningEnv
) -> tuple[HarnessInstruction, ...]:
    """Plan the full action-plan execution as one instruction stream (pure)."""
    instructions: list[HarnessInstruction] = list(_plan_requested_actions_header(actions))
    for action in actions:
        instructions.append(_print(""))
        instructions.extend(_plan_action_effects(action, env))
    instructions.append(_print(""))
    return tuple(instructions)


# ---------------------------------------------------------------------------
# Effect interpreter (the single imperative edge for action execution)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionRuntime:
    """Boundary objects the interpreter needs to perform effects."""

    session: ReplSession
    console: Console
    confirm_fn: Callable[[str], str] | None
    is_tty: bool | None


def _confirm_instruction(instruction: ConfirmedEffects, runtime: ActionRuntime) -> bool:
    from interactive_shell.tools.shared import allow_tool
    from interactive_shell.ui.execution_confirm import execution_allowed

    return execution_allowed(
        allow_tool(instruction.policy_tool),
        session=runtime.session,
        console=runtime.console,
        action_summary=instruction.summary,
        confirm_fn=runtime.confirm_fn,
        is_tty=runtime.is_tty,
        action_already_listed=True,
    )


def _interpret_effect(effect: HarnessEffect, runtime: ActionRuntime) -> None:
    console = runtime.console
    session = runtime.session
    match effect.type:
        case "print":
            console.print(effect.message)
        case "switch_llm_provider":
            from interactive_shell.command_registry import switch_llm_provider

            action = effect.action
            assert action is not None  # planner always attaches the action
            switch_llm_provider(
                action.provider,
                console,
                model=action.model or None,
                toolcall_model=action.toolcall_model or None,
            )
        case "switch_toolcall_model":
            from interactive_shell.command_registry import switch_toolcall_model

            action = effect.action
            assert action is not None  # planner always attaches the action
            switch_toolcall_model(action.model, console)
        case "dispatch_slash":
            from interactive_shell.command_registry import dispatch_slash

            dispatch_slash(
                effect.command,
                session,
                console,
                confirm_fn=runtime.confirm_fn,
                is_tty=runtime.is_tty,
                policy_precleared=effect.policy_precleared,
            )
        case "run_opensre_cli":
            from interactive_shell.runtime.subprocess_runner import run_opensre_cli_command

            run_opensre_cli_command(
                effect.command,
                session,
                console,
                confirm_fn=runtime.confirm_fn,
                is_tty=runtime.is_tty,
            )
        case "queue_interactive_command":
            session.queue_auto_command(effect.command)
        case "record_session":
            session.record("slash", effect.command, ok=effect.ok)
        case _:
            raise ValueError(f"unknown harness effect type: {effect.type!r}")


def _interpret_instruction(instruction: HarnessInstruction, runtime: ActionRuntime) -> None:
    if isinstance(instruction, ConfirmedEffects):
        branch = (
            instruction.on_allowed
            if _confirm_instruction(instruction, runtime)
            else instruction.on_denied
        )
        _interpret_instructions(branch, runtime)
        return
    _interpret_effect(instruction, runtime)


def _interpret_instructions(
    instructions: tuple[HarnessInstruction, ...], runtime: ActionRuntime
) -> None:
    for instruction in instructions:
        _interpret_instruction(instruction, runtime)


def _execute_action_plan(
    actions: tuple[ActionPlanAction, ...],
    session: ReplSession,
    console: Console,
    *,
    confirm_fn: Callable[[str], str] | None = None,
    is_tty: bool | None = None,
) -> bool:
    """Plan and perform an action plan; return True iff anything was eligible."""
    if not actions:
        return False

    env = _read_action_planning_env(session)
    allowed = _filter_actions_by_capabilities(tuple(actions), env.capabilities)
    if not allowed:
        return False

    _interpret_instructions(
        _plan_action_plan_effects(allowed, env),
        ActionRuntime(session=session, console=console, confirm_fn=confirm_fn, is_tty=is_tty),
    )
    return True


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


def _build_integration_guard(ctx: TurnContext) -> str:
    """Render the no-integrations guidance block (pure over the snapshot)."""
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


# ---------------------------------------------------------------------------
# CLI agent prompt (pure render + snapshot collector)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CliAgentPromptContext:
    """All string inputs needed to render the CLI-agent prompt, frozen."""

    reference: str
    agents_md: str
    investigation_flow: str
    history: str
    prior_investigation: str
    environment: str
    integration_guard: str
    observation_block: str
    synthetic_block: str
    user_message: str


def _render_cli_agent_prompt(ctx: CliAgentPromptContext) -> str:
    """Render the final prompt string from collected context (pure)."""
    system = _build_system_prompt(
        ctx.reference,
        ctx.history,
        agents_md=ctx.agents_md,
        investigation_flow=ctx.investigation_flow,
        prior_investigation=ctx.prior_investigation,
        environment=ctx.environment,
    )

    return (
        f"{system}\n"
        f"{ctx.integration_guard}"
        f"{ctx.observation_block}"
        f"{ctx.synthetic_block}"
        f"--- User message ---\n{ctx.user_message}"
    )


def _collect_cli_agent_prompt_context(
    *,
    message: str,
    session: ReplSession,
    tool_observation: str | None,
    tool_observation_on_screen: bool,
    turn_ctx: TurnContext,
) -> CliAgentPromptContext:
    """Read grounding sources / files / snapshot once into prompt context."""
    session.grounding.log_cache_diagnostics("cli_agent_grounding")

    return CliAgentPromptContext(
        reference=session.grounding.cli.build_text(),
        agents_md=session.grounding.agents_md.build_text(),
        investigation_flow=build_investigation_flow_reference_text(),
        history=format_recent_conversation(list(turn_ctx.conversation_messages)),
        prior_investigation=(
            _summarize_last_state(turn_ctx.last_state) if turn_ctx.last_state is not None else ""
        ),
        environment=_build_environment_block(session),
        integration_guard=_build_integration_guard(turn_ctx),
        observation_block=_build_observation_block(
            tool_observation, on_screen=tool_observation_on_screen
        ),
        synthetic_block=_build_synthetic_failure_block(turn_ctx),
        user_message=message,
    )


# ---------------------------------------------------------------------------
# CLI agent answer (interpreter edge for the conversational turn)
# ---------------------------------------------------------------------------


def _load_reasoning_client(console: Console) -> Any | None:
    try:
        from core.runtime.llm.llm_client import get_llm_for_reasoning
    except Exception as exc:
        report_exception(exc, context="interactive_shell.cli_agent.import")
        console.print(f"[{ERROR}]LLM client unavailable:[/] {escape(str(exc))}")
        return None

    return get_llm_for_reasoning()


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

    prompt = _render_cli_agent_prompt(
        _collect_cli_agent_prompt_context(
            message=message,
            session=session,
            tool_observation=tool_observation,
            tool_observation_on_screen=tool_observation_on_screen,
            turn_ctx=ctx,
        )
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
    handled = _execute_action_plan(
        _parse_action_plan(text_str),
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
# Turn routing (pure router + snapshot adapter)
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


@dataclass(frozen=True)
class TurnRoutingInput:
    """Minimal facts the turn router decides on, snapshotted from the world."""

    action_handled: bool
    executed_success_count: int
    has_observation: bool


@dataclass(frozen=True)
class TurnRoute:
    """The chosen turn path."""

    intent: Literal["summarize_observation", "handled_without_llm", "gather_and_answer"]


def _route_turn(routing: TurnRoutingInput) -> TurnRoute:
    """Decide the turn path from routing facts (pure)."""
    if routing.action_handled and routing.has_observation and routing.executed_success_count > 0:
        return TurnRoute(intent="summarize_observation")
    if routing.action_handled:
        return TurnRoute(intent="handled_without_llm")
    return TurnRoute(intent="gather_and_answer")


def _routing_input_from_result(
    action_result: ToolCallingTurnResult, observation: str | None
) -> TurnRoutingInput:
    return TurnRoutingInput(
        action_handled=action_result.handled,
        executed_success_count=action_result.executed_success_count,
        has_observation=observation is not None,
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

    # When evidence was gathered, mark it off-screen so the prompt builder
    # includes it. When nothing was gathered, omit the flag entirely so the
    # call shape matches the plain conversational (no-observation) path.
    on_screen: dict[str, bool] = {"tool_observation_on_screen": False} if gathered else {}

    return deps.answer_agent(
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
    answer_agent: AnswerAgent | None = None,
) -> ShellTurnResult:
    """Run one interactive-shell turn through three paths, in order:

    1. ``summarize_observation`` — a successful action left discovery output, so
       summarize it into a direct answer.
    2. ``handled_without_llm`` — the action fully handled the turn; stop without the LLM.
    3. ``gather_and_answer`` — nothing was handled; gather evidence and answer.

    The path choice is the pure ``_route_turn``; this function is the imperative
    shell that performs the chosen path's effects.
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
    route = _route_turn(_routing_input_from_result(action_result, observation))

    match route.intent:
        case "summarize_observation":
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


# ---------------------------------------------------------------------------
# Agent lifecycle: pure presentation reducer + effectful transition
# ---------------------------------------------------------------------------


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


@dataclass(frozen=True)
class AgentPresentationState:
    """Immutable presentation state evolved across lifecycle events."""

    show_spinner: bool = False
    prompt_suppressed: bool = False


def _reduce_agent_presentation(
    state: AgentPresentationState,
    event: AgentEvent,
    *,
    should_show_spinner: bool,
) -> AgentPresentationState:
    """Compute the next presentation state for *event* (pure)."""
    match event.type:
        case "turn_start":
            return AgentPresentationState(
                show_spinner=should_show_spinner,
                prompt_suppressed=should_show_spinner,
            )
        case "turn_end":
            return AgentPresentationState()
        case "turn_interrupted" | "turn_error":
            return state
        case _:
            raise ValueError(f"Unknown agent event type: {event.type!r}")


async def _render_agent_presentation_transition(
    *,
    previous: AgentPresentationState,
    current: AgentPresentationState,
    event: AgentEvent,
    console: StreamingConsole,
    spinner: SpinnerState,
) -> None:
    """Perform the terminal side effects for one presentation transition."""
    from interactive_shell.ui.output import set_prompt_suppress_fn

    match event.type:
        case "turn_start":
            if current.show_spinner:
                spinner.start()
                set_prompt_suppress_fn(console.suppress_prompt_spinner)
        case "turn_interrupted":
            console.print(f"[{WARNING}]· interrupted[/]")
        case "turn_error":
            exc = event.error
            if exc is None:
                raise ValueError("turn_error event requires an error")
            console.print(f"[{ERROR}]turn error:[/] {escape(str(exc))}")
        case "turn_end":
            set_prompt_suppress_fn(None)
            if previous.show_spinner:
                spinner.stop()
            await asyncio.sleep(0.05)
            drain_stale_cpr_bytes()
        case _:
            raise ValueError(f"Unknown agent event type: {event.type!r}")


class ConsoleAgentEventSink:
    """Render agent lifecycle events to the terminal console.

    Imperative shell: it holds the evolving ``AgentPresentationState`` and routes
    each event through the pure ``_reduce_agent_presentation`` reducer and the
    effectful ``_render_agent_presentation_transition`` renderer.
    """

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
        self.state = AgentPresentationState()

    async def __call__(self, event: AgentEvent) -> None:
        previous = self.state
        self.state = _reduce_agent_presentation(
            previous,
            event,
            should_show_spinner=turn_should_show_spinner(event.text or "", self.session),
        )
        await _render_agent_presentation_transition(
            previous=previous,
            current=self.state,
            event=event,
            console=self.console,
            spinner=self.spinner,
        )


# ---------------------------------------------------------------------------
# Per-turn runtime: functional record + driver, with class compat wrapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentTurnRuntime:
    """Immutable dependencies for running one submitted shell turn."""

    session: ReplSession
    state: ReplState
    spinner: SpinnerState
    invalidate_prompt: Callable[[], None]


async def run_agent_turn(runtime: AgentTurnRuntime, text: str) -> None:
    """Set up shell presentation for one turn and drive its lifecycle."""
    dispatch_cancel = threading.Event()
    console = StreamingConsole(
        runtime.spinner,
        dispatch_cancel,
        prompt_invalidator=runtime.invalidate_prompt,
        highlight=False,
        force_terminal=True,
        color_system="truecolor",
        legacy_windows=False,
    )
    emit = ConsoleAgentEventSink(
        session=runtime.session,
        spinner=runtime.spinner,
        console=console,
    )
    recorder = PromptRecorder.start(
        session=runtime.session,
        text=text,
        turn_kind=_AGENT_TURN_KIND,
    )
    progress_scope = (
        contextlib.nullcontext()
        if turn_needs_exclusive_stdin(text, runtime.session)
        else repl_safe_progress_scope()
    )
    with progress_scope:
        await _run_agent_turn_loop(
            runtime=runtime,
            text=text,
            output=console,
            recorder=recorder,
            confirm=lambda prompt: request_confirmation_via_prompt(runtime.state, prompt),
            emit=emit,
            dispatch_cancel=dispatch_cancel,
        )


async def _run_agent_turn_loop(
    *,
    runtime: AgentTurnRuntime,
    text: str,
    output: StreamingConsole,
    recorder: PromptRecorder | None,
    confirm: Callable[[str], str],
    emit: AgentEventSink,
    dispatch_cancel: threading.Event,
) -> None:
    current_task = asyncio.current_task()
    if current_task is not None:
        runtime.state.start_dispatch(task=current_task, cancel_event=dispatch_cancel)
    else:
        runtime.state.attach_cancel_event(dispatch_cancel)

    await emit(AgentEvent(type="turn_start", text=text))
    try:
        await _execute_agent_turn(
            session=runtime.session,
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
        runtime.state.finish_dispatch(dispatch_cancel)
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


class AgentTurnRunner:
    """Stable class API over the functional ``run_agent_turn`` driver."""

    def __init__(
        self,
        *,
        session: ReplSession,
        state: ReplState,
        spinner: SpinnerState,
        invalidate_prompt: Callable[[], None],
    ) -> None:
        self.runtime = AgentTurnRuntime(
            session=session,
            state=state,
            spinner=spinner,
            invalidate_prompt=invalidate_prompt,
        )

    @property
    def session(self) -> ReplSession:
        return self.runtime.session

    @property
    def state(self) -> ReplState:
        return self.runtime.state

    @property
    def spinner(self) -> SpinnerState:
        return self.runtime.spinner

    async def run_agent_turn(self, text: str) -> None:
        await run_agent_turn(self.runtime, text)


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
