"""Action-plan execution for the conversational assistant path.

Takes the parsed :class:`ActionPlanAction` tuple produced by
:mod:`interactive_shell.harness.action_plan` and performs its effects: reads a
frozen view of the world (:class:`ActionPlanningEnv`), filters by disabled
capabilities, and dispatches each action through the execution-policy gates.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from rich.console import Console
from rich.markup import escape

from interactive_shell.harness.action_plan import _ACTION_CAPABILITY, ActionPlanAction
from interactive_shell.runtime import ReplSession
from interactive_shell.ui import (
    BOLD_BRAND,
    DIM,
    ERROR,
    STREAM_LABEL_ASSISTANT,
    WARNING,
)

_ALLOWED_SLASH_ACTIONS = frozenset(
    {
        "/model show",
        "/health",
        "/doctor",
        "/version",
    }
)


@dataclass(frozen=True)
class ActionPlanningEnv:
    """Immutable snapshot of everything action execution needs from the world."""

    allowed_slash_actions: frozenset[str]
    registered_slash_commands: frozenset[str]
    configured_integrations_known: bool
    configured_integrations_count: int
    disabled_capabilities: frozenset[str]
    repl_tty_interactive: bool


def _read_action_planning_env(session: ReplSession) -> ActionPlanningEnv:
    """Read the live world once into a frozen execution environment."""
    from interactive_shell.command_registry import SLASH_COMMANDS
    from interactive_shell.tools.tool_contracts import capability_not_explicitly_disabled
    from interactive_shell.ui.components.choice_menu import repl_tty_interactive

    disabled = frozenset(
        capability
        for capability in frozenset(_ACTION_CAPABILITY.values())
        if not capability_not_explicitly_disabled(session, capability)
    )
    return ActionPlanningEnv(
        allowed_slash_actions=_ALLOWED_SLASH_ACTIONS,
        registered_slash_commands=frozenset(SLASH_COMMANDS),
        configured_integrations_known=session.configured_integrations_known,
        configured_integrations_count=len(session.configured_integrations),
        disabled_capabilities=disabled,
        repl_tty_interactive=repl_tty_interactive(),
    )


def _filter_actions_by_capabilities(
    actions: tuple[ActionPlanAction, ...], env: ActionPlanningEnv
) -> tuple[ActionPlanAction, ...]:
    """Drop actions whose capability surface is explicitly disabled (pure)."""
    return tuple(
        action
        for action in actions
        if action.capability is None or action.capability not in env.disabled_capabilities
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


@dataclass(frozen=True)
class ActionRuntime:
    """Boundary objects the action handlers need to perform their effects."""

    session: ReplSession
    console: Console
    confirm_fn: Callable[[str], str] | None
    is_tty: bool | None


def _print_error(console: Console, message: str) -> None:
    console.print(f"[{ERROR}]{escape(message)}[/]")


def _model_set_command(action: ActionPlanAction) -> str:
    command = f"/model set {action.provider}"
    if action.model:
        command += f" {action.model}"
    if action.toolcall_model:
        command += f" --toolcall-model {action.toolcall_model}"
    return command


def _execution_allowed(*, tool: str, summary: str, runtime: ActionRuntime) -> bool:
    """Resolve the execution policy / confirmation for one action (boundary)."""
    from interactive_shell.tools.shared import allow_tool
    from interactive_shell.ui.execution_confirm import execution_allowed

    return execution_allowed(
        allow_tool(tool),
        session=runtime.session,
        console=runtime.console,
        action_summary=summary,
        confirm_fn=runtime.confirm_fn,
        is_tty=runtime.is_tty,
        action_already_listed=True,
    )


def _render_requested_actions(console: Console, actions: tuple[ActionPlanAction, ...]) -> None:
    console.print()
    console.print(f"[{BOLD_BRAND}]{STREAM_LABEL_ASSISTANT}:[/]")
    console.print(f"[{DIM}]Requested actions:[/]")
    for index, action in enumerate(actions, start=1):
        console.print(f"[{DIM}]{index}.[/] [{BOLD_BRAND}]{escape(action.label)}[/]")
    console.print()


def _execute_switch_llm_provider(action: ActionPlanAction, runtime: ActionRuntime) -> None:
    if not action.provider:
        _print_error(runtime.console, "missing provider for switch_llm_provider action")
        return

    command = _model_set_command(action)
    if not _execution_allowed(tool="switch_llm_provider", summary=command, runtime=runtime):
        return

    from interactive_shell.command_registry import switch_llm_provider

    runtime.console.print(f"[bold]$ {escape(command)}[/bold]")
    switch_llm_provider(
        action.provider,
        runtime.console,
        model=action.model or None,
        toolcall_model=action.toolcall_model or None,
    )
    runtime.session.record("slash", command, ok=True)


def _execute_switch_toolcall_model(action: ActionPlanAction, runtime: ActionRuntime) -> None:
    if not action.model:
        _print_error(runtime.console, "missing model for switch_toolcall_model action")
        return

    command = f"/model toolcall set {action.model}"
    if not _execution_allowed(tool="switch_toolcall_model", summary=command, runtime=runtime):
        return

    from interactive_shell.command_registry import switch_toolcall_model

    runtime.console.print(f"[bold]$ {escape(command)}[/bold]")
    switch_toolcall_model(action.model, runtime.console)
    runtime.session.record("slash", command, ok=True)


def _execute_slash_action(
    action: ActionPlanAction, runtime: ActionRuntime, env: ActionPlanningEnv
) -> None:
    command = action.command
    if command not in env.allowed_slash_actions:
        _print_error(runtime.console, f"unsupported action command: {command}")
        return

    from interactive_shell.command_registry import dispatch_slash

    stripped = command.strip()
    name = stripped.split()[0].lower()

    # Unknown to the dispatcher: hand straight to dispatch_slash, which renders
    # its own "unknown command" feedback (no policy preclear).
    if name not in env.registered_slash_commands:
        dispatch_slash(
            command,
            runtime.session,
            runtime.console,
            confirm_fn=runtime.confirm_fn,
            is_tty=runtime.is_tty,
            policy_precleared=False,
        )
        return

    if not _execution_allowed(tool="slash", summary=stripped, runtime=runtime):
        runtime.session.record("slash", stripped, ok=False)
        return

    runtime.console.print(f"[bold]$ {escape(command)}[/bold]")
    dispatch_slash(
        command,
        runtime.session,
        runtime.console,
        confirm_fn=runtime.confirm_fn,
        is_tty=runtime.is_tty,
        policy_precleared=True,
    )


def _execute_cli_command(
    action: ActionPlanAction, runtime: ActionRuntime, env: ActionPlanningEnv
) -> None:
    if not action.args:
        _print_error(runtime.console, "missing args for run_cli_command action")
        return

    if _integration_command_blocked(action.args, env):
        runtime.console.print(
            f"[{WARNING}]integration command blocked: no integrations are configured "
            "in this session.[/]"
        )
        return

    from interactive_shell.runtime.subprocess_runner import run_opensre_cli_command

    run_opensre_cli_command(
        action.args,
        runtime.session,
        runtime.console,
        confirm_fn=runtime.confirm_fn,
        is_tty=runtime.is_tty,
    )


def _execute_interactive_command(
    action: ActionPlanAction, runtime: ActionRuntime, env: ActionPlanningEnv
) -> None:
    command = action.command
    if not _registered_interactive_command(command, env.registered_slash_commands):
        _print_error(runtime.console, f"unsupported interactive command: {command}")
        return

    if not env.repl_tty_interactive:
        runtime.console.print(
            f"Run [bold]{escape(command)}[/bold] in the interactive shell to continue."
        )
        return

    runtime.console.print(f"[{DIM}]Launching[/] [{BOLD_BRAND}]{escape(command)}[/]…")
    runtime.session.queue_auto_command(command)


def _execute_action(
    action: ActionPlanAction, runtime: ActionRuntime, env: ActionPlanningEnv
) -> None:
    match action.kind:
        case "switch_llm_provider":
            _execute_switch_llm_provider(action, runtime)
        case "switch_toolcall_model":
            _execute_switch_toolcall_model(action, runtime)
        case "slash":
            _execute_slash_action(action, runtime, env)
        case "run_cli_command":
            _execute_cli_command(action, runtime, env)
        case "run_interactive":
            _execute_interactive_command(action, runtime, env)
        case _:
            _print_error(runtime.console, f"unsupported action: {action.kind or '?'}")


def _execute_action_plan(
    actions: tuple[ActionPlanAction, ...],
    session: ReplSession,
    console: Console,
    *,
    confirm_fn: Callable[[str], str] | None = None,
    is_tty: bool | None = None,
) -> bool:
    """Execute an action plan directly; return True iff anything was eligible."""
    if not actions:
        return False

    env = _read_action_planning_env(session)
    allowed = _filter_actions_by_capabilities(tuple(actions), env)
    if not allowed:
        return False

    runtime = ActionRuntime(
        session=session,
        console=console,
        confirm_fn=confirm_fn,
        is_tty=is_tty,
    )

    _render_requested_actions(console, allowed)
    for action in allowed:
        console.print()
        _execute_action(action, runtime, env)
    console.print()
    return True


__all__ = [
    "ActionPlanningEnv",
    "ActionRuntime",
    "_execute_action_plan",
]
