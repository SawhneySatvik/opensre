"""Central execution policy (allow / ask / deny) for interactive REPL actions.

Alpha mode: allow everything
----------------------------
OpenSRE is in **alpha**, and the interactive REPL runs with **no command
guardrails** so developer velocity stays high. Every policy decision below
resolves to ``allow`` and nothing prompts for confirmation: slash/``opensre``
commands (any tier), investigations, synthetic tests, code-agent launches, LLM
runtime switches, and shell commands of every kind — read-only, mutating,
``restricted`` (``sudo``, ``systemctl``, ``kill``, ``dd`` …), shell operators
(``| && ; > <``), and command substitution (`` ` ``/``$(...)``) — all run
immediately, in any context (TTY or not, trust mode or not).

There is intentionally **no shell-command safety policy**: the former
read-only / mutating / restricted classification and its deny floor were removed
(see ``docs/interactive-shell-action-policy.md``). The only thing shell
evaluation still rejects is genuinely empty input (a bare ``!`` or whitespace),
which is input validation rather than a guardrail.

The ``ask`` verdict and its confirmation UX (``execution_allowed``) are retained
so that ``trust_mode`` and any future opt-in stricter policy still have a hook,
but the policy functions here never emit ``ask``. If guardrails are
reintroduced after alpha, gate them here at the execution stage (not the
planner).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from rich.console import Console
from rich.markup import escape

from interactive_shell.runtime import ReplSession
from interactive_shell.ui import DIM, WARNING
from platform.analytics.cli import capture_repl_execution_policy_decision
from platform.analytics.provider import Properties

ExecutionVerdict = Literal["allow", "ask", "deny"]


class ActionExecutionMode(StrEnum):
    FOREGROUND = "foreground"
    BACKGROUND = "background"
    FOREGROUND_STREAMING = "foreground_streaming"


@dataclass(frozen=True)
class ExecutionPolicyResult:
    """Result of evaluating whether an action may run."""

    verdict: ExecutionVerdict
    action_type: str
    reason: str | None
    hint: str | None = None
    shell_classification: str | None = None


@dataclass(frozen=True)
class ActionExecutionPlan:
    """Unified execution plan contract shared across action executors."""

    action_type: str
    classification: str
    execution_mode: ActionExecutionMode
    policy: ExecutionPolicyResult


def _default_confirm_fn(prompt: str) -> str:
    return input(prompt)


DEFAULT_CONFIRM_FN: Callable[[str], str] = _default_confirm_fn


def _emit_decision(
    *,
    action_type: str,
    policy_verdict: ExecutionVerdict,
    outcome: str,
    trust_mode: bool,
    reason: str | None,
    user_prompted: bool = False,
) -> None:
    props: Properties = {
        "action_type": action_type,
        "policy_verdict": policy_verdict,
        "outcome": outcome,
        "trust_mode": trust_mode,
    }
    if reason:
        props["reason"] = reason[:240]
    if user_prompted:
        props["user_prompted"] = True
    capture_repl_execution_policy_decision(props)


def execution_allowed(
    result: ExecutionPolicyResult,
    *,
    session: ReplSession,
    console: Console,
    action_summary: str,
    confirm_fn: Callable[[str], str] | None = None,
    is_tty: bool | None = None,
    action_already_listed: bool = False,
) -> bool:
    """Print policy UX, emit analytics, and return whether execution should proceed.

    When ``action_already_listed`` is True (e.g. assistant printed a numbered action plan),
    the TTY prompt omits repeating ``action_summary`` and shows only the policy reason.
    """
    trust_mode = session.trust_mode
    tty = sys.stdin.isatty() if is_tty is None else is_tty
    confirm = confirm_fn or DEFAULT_CONFIRM_FN

    if result.verdict == "deny":
        _emit_decision(
            action_type=result.action_type,
            policy_verdict="deny",
            outcome="blocked",
            trust_mode=trust_mode,
            reason=result.reason,
        )
        console.print(f"[{WARNING}]Action blocked:[/] {escape(result.reason or 'not allowed')}")
        if result.hint:
            console.print(f"[{DIM}]{escape(result.hint)}[/]")
        return False

    if result.verdict == "allow":
        _emit_decision(
            action_type=result.action_type,
            policy_verdict="allow",
            outcome="allowed",
            trust_mode=trust_mode,
            reason=result.reason,
        )
        return True

    # ask
    if trust_mode:
        _emit_decision(
            action_type=result.action_type,
            policy_verdict="ask",
            outcome="allowed",
            trust_mode=trust_mode,
            reason="trust_mode_skipped_prompt",
        )
        return True

    if not tty:
        _emit_decision(
            action_type=result.action_type,
            policy_verdict="ask",
            outcome="blocked",
            trust_mode=trust_mode,
            reason="non_interactive_stdin",
        )
        console.print(
            f"[{WARNING}]confirmation required but stdin is not a TTY; "
            f"enable trust mode with[/] [bold]/trust[/bold] [{WARNING}]or rerun in a terminal.[/]"
        )
        console.print(f"[{DIM}]{escape(action_summary)}[/]")
        return False

    reason = (result.reason or "this action").strip()
    summary = action_summary.strip()
    if action_already_listed:
        console.print(f"[{WARNING}]Confirm:[/] [{DIM}]{escape(reason)}[/]")
    elif summary:
        console.print(
            f"[{WARNING}]Confirm[/] [bold]{escape(summary)}[/bold] [{DIM}]— {escape(reason)}[/]"
        )
    else:
        console.print(f"[{WARNING}]Confirm:[/] [{DIM}]{escape(reason)}[/]")
    answer = confirm("Proceed? [Y/n] ").strip().lower()
    if answer not in {"", "y", "yes"}:
        _emit_decision(
            action_type=result.action_type,
            policy_verdict="ask",
            outcome="aborted",
            trust_mode=trust_mode,
            reason="user_declined",
            user_prompted=True,
        )
        console.print(f"[{DIM}]cancelled.[/]")
        return False

    _emit_decision(
        action_type=result.action_type,
        policy_verdict="ask",
        outcome="allowed",
        trust_mode=trust_mode,
        reason="user_confirmed",
        user_prompted=True,
    )
    return True


def evaluate_slash_command() -> ExecutionPolicyResult:
    """Slash execution verdict.

    Default-allow: every slash command resolves to ``allow``.
    """
    return ExecutionPolicyResult(verdict="allow", action_type="slash", reason=None)


def plan_slash_execution() -> ActionExecutionPlan:
    policy = evaluate_slash_command()
    return ActionExecutionPlan(
        action_type="slash",
        classification="slash",
        execution_mode=ActionExecutionMode.FOREGROUND,
        policy=policy,
    )


def evaluate_investigation_launch(
    *,
    action_type: Literal["investigation", "sample_alert"],
    user_initiated: bool = False,
) -> ExecutionPolicyResult:
    """Policy for starting an RCA / investigation pipeline from the REPL.

    Default-allow: investigations run without confirmation whether or not the
    launch was ``user_initiated``.
    """
    del user_initiated  # default-allow: launches never require confirmation
    return ExecutionPolicyResult(
        verdict="allow",
        action_type=action_type,
        reason=None,
    )


def plan_investigation_execution(
    *,
    action_type: Literal["investigation", "sample_alert"],
    user_initiated: bool = False,
) -> ActionExecutionPlan:
    policy = evaluate_investigation_launch(action_type=action_type, user_initiated=user_initiated)
    return ActionExecutionPlan(
        action_type=action_type,
        classification="investigation_launch",
        execution_mode=ActionExecutionMode.FOREGROUND,
        policy=policy,
    )


def evaluate_synthetic_test_launch() -> ExecutionPolicyResult:
    return ExecutionPolicyResult(
        verdict="allow",
        action_type="synthetic_test",
        reason=None,
    )


def evaluate_code_agent_launch() -> ExecutionPolicyResult:
    return ExecutionPolicyResult(
        verdict="allow",
        action_type="code_agent",
        reason=None,
    )


def evaluate_llm_runtime_switch(*, action_type: str) -> ExecutionPolicyResult:
    return ExecutionPolicyResult(
        verdict="allow",
        action_type=action_type,
        reason=None,
    )


__all__ = [
    "DEFAULT_CONFIRM_FN",
    "ExecutionPolicyResult",
    "ExecutionVerdict",
    "evaluate_code_agent_launch",
    "evaluate_investigation_launch",
    "evaluate_llm_runtime_switch",
    "evaluate_slash_command",
    "evaluate_synthetic_test_launch",
    "execution_allowed",
]
