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

The ``ask`` verdict is retained so that ``trust_mode`` and any future opt-in
stricter policy still have a hook, but the policy functions here never emit
``ask``. If guardrails are reintroduced after alpha, gate them here at the
execution stage (not the planner).

This module is intentionally **pure**: it has no terminal I/O, no analytics, and
no console dependency. The decision is computed by :func:`resolve_confirmation`,
and the interaction layer (printing the reason/hint, the ``Proceed? [Y/n]``
prompt, and analytics emission) lives in
``interactive_shell.ui.execution_confirm.execution_allowed``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

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


class ConfirmationOutcome(StrEnum):
    """Pure decision for how the interaction layer should treat an action."""

    ALLOW = "allow"  # proceed, no prompt
    DENY = "deny"  # blocked by policy (show reason + hint)
    BLOCK_NON_TTY = "block_non_tty"  # ask verdict but stdin is not a TTY
    NEEDS_CONFIRMATION = "needs_confirmation"  # prompt the user


@dataclass(frozen=True)
class ConfirmationPlan:
    """Result of :func:`resolve_confirmation` (side-effect free).

    ``analytics_outcome`` / ``analytics_reason`` carry the values the interaction
    layer should emit for the non-prompt outcomes (ALLOW / DENY / BLOCK_NON_TTY).
    For ``NEEDS_CONFIRMATION`` the analytics outcome depends on the user's answer
    and is decided by the interaction layer, so both fields are ``None``.
    """

    outcome: ConfirmationOutcome
    result: ExecutionPolicyResult
    analytics_outcome: str | None = None
    analytics_reason: str | None = None


def resolve_confirmation(
    result: ExecutionPolicyResult,
    *,
    trust_mode: bool,
    is_tty: bool,
) -> ConfirmationPlan:
    """Resolve a policy result into a confirmation decision, with no side effects.

    Pure function: no console, no ``input``, no analytics. The interaction layer
    (``interactive_shell.ui.execution_confirm``) renders the decision and emits
    analytics.
    """
    if result.verdict == "deny":
        return ConfirmationPlan(
            outcome=ConfirmationOutcome.DENY,
            result=result,
            analytics_outcome="blocked",
            analytics_reason=result.reason,
        )

    if result.verdict == "allow":
        return ConfirmationPlan(
            outcome=ConfirmationOutcome.ALLOW,
            result=result,
            analytics_outcome="allowed",
            analytics_reason=result.reason,
        )

    # ask
    if trust_mode:
        return ConfirmationPlan(
            outcome=ConfirmationOutcome.ALLOW,
            result=result,
            analytics_outcome="allowed",
            analytics_reason="trust_mode_skipped_prompt",
        )

    if not is_tty:
        return ConfirmationPlan(
            outcome=ConfirmationOutcome.BLOCK_NON_TTY,
            result=result,
            analytics_outcome="blocked",
            analytics_reason="non_interactive_stdin",
        )

    return ConfirmationPlan(
        outcome=ConfirmationOutcome.NEEDS_CONFIRMATION,
        result=result,
    )


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
    "ActionExecutionMode",
    "ActionExecutionPlan",
    "ConfirmationOutcome",
    "ConfirmationPlan",
    "ExecutionPolicyResult",
    "ExecutionVerdict",
    "evaluate_code_agent_launch",
    "evaluate_investigation_launch",
    "evaluate_llm_runtime_switch",
    "evaluate_slash_command",
    "evaluate_synthetic_test_launch",
    "plan_investigation_execution",
    "plan_slash_execution",
    "resolve_confirmation",
]
