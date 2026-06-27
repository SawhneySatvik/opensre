"""Shell-specific execution policy for the interactive REPL.

Alpha mode allows every shell command; the only rejected case is genuinely
empty input. These helpers live next to the rest of the shell machinery so the
shared ``execution_policy`` module is not imported for shell-only concerns by
other tools. They reuse the shared policy contracts
(``ExecutionPolicyResult`` / ``ActionExecutionPlan``) from ``execution_policy``.
"""

from __future__ import annotations

import config.constants.platform as _platform
from interactive_shell.harness.orchestration.execution_policy import (
    ActionExecutionMode,
    ActionExecutionPlan,
    ExecutionPolicyResult,
)
from interactive_shell.tools.shell.parsing import (
    ParsedShellCommand,
    parse_shell_command,
)


def evaluate_shell_from_parsed(parsed: ParsedShellCommand) -> ExecutionPolicyResult:
    """Alpha mode: allow every shell command; only reject empty input.

    There is no command classification or deny floor — any command (mutating,
    ``restricted``, operators, substitution, passthrough) is allowed. A
    ``parse_error`` only occurs for empty input (e.g. a bare ``!``), which is
    rejected because there is nothing to run.
    """
    if parsed.parse_error is not None:
        return ExecutionPolicyResult(
            verdict="deny",
            action_type="shell",
            reason=parsed.parse_error,
            hint="Enter a command to run.",
            shell_classification="unrestricted",
        )

    return ExecutionPolicyResult(
        verdict="allow",
        action_type="shell",
        reason=None,
        shell_classification="unrestricted",
    )


def plan_shell_execution(parsed: ParsedShellCommand) -> ActionExecutionPlan:
    policy = evaluate_shell_from_parsed(parsed)
    classification = policy.shell_classification or "unrestricted"
    return ActionExecutionPlan(
        action_type="shell",
        classification=classification,
        execution_mode=ActionExecutionMode.FOREGROUND,
        policy=policy,
    )


def evaluate_shell_command(command: str) -> ExecutionPolicyResult:
    """Map shell policy + passthrough rules into allow/ask/deny."""
    parsed = parse_shell_command(command, is_windows=_platform.IS_WINDOWS)
    return evaluate_shell_from_parsed(parsed)


__all__ = [
    "evaluate_shell_command",
    "evaluate_shell_from_parsed",
    "plan_shell_execution",
]
