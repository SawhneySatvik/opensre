"""Unit tests for the pure REPL execution policy.

Alpha mode: policy functions resolve to ``allow`` with no confirmation prompt
and there is no command guardrail. The ``ask`` verdict is retained for
``trust_mode`` / future opt-in stricter policy, so those paths are covered here
by exercising :func:`resolve_confirmation` with explicitly-constructed ``ask`` /
``deny`` results.

This module is pure (no console, no ``input``, no analytics). The interaction
layer (``execution_allowed``) and its terminal/analytics behavior are covered by
``tests/interactive_shell/ui/test_execution_confirm.py``. Shell-specific policy
lives in ``interactive_shell.tools.shell.policy`` and is covered by
``tests/interactive_shell/shell/test_policy.py``.
"""

from __future__ import annotations

from interactive_shell.harness.execution_policy import (
    ConfirmationOutcome,
    ExecutionPolicyResult,
    evaluate_code_agent_launch,
    evaluate_investigation_launch,
    evaluate_llm_runtime_switch,
    evaluate_slash_command,
    evaluate_synthetic_test_launch,
    resolve_confirmation,
)


def _ask_result() -> ExecutionPolicyResult:
    """An explicit ``ask`` verdict (the default policy no longer emits these)."""
    return ExecutionPolicyResult(
        verdict="ask",
        action_type="slash",
        reason="this command may change configuration or run heavy work",
    )


# --- Default-allow policy decisions -----------------------------------------


def test_slash_command_is_allow() -> None:
    r = evaluate_slash_command()
    assert r.verdict == "allow"
    assert r.action_type == "slash"


def test_investigation_launch_is_allow() -> None:
    r = evaluate_investigation_launch(action_type="investigation")
    assert r.verdict == "allow"
    assert r.action_type == "investigation"


def test_investigation_launch_user_initiated_is_allow() -> None:
    r = evaluate_investigation_launch(action_type="investigation", user_initiated=True)
    assert r.verdict == "allow"
    assert r.action_type == "investigation"


def test_sample_alert_user_initiated_is_allow() -> None:
    r = evaluate_investigation_launch(action_type="sample_alert", user_initiated=True)
    assert r.verdict == "allow"
    assert r.action_type == "sample_alert"


def test_synthetic_is_allow() -> None:
    r = evaluate_synthetic_test_launch()
    assert r.verdict == "allow"


def test_code_agent_is_allow() -> None:
    r = evaluate_code_agent_launch()
    assert r.verdict == "allow"


def test_llm_runtime_switch_is_allow() -> None:
    r = evaluate_llm_runtime_switch(action_type="llm_runtime")
    assert r.verdict == "allow"


# --- resolve_confirmation: pure decision (no side effects) ------------------


def test_resolve_allow_verdict_proceeds() -> None:
    plan = resolve_confirmation(evaluate_slash_command(), trust_mode=False, is_tty=True)
    assert plan.outcome == ConfirmationOutcome.ALLOW
    assert plan.analytics_outcome == "allowed"


def test_resolve_deny_verdict_blocks() -> None:
    result = ExecutionPolicyResult(
        verdict="deny",
        action_type="shell",
        reason="empty command.",
        hint="Enter a command to run.",
    )
    plan = resolve_confirmation(result, trust_mode=False, is_tty=True)
    assert plan.outcome == ConfirmationOutcome.DENY
    assert plan.analytics_outcome == "blocked"
    assert plan.analytics_reason == "empty command."


def test_resolve_ask_trust_mode_allows_without_prompt() -> None:
    plan = resolve_confirmation(_ask_result(), trust_mode=True, is_tty=True)
    assert plan.outcome == ConfirmationOutcome.ALLOW
    assert plan.analytics_outcome == "allowed"
    assert plan.analytics_reason == "trust_mode_skipped_prompt"


def test_resolve_ask_non_tty_blocks() -> None:
    plan = resolve_confirmation(_ask_result(), trust_mode=False, is_tty=False)
    assert plan.outcome == ConfirmationOutcome.BLOCK_NON_TTY
    assert plan.analytics_outcome == "blocked"
    assert plan.analytics_reason == "non_interactive_stdin"


def test_resolve_ask_tty_needs_confirmation() -> None:
    plan = resolve_confirmation(_ask_result(), trust_mode=False, is_tty=True)
    assert plan.outcome == ConfirmationOutcome.NEEDS_CONFIRMATION
    # The analytics outcome for a prompt is decided by the interaction layer.
    assert plan.analytics_outcome is None
    assert plan.analytics_reason is None
