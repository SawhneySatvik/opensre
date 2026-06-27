"""Investigation action tool."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.markup import escape

from interactive_shell.runtime import ReplSession
from interactive_shell.tools.shared import plan_foreground_tool
from interactive_shell.tools.tool_contracts import (
    ToolContext,
    ToolEntry,
    object_schema,
    string_property,
)
from interactive_shell.ui.execution_confirm import execution_allowed
from interactive_shell.ui.foreground_investigation import run_foreground_investigation
from platform.common.task_types import TaskRecord


def run_text_investigation(
    alert_text: str,
    session: ReplSession,
    console: Console,
    *,
    confirm_fn: Callable[[str], str] | None = None,
    is_tty: bool | None = None,
    action_already_listed: bool = False,
) -> None:
    from cli.investigation import run_investigation_for_session

    plan = plan_foreground_tool("investigation", "investigation_launch")
    if not execution_allowed(
        plan.policy,
        session=session,
        console=console,
        action_summary=f'investigation from text "{alert_text}"',
        confirm_fn=confirm_fn,
        is_tty=is_tty,
        action_already_listed=action_already_listed,
    ):
        session.record("alert", alert_text, ok=False)
        return

    console.print(f"[bold]investigation:[/bold] {escape(alert_text)}")
    if session.background_mode_enabled:
        from interactive_shell.runtime.background.runner import (
            start_background_text_investigation,
        )

        start_background_text_investigation(
            alert_text=alert_text,
            session=session,
            console=console,
            display_command="background free-text investigation",
        )
        session.record("alert", alert_text)
        return

    def _run(task: TaskRecord) -> dict[str, object]:
        return run_investigation_for_session(
            alert_text=alert_text,
            context_overrides=session.accumulated_context or None,
            cancel_requested=task.cancel_requested,
        )

    if (
        run_foreground_investigation(
            session=session,
            console=console,
            task_command=f"investigate:{alert_text}",
            run=_run,
            exception_context="interactive_shell.text_investigation",
        )
        is None
    ):
        session.record("alert", alert_text, ok=False)
        return

    session.record("alert", alert_text)


def execute_investigation_action(args: dict[str, Any], ctx: ToolContext) -> bool:
    alert_text = str(args.get("alert_text", "")).strip()
    if not alert_text:
        return False
    run_text_investigation(
        alert_text,
        ctx.session,
        ctx.console,
        confirm_fn=ctx.confirm_fn,
        is_tty=ctx.is_tty,
        action_already_listed=ctx.action_already_listed,
    )
    return True


TOOL_ENTRY = ToolEntry(
    name="investigation_start",
    description=(
        "Start an investigation with the provided alert text or quoted payload. "
        "Use whenever the user explicitly instructs you to investigate, RCA, "
        "diagnose, analyze, root-cause, or send an investigation payload — including "
        "'investigate why X ...' and placeholder quoted text like 'hello world' — "
        "regardless of CONNECTED INTEGRATIONS. In compound turns like `run /remote "
        'and then investigate "hello world"`, emit this as a separate second tool '
        "call; never drop the quoted investigation after emitting the slash command. "
        "Do NOT use for bare incident statements with no investigate verb, generic "
        "'Run an investigation.' with no subject, sample/demo alerts, or plain data "
        "lookups."
    ),
    input_schema=object_schema(
        properties={
            "alert_text": string_property(
                description="Alert text or incident details to investigate.",
                min_length=1,
            )
        },
        required=("alert_text",),
    ),
    execute=execute_investigation_action,
)


__all__ = ["TOOL_ENTRY", "execute_investigation_action", "run_text_investigation"]
