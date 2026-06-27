"""Sample alert action tool."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.markup import escape

from interactive_shell.harness.orchestration.execution_policy import (
    execution_allowed,
    plan_investigation_execution,
)
from interactive_shell.runtime import ReplSession
from interactive_shell.tools.tool_contracts import (
    ToolContext,
    ToolEntry,
    object_schema,
    string_property,
)
from interactive_shell.ui.foreground_investigation import run_foreground_investigation
from platform.common.task_types import TaskRecord

_SAMPLE_ALERT_TEMPLATES = ("generic",)


def run_sample_alert(
    template_name: str,
    session: ReplSession,
    console: Console,
    *,
    confirm_fn: Callable[[str], str] | None = None,
    is_tty: bool | None = None,
    action_already_listed: bool = False,
) -> None:
    from cli.investigation import run_sample_alert_for_session

    plan = plan_investigation_execution(action_type="sample_alert", user_initiated=True)
    if not execution_allowed(
        plan.policy,
        session=session,
        console=console,
        action_summary=f"sample alert investigation ({template_name})",
        confirm_fn=confirm_fn,
        is_tty=is_tty,
        action_already_listed=action_already_listed,
    ):
        session.record("alert", f"sample:{template_name}", ok=False)
        return

    console.print(f"[bold]sample alert:[/bold] {escape(template_name)}")
    if session.background_mode_enabled:
        from interactive_shell.runtime.background.runner import (
            start_background_template_investigation,
        )

        start_background_template_investigation(
            template_name=template_name,
            session=session,
            console=console,
            display_command=f"sample alert:{template_name}",
        )
        session.record("alert", f"sample:{template_name}")
        return

    def _run(task: TaskRecord) -> dict[str, object]:
        return run_sample_alert_for_session(
            template_name=template_name,
            context_overrides=session.accumulated_context or None,
            cancel_requested=task.cancel_requested,
        )

    if (
        run_foreground_investigation(
            session=session,
            console=console,
            task_command=f"sample alert:{template_name}",
            run=_run,
            exception_context="interactive_shell.sample_alert",
        )
        is None
    ):
        session.record("alert", f"sample:{template_name}", ok=False)
        return

    session.record("alert", f"sample:{template_name}")


def execute_sample_alert_action(args: dict[str, Any], ctx: ToolContext) -> bool:
    template = str(args.get("template", "")).strip()
    if not template:
        return False
    run_sample_alert(
        template,
        ctx.session,
        ctx.console,
        confirm_fn=ctx.confirm_fn,
        is_tty=ctx.is_tty,
        action_already_listed=ctx.action_already_listed,
    )
    return True


TOOL_ENTRY = ToolEntry(
    name="alert_sample",
    description=(
        "Run the built-in synthetic sample alert end-to-end (read alert → "
        "investigate → diagnose). Use for any request to run/try/start/launch/"
        "fire/trigger/investigate/look at a 'sample alert', 'test alert', or "
        "'demo alert' (e.g. 'investigate a sample test alert?', 'kick off a "
        "sample alert'). These requests carry NO real pasted alert text — that "
        "is what separates them from investigation_start. Prefer this over "
        "investigation_start and assistant_handoff for sample/test/demo alerts, "
        "regardless of the verb or a trailing '?'."
    ),
    input_schema=object_schema(
        properties={
            "template": string_property(
                description="Sample alert template name to run.",
                enum=_SAMPLE_ALERT_TEMPLATES,
            )
        },
        required=("template",),
    ),
    execute=execute_sample_alert_action,
)


__all__ = ["TOOL_ENTRY", "execute_sample_alert_action", "run_sample_alert"]
