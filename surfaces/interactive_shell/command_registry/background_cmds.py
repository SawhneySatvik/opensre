"""Slash commands for background investigation mode.

Read forms answer from the durable store as well as the session, so a completed
RCA can be looked up from a chat transport. Write forms stay shell-only.
"""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape

from core.agent_harness.spi.session_state import (
    background_investigations,
    background_mode_enabled,
    background_notification_channels,
    session_terminal,
)
from platform.background_investigations.types import BackgroundInvestigationRecord
from surfaces.interactive_shell.command_registry.types import SlashCommand
from surfaces.interactive_shell.runtime import Session
from surfaces.interactive_shell.ui import (
    BOLD_BRAND,
    DIM,
    ERROR,
    HIGHLIGHT,
    WARNING,
    print_repl_table,
    repl_table,
)
from surfaces.interactive_shell.utils.error_handling.exception_reporting import report_exception

# A chat message caps at 4096 characters and the transports tail-truncate.
_CHAT_LIST_LIMIT = 10
_CHAT_CAUSE_CHARS = 120


def _allowed_notify_channels() -> tuple[str, ...]:
    """Channels that advertise background-RCA delivery, sorted.

    Derived from the adapter registry rather than a curated tuple, so adding an
    adapter makes its channel selectable without editing this file. Registration
    is triggered here rather than at REPL boot: the four adapter modules keep
    their vendor clients inside the delivery functions, so this costs nine
    modules at command time and nothing on the boot path.
    """
    from bootstrap.adapters import install_notification_adapters
    from platform.delivery.notifications.outbound_registry import (
        BACKGROUND_RCA,
        outbound_adapter_names_for,
    )

    install_notification_adapters()
    return outbound_adapter_names_for(BACKGROUND_RCA)


_BACKGROUND_FIRST_ARGS: tuple[tuple[str, str], ...] = (
    ("on", "enable background investigation mode"),
    ("off", "disable background investigation mode"),
    ("status", "show current background-mode state"),
    ("list", "list tracked background investigations"),
    ("show", "show one background investigation summary"),
    ("use", "promote a completed background RCA into active follow-up context"),
    ("notify", "show or update background notification channels"),
)


def _tracked_records(
    session: Session, console: Console
) -> dict[str, BackgroundInvestigationRecord]:
    """Records this session is holding, with the durable ones behind them.

    Session-live first: the runner persists only on the way out, so an in-flight
    record is fresher in memory, and only the live copy carries ``final_state``,
    which ``/background use`` needs.

    Never raises: ``dispatch_slash`` puts no ``try`` around the handler. The
    reported message stays generic because a chat transport is an external sink
    and the store's error text carries the absolute document path.
    """
    records: dict[str, BackgroundInvestigationRecord] = dict(background_investigations(session))
    try:
        from platform.background_investigations.store import (
            background_investigation_store,
        )

        stored = background_investigation_store().list_recent(limit=None)
    except Exception as exc:  # noqa: BLE001
        report_exception(exc, context="surfaces.interactive_shell.background_read")
        console.print(
            f"[{WARNING}]stored background records are unavailable[/] "
            f"[{DIM}]({escape(type(exc).__name__)}); showing this session only.[/]"
        )
        return records
    for record in stored:
        records.setdefault(record.task_id, record)
    return records


def _notify_channels(session: Session) -> tuple[str, ...]:
    """Channels for this session, falling back to the durable ones.

    A chat session has no terminal facet to hold preferences, so without the
    store it would report "none" however the shell was configured.
    """
    channels = background_notification_channels(session)
    if channels or session_terminal(session) is not None:
        return channels
    try:
        from platform.background_investigations.store import (
            background_investigation_store,
        )

        return background_investigation_store().notify_channels()
    except Exception:  # noqa: BLE001
        return ()


def _persist_notify_channels(console: Console, channels: tuple[str, ...]) -> None:
    """Write the channels so the next shell starts with them.

    Reported rather than raised: the in-session change already took effect, and
    the user should know it will not outlive the session rather than lose the
    turn to a traceback.
    """
    try:
        from platform.background_investigations.store import (
            background_investigation_store,
        )

        background_investigation_store().set_notify_channels(channels)
    except Exception as exc:  # noqa: BLE001
        report_exception(exc, context="surfaces.interactive_shell.background_notify_persist")
        console.print(
            f"[{WARNING}]channels not saved for the next session[/] "
            f"[{DIM}]({escape(type(exc).__name__)}); they apply to this one.[/]"
        )


def _plain_status(session: Session, records: dict[str, BackgroundInvestigationRecord]) -> str:
    return "\n".join(
        (
            "Background mode",
            f"  enabled: {'yes' if background_mode_enabled(session) else 'no'}",
            f"  tracked jobs: {len(records)}",
            f"  notify channels: {', '.join(_notify_channels(session)) or 'none'}",
        )
    )


def _plain_list(records: dict[str, BackgroundInvestigationRecord]) -> str:
    """One line per record, root cause trimmed.

    Bounded because the transports tail-truncate: an unbounded list drops the
    closing hint first, which is the line telling the reader how to get the rest.
    """
    from platform.common.truncation import truncate

    lines = ["Background investigations", ""]
    for task_id, record in list(records.items())[:_CHAT_LIST_LIMIT]:
        cause = (
            truncate(record.root_cause, _CHAT_CAUSE_CHARS, suffix="…") if record.root_cause else "-"
        )
        lines.append(f"  {task_id}  [{record.status}]  {record.command}")
        lines.append(f"      {cause}")
    remaining = len(records) - _CHAT_LIST_LIMIT
    if remaining > 0:
        lines.append(f"  ... and {remaining} more")
    lines.append("")
    lines.append("Use /background show <task_id> for the full RCA.")
    return "\n".join(lines)


def _chat_safe_outcome(outcome: str) -> str:
    return "failed" if outcome.startswith("failed:") else outcome


def _plain_show(task_id: str, record: BackgroundInvestigationRecord) -> str:
    """The same bounded sections the chat notification adapters already send."""
    from platform.notifications.rca_summary import summary_sections

    command, root_cause, top_analysis, next_steps = summary_sections(record)
    lines = [
        f"Background investigation {task_id}",
        "",
        f"  status: {record.status}",
        f"  command: {command}",
        "",
        f"Root cause: {root_cause or '-'}",
    ]
    if top_analysis:
        lines += ["", "Top analysis:"] + [f"  - {item}" for item in top_analysis]
    if next_steps:
        lines += ["", "What to do next:"] + [f"  - {item}" for item in next_steps]
    if record.notification_results:
        delivered = ", ".join(
            f"{k}:{_chat_safe_outcome(v)}" for k, v in record.notification_results.items()
        )
        lines += ["", f"Notified: {delivered}"]
    return "\n".join(lines)


def _reply_headless(session: Session, console: Console, message: str) -> bool:
    """Answer a chat surface in plain text, printed as well as published."""
    from surfaces.interactive_shell.command_registry.cli_parity import (
        publish_headless_slash_response,
    )

    console.print(escape(message))
    publish_headless_slash_response(session, message=message)
    return True


def _render_background_status(session: Session, console: Console) -> None:
    records = _tracked_records(session, console)
    if session_terminal(session) is None:
        _reply_headless(session, console, _plain_status(session, records))
        return
    table = repl_table(title="Background mode\n", title_style=BOLD_BRAND, show_header=False)
    table.add_column("key", style="bold")
    table.add_column("value")
    table.add_row("enabled", "yes" if background_mode_enabled(session) else "no")
    table.add_row("tracked jobs", str(len(records)))
    table.add_row("notify channels", ", ".join(_notify_channels(session)) or "none")
    print_repl_table(console, table)


def _reject_repl_only(session: Session, console: Console, form: str) -> bool:
    console.print(
        f"[{ERROR}]/background {escape(form)} changes interactive-shell state.[/]\n"
        f"[{DIM}]Run it in the REPL:[/] uv run opensre"
    )
    session.mark_latest(ok=False, kind="slash")
    return True


def _cmd_background(session: Session, console: Console, args: list[str]) -> bool:
    sub = (args[0].lower() if args else "status").strip()

    if sub == "on":
        if session_terminal(session) is None:
            return _reject_repl_only(session, console, "on")
        session.terminal.background_mode_enabled = True
        console.print(f"[{HIGHLIGHT}]background mode enabled[/]")
        return True
    if sub == "off":
        if session_terminal(session) is None:
            return _reject_repl_only(session, console, "off")
        session.terminal.background_mode_enabled = False
        console.print(f"[{DIM}]background mode disabled[/]")
        return True
    if sub == "status":
        _render_background_status(session, console)
        return True
    if sub == "list":
        tracked = _tracked_records(session, console)
        if not tracked:
            console.print(f"[{DIM}]no background investigations tracked.[/]")
            return True
        if session_terminal(session) is None:
            return _reply_headless(session, console, _plain_list(tracked))
        table = repl_table(title="Background investigations\n", title_style=BOLD_BRAND)
        table.add_column("id", style="bold")
        table.add_column("status")
        table.add_column("command")
        table.add_column("root cause", overflow="fold")
        for task_id, tracked_record in tracked.items():
            table.add_row(
                task_id,
                tracked_record.status,
                tracked_record.command,
                escape(tracked_record.root_cause or "—"),
            )
        print_repl_table(console, table)
        return True
    if sub == "show":
        if len(args) < 2:
            console.print(f"[{ERROR}]usage:[/] /background show <task_id>")
            session.mark_latest(ok=False, kind="slash")
            return True
        task_id = args[1]
        selected_record = _tracked_records(session, console).get(task_id)
        if selected_record is None:
            console.print(f"[{ERROR}]unknown background task:[/] {escape(task_id)}")
            session.mark_latest(ok=False, kind="slash")
            return True
        if session_terminal(session) is None:
            return _reply_headless(session, console, _plain_show(task_id, selected_record))
        table = repl_table(
            title=f"Background investigation: {task_id}\n",
            title_style=BOLD_BRAND,
            show_header=False,
        )
        table.add_column("key", style="bold")
        table.add_column("value", overflow="fold")
        table.add_row("status", selected_record.status)
        table.add_row("command", selected_record.command)
        table.add_row("root cause", escape(selected_record.root_cause or "—"))
        table.add_row(
            "top analysis",
            escape(
                "; ".join(selected_record.top_analysis) if selected_record.top_analysis else "—"
            ),
        )
        table.add_row(
            "next steps",
            escape("; ".join(selected_record.next_steps) if selected_record.next_steps else "—"),
        )
        table.add_row(
            "notify",
            escape(
                ", ".join(f"{k}:{v}" for k, v in selected_record.notification_results.items())
                or "—"
            ),
        )
        print_repl_table(console, table)
        return True
    if sub == "use":
        # Before the lookup: the promoted state is not persisted, so a chat
        # surface would otherwise call an id it just listed unknown.
        if session_terminal(session) is None:
            return _reject_repl_only(session, console, "use")
        if len(args) < 2:
            console.print(f"[{ERROR}]usage:[/] /background use <task_id>")
            session.mark_latest(ok=False, kind="slash")
            return True
        task_id = args[1]
        selected_record = _tracked_records(session, console).get(task_id)
        if selected_record is None:
            console.print(f"[{ERROR}]unknown background task:[/] {escape(task_id)}")
            session.mark_latest(ok=False, kind="slash")
            return True
        if not selected_record.final_state:
            console.print(
                f"[{ERROR}]background task has no completed RCA state yet:[/] {escape(task_id)}"
            )
            session.mark_latest(ok=False, kind="slash")
            return True
        session.last_state = dict(selected_record.final_state)
        session.accumulate_from_state(selected_record.final_state)
        console.print(
            f"[{HIGHLIGHT}]background RCA active[/] "
            f"[{DIM}]— follow-up context now points to {escape(task_id)}.[/]"
        )
        return True
    if sub == "notify":
        action = (args[1].lower() if len(args) > 1 else "list").strip()
        if action == "list":
            channels = _notify_channels(session)
            if session_terminal(session) is None:
                return _reply_headless(
                    session,
                    console,
                    f"Background notify channels: {', '.join(channels) or 'none'}",
                )
            console.print(f"[{DIM}]background notify channels:[/] {', '.join(channels) or 'none'}")
            return True
        if action == "set":
            if session_terminal(session) is None:
                return _reject_repl_only(session, console, "notify set")
            if len(args) < 3:
                console.print(f"[{ERROR}]usage:[/] /background notify set <channel[,channel...]>")
                session.mark_latest(ok=False, kind="slash")
                return True
            requested = [part.strip().lower() for part in args[2].split(",") if part.strip()]
            allowed = _allowed_notify_channels()
            invalid = [part for part in requested if part not in allowed]
            if invalid:
                console.print(
                    f"[{ERROR}]invalid channel(s):[/] {escape(', '.join(invalid))} "
                    f"[{DIM}](allowed: {', '.join(allowed)})[/]"
                )
                session.mark_latest(ok=False, kind="slash")
                return True
            preferences = session.terminal.background_notification_preferences
            preferences.set_channels(requested)
            _persist_notify_channels(console, preferences.channels)
            console.print(
                f"[{HIGHLIGHT}]background notify channels set:[/] {', '.join(preferences.channels)}"
            )
            return True
        console.print(f"[{ERROR}]unknown notify subcommand:[/] {escape(action)}")
        session.mark_latest(ok=False, kind="slash")
        return True

    console.print(
        f"[{ERROR}]unknown subcommand:[/] {escape(sub)}  "
        "(try [bold]/background status[/bold], [bold]/background list[/bold], "
        "[bold]/background show <task_id>[/bold], or [bold]/background notify list[/bold])"
    )
    session.mark_latest(ok=False, kind="slash")
    return True


COMMANDS: list[SlashCommand] = [
    SlashCommand(
        "/background",
        "Manage background investigation mode and completed RCA summaries.",
        _cmd_background,
        usage=(
            "/background on",
            "/background off",
            "/background status",
            "/background list",
            "/background show <task_id>",
            "/background use <task_id>",
            "/background notify list",
            "/background notify set <channel[,channel...]>",
        ),
        first_arg_completions=_BACKGROUND_FIRST_ARGS,
    )
]

__all__ = ["COMMANDS"]
