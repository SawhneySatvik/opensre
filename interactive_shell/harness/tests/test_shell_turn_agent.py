"""ShellTurnAgent lifecycle and boundary tests."""

from __future__ import annotations

import ast
import io
from pathlib import Path

import pytest
from rich.console import Console

from context.session import ReplSession
from interactive_shell.harness.agent import ShellTurnAgent
from interactive_shell.harness.events import AgentEvent
from interactive_shell.runtime.core.confirmation import DispatchCancelled
from interactive_shell.runtime.core.turn_accounting import ToolCallingTurnResult


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, highlight=False)


def _handled_turn(*_args: object, **_kwargs: object) -> ToolCallingTurnResult:
    return ToolCallingTurnResult(
        planned_count=1,
        executed_count=1,
        executed_success_count=1,
        has_unhandled_clause=False,
        handled=True,
        response_text="done",
    )


def test_shell_turn_agent_emits_lifecycle_events() -> None:
    events: list[AgentEvent] = []
    session = ReplSession()
    agent = ShellTurnAgent(
        session,
        execute_actions=_handled_turn,
        response_generator=lambda *_a, **_k: None,
    )
    agent.subscribe(events.append)

    result = agent.run_turn("run something", console=_console(), recorder=None)

    assert result.final_intent == "cli_agent_handled"
    assert [event.type for event in events] == ["turn_start", "turn_end"]
    assert events[0].text == "run something"


def test_shell_turn_agent_emits_interruption_event() -> None:
    events: list[AgentEvent] = []

    def _cancelled(*_args: object, **_kwargs: object) -> ToolCallingTurnResult:
        raise DispatchCancelled("cancelled")

    agent = ShellTurnAgent(ReplSession(), execute_actions=_cancelled)
    agent.subscribe(events.append)

    with pytest.raises(DispatchCancelled):
        agent.run_turn("cancel me", console=_console(), recorder=None)

    assert [event.type for event in events] == [
        "turn_start",
        "turn_interrupted",
        "turn_end",
    ]


def test_shell_turn_agent_lifecycle_module_does_not_import_core_domain_or_orchestration() -> None:
    module_path = Path(__file__).parents[1] / "agent.py"
    tree = ast.parse(module_path.read_text())

    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    disallowed = [
        name for name in imports if name.startswith(("core.domain", "core.orchestration"))
    ]
    assert disallowed == []
