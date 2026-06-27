"""Enforce the decoupled ``agent`` package boundary and headless executability.

The whole point of the top-level :mod:`agent` package is that the agentic turn
engine no longer depends on the interactive terminal. Two invariants protect
that:

1. **No ``interactive_shell`` imports** anywhere under ``agent/`` (static check).
   The dependency direction must stay one-way: ``interactive_shell -> agent ->
   core.runtime``. If this test fails, an adapter leaked a terminal import into
   the engine — move it behind a port in :mod:`agent.ports` instead.
2. **Runs headlessly via a plain API call** (:func:`agent.api.run_agent_turn`)
   using only the in-memory headless adapters — no terminal, no ``ReplSession``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import agent


def _agent_package_root() -> Path:
    return Path(agent.__file__).resolve().parent


def _python_sources() -> list[Path]:
    return sorted(_agent_package_root().rglob("*.py"))


def test_agent_package_has_no_interactive_shell_imports() -> None:
    offenders: list[str] = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name == "interactive_shell" or name.startswith("interactive_shell."):
                    offenders.append(f"{path.name}:{node.lineno} imports {name}")

    assert not offenders, "agent/ must not import interactive_shell:\n" + "\n".join(offenders)


def test_headless_turn_runs_via_api_without_a_terminal() -> None:
    from agent.api import run_agent_turn
    from agent.results import ShellTurnResult

    result = run_agent_turn("hello, what can you do?")

    assert isinstance(result, ShellTurnResult)
    assert result.final_intent
