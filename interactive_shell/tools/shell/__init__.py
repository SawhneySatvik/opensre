"""Shell command machinery for the interactive REPL.

Groups the shell command-line concern next to the agent-facing
``interactive_shell.tools.shell_tool``:

* ``parsing`` turns command text into an executable shape,
* ``policy`` resolves the (alpha-mode, allow-everything) shell execution plan,
* ``execution`` runs the subprocess and returns a structured result,
* ``runner`` wires parsing, policy, builtins (``cd`` / ``pwd``), and execution
  together and records the turn.

Import submodules explicitly (for example ``interactive_shell.tools.shell.runner``)
rather than relying on this package initializer, to keep interactive-shell
startup import-light.
"""

from __future__ import annotations
