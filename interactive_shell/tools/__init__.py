"""Interactive-shell action tools.

Import tool submodules explicitly (for example
``interactive_shell.tools.slash_tool`` or ``interactive_shell.tools.catalog``)
rather than relying on this package initializer to eagerly import them.

``tool_contracts`` lives in this package and is imported by
``command_registry.slash_catalog`` during early import wiring. Eagerly importing
the tool submodules here (several of which import back into ``command_registry``)
would reintroduce a circular import during interactive-shell startup.
"""

from __future__ import annotations
