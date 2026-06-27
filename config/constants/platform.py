"""Host platform flags shared across the application."""

from __future__ import annotations

import os

IS_WINDOWS: bool = os.name == "nt"

__all__ = [
    "IS_WINDOWS",
]
