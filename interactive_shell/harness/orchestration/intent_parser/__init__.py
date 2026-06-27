"""Platform flag shared by the shell execution layer.

"""

from __future__ import annotations

import os

IS_WINDOWS = os.name == "nt"

__all__ = [
    "IS_WINDOWS",
]
