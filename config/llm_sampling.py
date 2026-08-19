"""Sampling controls for the agent tool-calling tier, deterministic by default.

Callers must read these at request-build time, not client construction:
``core.llm.factory`` caches one client per ``(role, transport, provider)`` and
the cache key excludes sampling, so a constructor-resolved value would be frozen
at first construction.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Final

AGENT_TEMPERATURE_ENV: Final[str] = "OPENSRE_AGENT_TEMPERATURE"
AGENT_SEED_ENV: Final[str] = "OPENSRE_AGENT_SEED"

DEFAULT_AGENT_TEMPERATURE: Final[float] = 0.0
DEFAULT_AGENT_SEED: Final[int] = 0

_DEFER_VALUES: Final[frozenset[str]] = frozenset({"none", "default"})


def _resolve[Value: (float, int)](
    env_name: str, default: Value, parse: Callable[[str], Value]
) -> Value | None:
    """Read a sampling knob from the environment; ``None`` means omit the param.

    Unparseable values fall back to ``default`` so a typo cannot silently
    restore provider sampling.
    """
    raw = os.getenv(env_name, "").strip().lower()
    if not raw:
        return default
    if raw in _DEFER_VALUES:
        return None
    try:
        value = parse(raw)
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def get_agent_temperature() -> float | None:
    """Sampling temperature for agent tool-calling requests, or ``None`` to omit."""
    return _resolve(AGENT_TEMPERATURE_ENV, DEFAULT_AGENT_TEMPERATURE, float)


def get_agent_seed() -> int | None:
    """Sampling seed for agent requests on providers that accept one, or ``None`` to omit."""
    return _resolve(AGENT_SEED_ENV, DEFAULT_AGENT_SEED, int)


__all__ = [
    "AGENT_SEED_ENV",
    "AGENT_TEMPERATURE_ENV",
    "DEFAULT_AGENT_SEED",
    "DEFAULT_AGENT_TEMPERATURE",
    "get_agent_seed",
    "get_agent_temperature",
]
