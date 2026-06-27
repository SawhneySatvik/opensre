"""Verbose diagnostics for interactive-shell grounding caches."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable

from interactive_shell.harness.llm_context.models import CacheStats

_logger = logging.getLogger(__name__)


class GroundingSource:
    """A single grounding cache source exposing typed stats for diagnostics."""

    def __init__(self, *, name: str, stats_fn: Callable[[], CacheStats]) -> None:
        self.name = name
        self.stats_fn = stats_fn


def log_grounding_cache_diagnostics(sources: Iterable[GroundingSource], reason: str) -> None:
    """Log the provided grounding cache stats when ``TRACER_VERBOSE=1``."""
    if os.environ.get("TRACER_VERBOSE") != "1":
        return
    for source in sources:
        stats = source.stats_fn()
        _logger.debug("grounding cache [%s] %s=%s", reason, source.name, stats)


__all__ = [
    "GroundingSource",
    "log_grounding_cache_diagnostics",
]
