"""Shared base for interactive-shell grounding references.

A grounding reference owns a cache of some corpus (CLI help, docs, AGENTS.md)
and renders it into prompt text. Every reference exposes the same diagnostics
surface (``name``, ``stats() -> CacheStats``, ``invalidate()``), so the single
``as_grounding_source()`` adapter lives here instead of being copy-pasted into
each concrete reference. ``build_text`` stays per-class because its signature
varies (the docs reference takes a query; the others take none).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from context.grounding.diagnostics import GroundingSource
from context.models import CacheStats


class GroundingReference(ABC):
    """Base for cached grounding corpora with uniform diagnostics."""

    name: str

    @abstractmethod
    def stats(self) -> CacheStats:
        """Return typed hit/miss/size diagnostics for this reference's cache."""

    @abstractmethod
    def invalidate(self) -> None:
        """Drop the cache (tests, forced refresh)."""

    def as_grounding_source(self) -> GroundingSource:
        """Adapt this reference to the diagnostics ``GroundingSource`` shape."""
        return GroundingSource(name=self.name, stats_fn=self.stats)


__all__ = ["GroundingReference"]
