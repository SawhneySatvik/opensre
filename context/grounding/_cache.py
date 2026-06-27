"""Shared fingerprint-LRU machinery for file-backed grounding references.

The docs and AGENTS.md grounding references both cache parsed file records keyed
by ``(resolved_root, fingerprint)`` where the fingerprint is a digest of every
tracked file's ``(relpath, size, st_mtime_ns)``. The walk runs on every call so
in-file edits during a long-running shell invalidate the fingerprint and trigger
a re-parse; the LRU bounds memory under churn. This module owns that machinery
once so the two references stay byte-for-byte symmetric.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

from context.models import CacheStats

# Delimiters keep the SHA-256 input unambiguous across (relpath, size, mtime)
# tuple boundaries: concatenating decimal digits without separators is only
# heuristic-safe, not injective in general.
FP_FIELD_SEP = b"\x00"
FP_RECORD_SEP = b"\xff"


def fingerprint_from_paths(root: Path, files: list[Path]) -> str:
    """Digest of tracked files using paths from a single tree walk."""
    digest = hashlib.sha256()
    if not root.exists() or not root.is_dir():
        digest.update(b"nodir")
        digest.update(FP_FIELD_SEP)
        digest.update(str(root.resolve() if root.exists() else root).encode())
        digest.update(FP_FIELD_SEP)
        return digest.hexdigest()

    for path in files:
        rel = path.relative_to(root).as_posix()
        try:
            st = path.stat()
            digest.update(rel.encode())
            digest.update(FP_FIELD_SEP)
            digest.update(str(st.st_size).encode())
            digest.update(FP_FIELD_SEP)
            digest.update(str(st.st_mtime_ns).encode())
            digest.update(FP_RECORD_SEP)
        except OSError:
            continue
    return digest.hexdigest()


def excerpt(body: str, max_chars: int) -> str:
    """Trim a body to ``max_chars``, preferring to cut at a paragraph boundary."""
    body = body.strip()
    if len(body) <= max_chars:
        return body
    cutoff = body.rfind("\n\n", 0, max_chars)
    if cutoff < max_chars // 2:
        cutoff = max_chars
    return body[:cutoff].rstrip() + "\n\n[... excerpt truncated ...]\n"


class FingerprintCache[T]:
    """Bounded LRU of parsed file records keyed by ``(resolved_root, fingerprint)``.

    Construction is per grounding-reference instance, so each ``GroundingContext``
    (and thus each ``ReplSession``) owns an isolated cache with no module-level
    mutable globals.
    """

    def __init__(self, *, max_entries: int) -> None:
        self._cache: OrderedDict[tuple[str, str], tuple[T, ...]] = OrderedDict()
        self._max_entries = max_entries
        self._hits = 0
        self._misses = 0

    def get_or_parse(
        self,
        root: Path,
        *,
        iter_files: Callable[[Path], list[Path]],
        parse: Callable[[Path, list[Path]], tuple[T, ...]],
    ) -> list[T]:
        """Return cached records for ``root`` or walk+parse and cache them.

        The walk (``iter_files``) and per-file fingerprint run on every call so
        edits between calls are detected; ``parse`` only runs on a cache miss.
        """
        resolved = root.resolve() if root.exists() else root
        root_key = str(resolved)

        files = iter_files(resolved)
        fingerprint = fingerprint_from_paths(resolved, files)
        cache_key = (root_key, fingerprint)

        cached = self._cache.get(cache_key)
        if cached is not None:
            self._hits += 1
            self._cache.move_to_end(cache_key)
            return list(cached)

        self._misses += 1
        parsed = parse(resolved, files)
        while len(self._cache) >= self._max_entries:
            self._cache.popitem(last=False)
        self._cache[cache_key] = parsed
        return list(parsed)

    def clear(self) -> None:
        """Drop all cached entries and reset hit/miss counters."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self, name: str) -> CacheStats:
        """Typed hit/miss/size diagnostics for this cache."""
        return CacheStats(
            name=name,
            hits=self._hits,
            misses=self._misses,
            currsize=len(self._cache),
            maxsize=self._max_entries,
        )


__all__ = [
    "FP_FIELD_SEP",
    "FP_RECORD_SEP",
    "FingerprintCache",
    "excerpt",
    "fingerprint_from_paths",
]
