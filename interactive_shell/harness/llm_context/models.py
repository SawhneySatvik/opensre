"""Typed value models for interactive-shell LLM prompt assembly.

These are the "schema layer" for prompt building: small, immutable,
runtime-validated models that replace the loosely-typed primitives that used to
flow through the prompt builders (``tuple[str, str]`` conversation pairs, raw
section strings, and ``dict[str, Any]`` cache stats).

They deliberately use a plain frozen :class:`pydantic.BaseModel` rather than
:class:`config.strict_config.StrictConfigModel`: the strict base strips every
string field, which would silently mutate verbatim prompt ``body`` / message
``content`` and break the byte-for-byte prompt output the live turn-scenario
suite depends on. ``StrictConfigModel`` is still the right base for genuinely
config-shaped models (see ``prompt_history``).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

Role = Literal["user", "assistant"]
SectionStyle = Literal["dash", "heading", "raw"]


class _FrozenModel(BaseModel):
    """Immutable base that forbids unknown fields and preserves strings verbatim."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ConversationMessage(_FrozenModel):
    """One recent CLI-conversation turn rendered into prompt context."""

    role: Role
    content: str

    @classmethod
    def from_role_content(cls, role: str, content: str) -> ConversationMessage:
        """Build from raw ``(role, content)``, mapping any non-``user`` role to ``assistant``.

        Mirrors the legacy rendering rule where only an exact ``"user"`` role is
        labelled ``User:`` and everything else is treated as the assistant.
        """
        return cls(role="user" if role == "user" else "assistant", content=content)

    @property
    def label(self) -> str:
        """Display label used in the rendered transcript line."""
        return "User" if self.role == "user" else "Assistant"

    def render_line(self) -> str:
        """Render this message as a single ``User:``/``Assistant:`` transcript line."""
        return f"{self.label}: {self.content}"


def coerce_messages(
    messages: Sequence[ConversationMessage | tuple[str, str]],
) -> tuple[ConversationMessage, ...]:
    """Normalize a mix of ``ConversationMessage`` and ``(role, content)`` pairs.

    Entries that are neither a ``ConversationMessage`` nor a valid 2-tuple of
    strings are dropped, matching the defensive behavior of the legacy
    conversation formatter (it never raised on malformed history entries).
    """
    out: list[ConversationMessage] = []
    for entry in messages:
        if isinstance(entry, ConversationMessage):
            out.append(entry)
            continue
        try:
            role, content = entry
        except (TypeError, ValueError):
            continue
        if isinstance(role, str) and isinstance(content, str):
            out.append(ConversationMessage.from_role_content(role, content))
    return tuple(out)


class PromptSection(_FrozenModel):
    """A single labelled block of prompt text with a fixed render style.

    Render styles reproduce the exact spacing the prompt builders have always
    emitted:

    - ``dash``    -> ``"--- {label} ---\\n{body}\\n\\n"``
    - ``heading`` -> ``"=== {label} ===\\n{body}\\n\\n"``
    - ``raw``     -> ``body`` unchanged (the ``label`` is ignored)
    """

    label: str
    body: str
    style: SectionStyle = "dash"

    def render(self) -> str:
        if self.style == "raw":
            return self.body
        marker = "---" if self.style == "dash" else "==="
        return f"{marker} {self.label} {marker}\n{self.body}\n\n"


def render_sections(sections: Iterable[PromptSection | None]) -> str:
    """Concatenate sections, dropping ``None`` and empty-body blocks.

    Mirrors the legacy ``f"--- X ---\\n{body}\\n\\n" if body else ""`` idiom:
    a section with an empty ``body`` contributes nothing.
    """
    return "".join(section.render() for section in sections if section is not None and section.body)


class CacheStats(_FrozenModel):
    """Typed grounding-cache diagnostics, replacing ``dict[str, Any]`` stats.

    Fingerprint-based caches (docs, AGENTS.md) populate ``currsize`` / ``maxsize``;
    the signature-based CLI cache populates ``cached`` / ``signature`` /
    ``created_at_monotonic``. Unused fields stay ``None``.
    """

    name: str
    hits: int = 0
    misses: int = 0
    currsize: int | None = None
    maxsize: int | None = None
    cached: bool | None = None
    signature: str | None = None
    created_at_monotonic: float | None = None

    def render(self) -> str:
        """Compact single-line summary for the ``/status`` diagnostics table.

        Signature-based caches (``cached`` populated) report cached yes/no;
        fingerprint-based caches (``currsize`` populated) report entry counts.
        """
        if self.cached is not None:
            return f"hits={self.hits} misses={self.misses} cached={'yes' if self.cached else 'no'}"
        if self.currsize is not None:
            return f"hits={self.hits} misses={self.misses} entries={self.currsize}/{self.maxsize}"
        return f"hits={self.hits} misses={self.misses}"


__all__ = [
    "CacheStats",
    "ConversationMessage",
    "PromptSection",
    "Role",
    "SectionStyle",
    "coerce_messages",
    "render_sections",
]
