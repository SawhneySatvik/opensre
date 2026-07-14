"""Per-model token pricing for the dashboard's ``$/hr`` column.

``$/hr`` is a *projected hourly burn rate* derived from the trailing
60 s usage window, not the actual spend over the last hour. The
sampler keeps input/output/cache buckets, so pricing applies the right
rate to each bucket instead of using the legacy 70/30 blend.

Rates come primarily from **LiteLLM's community-maintained cost table**
(``litellm.model_cost``, ~2.8k models), pinned to the snapshot bundled
inside the ``litellm`` package so lookups stay offline (see
:func:`~core.llm.transports.litellm.local_cost_map_bootstrap.ensure_local_model_cost_map`).
:data:`MODEL_PRICES` is now a small fallback table for the handful of
model ids LiteLLM's bundled map does not yet carry. Unknown models
return ``None`` so the dashboard renders ``-`` rather than inventing a
rate.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import TypeGuard

from core.llm.transports.litellm.frozen_tiktoken_bootstrap import (
    ensure_tiktoken_encodings_discoverable,
)
from core.llm.transports.litellm.local_cost_map_bootstrap import (
    ensure_local_model_cost_map,
)

# Set LITELLM_LOCAL_MODEL_COST_MAP before any ``import litellm`` so litellm's
# own import-time price-map load stays offline (no network fetch). We still read
# the snapshot file directly for import-order-independent determinism (see
# _litellm_local_cost_map); this only avoids the wasted fetch when we import
# first. Kept ahead of the sibling import below for the same reason.
ensure_local_model_cost_map()

from tools.system.fleet_monitoring.meters import TokenUsage  # noqa: E402

#: litellm's bundled price/context snapshot, read directly rather than via the
#: process-wide ``litellm.model_cost`` global (see _litellm_local_cost_map). The
#: filename is already a load-bearing contract — tests/packaging/
#: test_litellm_bundle_contract.py asserts it ships in frozen builds.
_LITELLM_PRICE_SNAPSHOT_FILENAME = "model_prices_and_context_window_backup.json"

#: Verification date for the local fallback table only. Primary rates come from
#: litellm's bundled snapshot.
RATES_VERIFIED_AT = "2026-05-17"

_USD_PER_M = 1_000_000

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPrice:
    usd_per_input_token: float
    usd_per_output_token: float
    usd_per_cached_input_token: float | None = None
    usd_per_cache_read_input_token: float | None = None
    usd_per_cache_creation_input_token: float | None = None

    @property
    def cached_input_rate(self) -> float:
        return (
            self.usd_per_cached_input_token
            if self.usd_per_cached_input_token is not None
            else self.cache_read_rate
        )

    @property
    def cache_read_rate(self) -> float:
        return (
            self.usd_per_cache_read_input_token
            if self.usd_per_cache_read_input_token is not None
            else self.usd_per_input_token
        )

    @property
    def cache_creation_rate(self) -> float:
        return (
            self.usd_per_cache_creation_input_token
            if self.usd_per_cache_creation_input_token is not None
            else self.usd_per_input_token
        )


@dataclass(frozen=True)
class PriceOverride:
    """Per-agent rate override loaded from ``agents.yaml``.

    Overrides are USD per 1M input/output tokens. Cache rates keep the
    base model's ratios when the model is known; for custom unknown
    models they fall back to the effective input rate.
    """

    input_usd_per_million: float | None = None
    output_usd_per_million: float | None = None


def _price(
    input_usd_per_million: float,
    output_usd_per_million: float,
    *,
    cache_read_usd_per_million: float | None = None,
    cache_write_usd_per_million: float | None = None,
) -> ModelPrice:
    input_rate = input_usd_per_million / _USD_PER_M
    cache_read_rate = (
        cache_read_usd_per_million / _USD_PER_M if cache_read_usd_per_million is not None else None
    )
    return ModelPrice(
        usd_per_input_token=input_rate,
        usd_per_output_token=output_usd_per_million / _USD_PER_M,
        usd_per_cached_input_token=cache_read_rate,
        usd_per_cache_read_input_token=cache_read_rate,
        usd_per_cache_creation_input_token=(
            cache_write_usd_per_million / _USD_PER_M
            if cache_write_usd_per_million is not None
            else None
        ),
    )


#: Fallback rates for model ids that LiteLLM's bundled snapshot does not carry
#: (values are USD per 1M tokens; ``_price`` converts to per-token). Everything
#: else is priced from that snapshot — see :func:`_litellm_price`. Keep this
#: list minimal: an id belongs here only while confirmed absent from the pinned
#: LiteLLM release (guarded by a reconciliation test). Re-verify on a bump.
MODEL_PRICES: dict[str, ModelPrice] = {
    # Legacy Claude 3.5 snapshots (dropped from LiteLLM's current map).
    "claude-3-5-sonnet-20240620": _price(
        3.00, 15.00, cache_read_usd_per_million=0.30, cache_write_usd_per_million=3.75
    ),
    "claude-3-5-sonnet-20241022": _price(
        3.00, 15.00, cache_read_usd_per_million=0.30, cache_write_usd_per_million=3.75
    ),
    "claude-3-5-haiku-20241022": _price(
        0.80, 4.00, cache_read_usd_per_million=0.08, cache_write_usd_per_million=1.00
    ),
    "claude-3-5-haiku-latest": _price(
        0.80, 4.00, cache_read_usd_per_million=0.08, cache_write_usd_per_million=1.00
    ),
    # Bare Anthropic aliases LiteLLM only keys under Bedrock-prefixed ids.
    "claude-sonnet-4": _price(
        3.00, 15.00, cache_read_usd_per_million=0.30, cache_write_usd_per_million=3.75
    ),
    "claude-sonnet-4-0": _price(
        3.00, 15.00, cache_read_usd_per_million=0.30, cache_write_usd_per_million=3.75
    ),
    "claude-opus-4": _price(
        15.00, 75.00, cache_read_usd_per_million=1.50, cache_write_usd_per_million=18.75
    ),
    "claude-opus-4-0": _price(
        15.00, 75.00, cache_read_usd_per_million=1.50, cache_write_usd_per_million=18.75
    ),
    # Recently GA'd models LiteLLM's bundled snapshot has not caught up on.
    "gpt-5.3-codex-spark": _price(1.75, 14.00, cache_read_usd_per_million=0.175),
    "gpt-5.6-luna": _price(1.00, 6.00, cache_read_usd_per_million=0.10),
    "gpt-5.6-sol": _price(5.00, 30.00, cache_read_usd_per_million=0.50),
    "gpt-5.6-terra": _price(2.50, 15.00, cache_read_usd_per_million=0.25),
}

# Only bare aliases whose *exact* id is not itself a key here. Deliberately
# NO broad ``claude-sonnet-4`` / ``claude-opus-4`` prefixes: those would
# collapse newer, differently-priced variants (e.g. ``claude-opus-4-5`` at
# 5/25) onto a legacy bare rate. Newer variants are priced by LiteLLM; only
# the bare legacy ids remain, and they resolve by direct lookup.
_UNSORTED_FAMILY_FALLBACKS: tuple[tuple[str, str], ...] = (
    ("claude-3-5-sonnet", "claude-3-5-sonnet-20241022"),
    ("claude-3-5-haiku", "claude-3-5-haiku-20241022"),
    # Per-tier rows so a suffixed variant (e.g. ``gpt-5.6-terra-preview``)
    # lands on its own tier, not on the shorter ``gpt-5.6`` → Sol alias.
    ("gpt-5.6-terra", "gpt-5.6-terra"),
    ("gpt-5.6-luna", "gpt-5.6-luna"),
    ("gpt-5.6-sol", "gpt-5.6-sol"),
    # OpenAI routes the bare `gpt-5.6` alias to Sol server-side.
    ("gpt-5.6", "gpt-5.6-sol"),
)

# Longest-prefix-first so more specific families win. Build it
# programmatically so a future edit cannot silently shadow a longer
# family with its shorter prefix.
_FAMILY_FALLBACKS: tuple[tuple[str, str], ...] = tuple(
    sorted(_UNSORTED_FAMILY_FALLBACKS, key=lambda item: len(item[0]), reverse=True)
)


def usd_for_usage(
    usage: TokenUsage,
    model: str | None,
    override: PriceOverride | None = None,
) -> float | None:
    """Return USD for a structured usage sample.

    Codex reports ``cached_input_tokens`` as a discounted subset of
    ``input_tokens``. If a future format reports cached input as a
    disjoint counter, clamp to the current convention and log at
    debug level instead of producing a negative non-cached input
    total.
    """
    price = _resolve_price(model, override)
    if price is None:
        return None

    input_tokens = max(0.0, usage.input_tokens)
    raw_cached_input_tokens = max(0.0, usage.cached_input_tokens)
    if raw_cached_input_tokens > input_tokens:
        logger.debug(
            "cached_input_tokens exceeded input_tokens; clamping to input total",
            extra={
                "model": model,
                "input_tokens": input_tokens,
                "cached_input_tokens": raw_cached_input_tokens,
            },
        )
    cached_input_tokens = min(raw_cached_input_tokens, input_tokens)
    non_cached_input_tokens = input_tokens - cached_input_tokens
    return (
        non_cached_input_tokens * price.usd_per_input_token
        + cached_input_tokens * price.cached_input_rate
        + max(0.0, usage.output_tokens) * price.usd_per_output_token
        + max(0.0, usage.cache_read_input_tokens) * price.cache_read_rate
        + max(0.0, usage.cache_creation_input_tokens) * price.cache_creation_rate
    )


def usd_per_hour_for_usage(
    usage_per_min: TokenUsage,
    model: str | None,
    override: PriceOverride | None = None,
) -> float | None:
    cost_per_min = usd_for_usage(usage_per_min, model, override)
    if cost_per_min is None:
        return None
    return cost_per_min * 60.0


def usd_per_token_blended(model: str | None, override: PriceOverride | None = None) -> float | None:
    price = _resolve_price(model, override)
    if price is None:
        return None
    return 0.7 * price.usd_per_input_token + 0.3 * price.usd_per_output_token


def usd_per_hour(
    tokens_per_min: float,
    model: str | None,
    override: PriceOverride | None = None,
) -> float | None:
    """Legacy blended API kept for callers/tests that only have a total."""
    rate = usd_per_token_blended(model, override)
    if rate is None:
        return None
    return tokens_per_min * 60.0 * rate


def normalize_model_name(model: str | None) -> str | None:
    """Return a clean canonical id for ``model`` for display and lookup.

    Prefers an id a price source recognizes: an exact local-table key, then the
    first non-vendor-qualified candidate LiteLLM prices (so a Bedrock id like
    ``us.anthropic.claude-sonnet-4-5-…-v1:0`` reduces to ``claude-sonnet-4-5``),
    then a family-prefix canonical, else the cleanest candidate.
    """
    if model is None:
        return None
    candidates = _model_candidates(model)
    for candidate in candidates:
        if candidate in MODEL_PRICES:
            return candidate
    for candidate in candidates:
        if _is_clean_id(candidate) and _litellm_prices_bare(candidate):
            return candidate
    for candidate in candidates:
        for prefix, canonical_id in _FAMILY_FALLBACKS:
            if candidate.startswith(prefix):
                return canonical_id
    for candidate in candidates:
        if _is_clean_id(candidate):
            return candidate
    return candidates[0] if candidates else None


def _resolve_price(model: str | None, override: PriceOverride | None) -> ModelPrice | None:
    base = _lookup_price(model) if model is not None else None
    if override is None:
        return base

    input_rate = (
        override.input_usd_per_million / _USD_PER_M
        if override.input_usd_per_million is not None
        else (base.usd_per_input_token if base is not None else None)
    )
    output_rate = (
        override.output_usd_per_million / _USD_PER_M
        if override.output_usd_per_million is not None
        else (base.usd_per_output_token if base is not None else None)
    )
    if input_rate is None or output_rate is None:
        return None

    return ModelPrice(
        usd_per_input_token=input_rate,
        usd_per_output_token=output_rate,
        usd_per_cached_input_token=_override_related_rate(
            input_rate,
            base.usd_per_cached_input_token if base is not None else None,
            base.usd_per_input_token if base is not None else None,
        ),
        usd_per_cache_read_input_token=_override_related_rate(
            input_rate,
            base.usd_per_cache_read_input_token if base is not None else None,
            base.usd_per_input_token if base is not None else None,
        ),
        usd_per_cache_creation_input_token=_override_related_rate(
            input_rate,
            base.usd_per_cache_creation_input_token if base is not None else None,
            base.usd_per_input_token if base is not None else None,
        ),
    )


def _override_related_rate(
    effective_input_rate: float,
    base_related_rate: float | None,
    base_input_rate: float | None,
) -> float | None:
    if base_related_rate is None or base_input_rate is None or base_input_rate == 0.0:
        return None
    return effective_input_rate * (base_related_rate / base_input_rate)


def _lookup_price(model: str) -> ModelPrice | None:
    litellm_price = _litellm_price(model)
    if litellm_price is not None:
        return litellm_price

    candidates = _model_candidates(model)
    for candidate in candidates:
        direct = MODEL_PRICES.get(candidate)
        if direct is not None:
            return direct
    for candidate in candidates:
        for prefix, canonical_id in _FAMILY_FALLBACKS:
            if candidate.startswith(prefix):
                resolved = MODEL_PRICES.get(canonical_id)
                if resolved is not None:
                    return resolved
    return None


#: Prefixes LiteLLM keys the same model under: bare, provider-namespaced, and
#: Bedrock/region-prefixed. Probed in order against each normalized candidate.
_LITELLM_KEY_PREFIXES: tuple[str, ...] = (
    "",
    "anthropic/",
    "anthropic.",
    "us.anthropic.",
    "eu.anthropic.",
    "openai/",
)

#: Providers that report a *bare* model id (via openai_compat_providers) while
#: LiteLLM only keys them under ``<provider>/``. Probed as a last resort after a
#: bare miss — kept narrow deliberately: guessing a prefix for an arbitrary
#: open-weight model risks applying a different host's price for the same name.
_COMPAT_PROVIDER_PREFIXES: tuple[str, ...] = ("groq/", "minimax/")

#: Case-folded snapshot (``lower(key) -> entry``). ``None`` until first read;
#: ``{}`` if the snapshot is unavailable, so lookups stop retrying the read.
_litellm_cost_map: dict[str, dict] | None = None


def _is_number(value: object) -> TypeGuard[float]:
    """True for a real int/float rate — ``bool`` is rejected (``True`` is an int)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


#: A candidate that is not a clean canonical display id: a provider/region
#: namespace (``openai/…``, ``anthropic.…``, ``us.anthropic.…``), a Bedrock
#: version marker (``-v1:0``), or an ``@`` variant tag.
_NOT_CLEAN_ID = re.compile(r"^([a-z0-9-]+\.)?anthropic\.|/|-v\d+:\d+$|@")


def _is_clean_id(candidate: str) -> bool:
    return not _NOT_CLEAN_ID.search(candidate)


def _litellm_local_cost_map() -> dict[str, dict]:
    """Return LiteLLM's bundled price snapshot, case-folded (``lower(key) -> entry``).

    Reads LiteLLM's packaged ``model_prices_and_context_window_backup.json``
    directly rather than the process-wide ``litellm.model_cost`` global: that
    global is populated by a live network fetch on whichever import touches it
    first, so depending on it would make pricing depend on unrelated import
    order elsewhere in the process. Reading the file is deterministic and
    offline. Keys are lower-cased because LiteLLM's own keys are not uniformly
    cased (e.g. ``minimax/MiniMax-M2.1``).

    Never raises: a missing/unreadable file or non-dict payload is cached as
    ``{}`` so the always-on sampler stops re-attempting the read on every tick.
    """
    global _litellm_cost_map
    if _litellm_cost_map is None:
        try:
            ensure_tiktoken_encodings_discoverable()
            raw = json.loads(
                files("litellm")
                .joinpath(_LITELLM_PRICE_SNAPSHOT_FILENAME)
                .read_text(encoding="utf-8")
            )
            _litellm_cost_map = (
                {k.lower(): v for k, v in raw.items() if isinstance(v, dict)}
                if isinstance(raw, dict)
                else {}
            )
        except Exception:  # noqa: BLE001 - defensive: missing/renamed snapshot
            logger.debug("litellm price snapshot unavailable; disabling it", exc_info=True)
            _litellm_cost_map = {}
    return _litellm_cost_map


def warm_cost_map() -> None:
    """Eagerly load LiteLLM's price snapshot, paying the one-time ``import litellm``.

    Call this off the UI/render thread (e.g. from the sampler's background loop)
    so the first ``$/hr`` price lookup does not pay the import cost synchronously
    on the interactive-shell render path.
    """
    _litellm_local_cost_map()


def _priced_entry(entry: object) -> TypeGuard[dict]:
    """True if ``entry`` is a LiteLLM row carrying both input and output rates."""
    return (
        isinstance(entry, dict)
        and _is_number(entry.get("input_cost_per_token"))
        and _is_number(entry.get("output_cost_per_token"))
    )


def _litellm_prices_bare(candidate: str) -> bool:
    """True if LiteLLM prices ``candidate`` under any of its keying conventions.

    Requires both input and output rates, matching :func:`_litellm_price`'s
    notion of "priced" so ``normalize_model_name`` never canonicalizes onto an
    id ``usd_for_usage`` then treats as unpriced.
    """
    cost_map = _litellm_local_cost_map()
    return any(
        _priced_entry(cost_map.get(prefix + candidate))
        for prefix in (*_LITELLM_KEY_PREFIXES, *_COMPAT_PROVIDER_PREFIXES)
    )


def _litellm_candidates(model: str) -> Iterator[str]:
    """Yield candidate snapshot keys for ``model`` (lower-cased) without repeats.

    Primary keying conventions first; the narrow compat-provider prefixes only
    after, so a bare hit always wins over a guessed ``<provider>/`` match.
    """
    seen: set[str] = set()
    bases = _model_candidates(model)
    for prefixes in (_LITELLM_KEY_PREFIXES, _COMPAT_PROVIDER_PREFIXES):
        for base in bases:
            for prefix in prefixes:
                key = prefix + base
                if key not in seen:
                    seen.add(key)
                    yield key


def _litellm_price(model: str) -> ModelPrice | None:
    """Build a :class:`ModelPrice` from LiteLLM's snapshot, or ``None`` on miss.

    A miss (or the snapshot being unavailable, in which case the map is empty)
    returns ``None`` so the caller falls back to :data:`MODEL_PRICES` and
    ultimately to unpriced — never a fabricated ``$0``.
    """
    cost_map = _litellm_local_cost_map()
    for key in _litellm_candidates(model):
        entry = cost_map.get(key)
        if not _priced_entry(entry):
            continue
        cache_read = entry.get("cache_read_input_token_cost")
        cache_creation = entry.get("cache_creation_input_token_cost")
        return ModelPrice(
            usd_per_input_token=float(entry["input_cost_per_token"]),
            usd_per_output_token=float(entry["output_cost_per_token"]),
            usd_per_cached_input_token=(float(cache_read) if _is_number(cache_read) else None),
            usd_per_cache_read_input_token=(float(cache_read) if _is_number(cache_read) else None),
            usd_per_cache_creation_input_token=(
                float(cache_creation) if _is_number(cache_creation) else None
            ),
        )
    return None


@lru_cache(maxsize=512)
def _model_candidates(raw: str) -> tuple[str, ...]:
    candidates: list[str] = []

    def append(value: str) -> None:
        normalized = value.strip().lower()
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    trimmed = raw.strip()
    append(trimmed)
    lower = trimmed.lower()
    for prefix in ("openai/", "anthropic/", "anthropic."):
        if lower.startswith(prefix):
            append(trimmed[len(prefix) :])

    if "claude-" in lower and "." in trimmed:
        tail = trimmed.rsplit(".", maxsplit=1)[-1]
        if tail.lower().startswith("claude-"):
            append(tail)

    index = 0
    while index < len(candidates):
        candidate = candidates[index]
        if "@" in candidate:
            base, suffix = candidate.split("@", maxsplit=1)
            append(base)
            if re.fullmatch(r"\d{8}", suffix):
                append(f"{base}-{suffix}")
        elif candidate.startswith("claude-"):
            append(f"{candidate}@default")

        for pattern in (r"-\d{4}-\d{2}-\d{2}$", r"-\d{8}$", r"-v\d+:\d+$"):
            stripped = re.sub(pattern, "", candidate)
            if stripped != candidate:
                append(stripped)
        index += 1

    return tuple(candidates)


__all__ = [
    "MODEL_PRICES",
    "ModelPrice",
    "PriceOverride",
    "RATES_VERIFIED_AT",
    "normalize_model_name",
    "usd_for_usage",
    "usd_per_hour",
    "usd_per_hour_for_usage",
    "usd_per_token_blended",
    "warm_cost_map",
]
