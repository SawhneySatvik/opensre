"""Pure cost / burn-rate estimation over token usage — no external calls.

Reuses the fleet-monitoring price table (``tools.system.fleet_monitoring.pricing``)
so estimates stay consistent with the rest of the codebase. Estimates come only
from tokens x the local price table; this never claims a live provider credit
balance (there is no billing API in the repo). Unknown models are surfaced as
``priced: false`` rather than silently costed at zero.
"""

from __future__ import annotations

from typing import Any

from tools.system.fleet_monitoring.meters import TokenUsage
from tools.system.fleet_monitoring.pricing import (
    RATES_VERIFIED_AT,
    PriceOverride,
    normalize_model_name,
    usd_for_usage,
    usd_per_hour_for_usage,
)

_BUCKET_KEYS = ("input", "cached_input", "output", "cache_read", "cache_creation")


def _token_usage_from_mapping(raw: Any) -> TokenUsage:
    """Build a clamped :class:`TokenUsage` from a loose token-bucket dict.

    Tolerant of malformed input: a non-mapping value is treated as empty usage,
    and each bucket is coerced defensively (``bool`` and non-numeric strings
    become zero, so a stray ``"input_tokens": true`` never adds a token).
    """
    data = raw if isinstance(raw, dict) else {}

    def _num(key: str) -> float:
        value = data.get(key)
        if value is None or isinstance(value, bool):
            return 0.0
        if not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return 0.0
        return float(value) if value > 0 else 0.0

    return TokenUsage(
        input_tokens=_num("input_tokens"),
        output_tokens=_num("output_tokens"),
        cached_input_tokens=_num("cached_input_tokens"),
        cache_read_input_tokens=_num("cache_read_input_tokens"),
        cache_creation_input_tokens=_num("cache_creation_input_tokens"),
    ).clamped()


def _price_override_from_mapping(raw: dict[str, Any] | None) -> PriceOverride | None:
    if not raw:
        return None
    inp = raw.get("input_usd_per_million")
    out = raw.get("output_usd_per_million")
    if inp is None and out is None:
        return None
    return PriceOverride(
        input_usd_per_million=None if inp is None else float(inp),
        output_usd_per_million=None if out is None else float(out),
    )


def _bucket_costs(
    usage: TokenUsage, model: str, override: PriceOverride | None
) -> dict[str, float] | None:
    """Per-bucket USD for one model, or ``None`` when the model has no price.

    ``usd_for_usage`` is linear in each bucket, so a single-bucket
    :class:`TokenUsage` yields exactly that bucket's cost. This reuses the
    shared price table instead of duplicating rate math, and the buckets sum
    to the model total.
    """
    cached = min(usage.cached_input_tokens, usage.input_tokens)
    non_cached_input = usage.input_tokens - cached
    parts: dict[str, float | None] = {
        "input": usd_for_usage(TokenUsage(input_tokens=non_cached_input), model, override),
        "cached_input": usd_for_usage(
            TokenUsage(input_tokens=cached, cached_input_tokens=cached), model, override
        ),
        "output": usd_for_usage(TokenUsage(output_tokens=usage.output_tokens), model, override),
        "cache_read": usd_for_usage(
            TokenUsage(cache_read_input_tokens=usage.cache_read_input_tokens), model, override
        ),
        "cache_creation": usd_for_usage(
            TokenUsage(cache_creation_input_tokens=usage.cache_creation_input_tokens),
            model,
            override,
        ),
    }
    result: dict[str, float] = {}
    for key in _BUCKET_KEYS:
        value = parts[key]
        if value is None:
            return None
        result[key] = float(value)
    return result


def _tokens_dict(usage: TokenUsage) -> dict[str, float]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens,
    }


def estimate_spend(
    token_usage_by_model: dict[str, Any],
    *,
    laps: int | None = None,
    run_label: str | None = None,
    price_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate total USD spend from per-model token usage.

    Returns a breakdown by model, by token bucket, and (when ``laps`` is
    provided) a cost-per-lap driver. Unknown models are listed under
    ``unpriced_models`` and never contribute to the total.
    """
    overrides = price_overrides or {}
    by_model: list[dict[str, Any]] = []
    unpriced: list[str] = []
    total_usd = 0.0
    priced_any = False
    bucket_totals = dict.fromkeys(_BUCKET_KEYS, 0.0)

    for model, raw_usage in token_usage_by_model.items():
        usage = _token_usage_from_mapping(raw_usage or {})
        override = _price_override_from_mapping(overrides.get(model))
        usd = usd_for_usage(usage, model, override)
        entry: dict[str, Any] = {
            "model": model,
            "normalized_model": normalize_model_name(model),
            "priced": usd is not None,
            "usd": None if usd is None else round(float(usd), 6),
            "tokens": _tokens_dict(usage),
        }
        if usd is None:
            unpriced.append(model)
        else:
            priced_any = True
            total_usd += float(usd)
            buckets = _bucket_costs(usage, model, override)
            if buckets is not None:
                entry["cost_by_bucket"] = {k: round(v, 6) for k, v in buckets.items()}
                for key, value in buckets.items():
                    bucket_totals[key] += value
        by_model.append(entry)

    usd_per_lap = None
    if laps is not None and laps > 0 and priced_any:
        usd_per_lap = round(total_usd / laps, 6)

    return {
        "run_label": run_label,
        "rates_verified_at": RATES_VERIFIED_AT,
        "priced": priced_any,
        "total_usd": round(total_usd, 6),
        "laps": laps,
        "usd_per_lap": usd_per_lap,
        "cost_by_bucket": {k: round(v, 6) for k, v in bucket_totals.items()},
        "by_model": by_model,
        "unpriced_models": unpriced,
        "note": (
            f"Estimated from tokens x local price table (verified {RATES_VERIFIED_AT}); "
            "does not reflect a live provider credit balance."
        ),
    }


def summarize_burn_rate(
    token_usage_by_model: dict[str, Any],
    *,
    elapsed_seconds: float | None = None,
    run_label: str | None = None,
    price_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project hourly burn ($/hr) from per-model token usage.

    When ``elapsed_seconds`` is given, ``token_usage_by_model`` is treated as
    totals observed over that window and normalized to per-minute before
    projecting to an hour. Otherwise the usage is treated as a per-minute
    sample (matching ``usd_per_hour_for_usage``).
    """
    overrides = price_overrides or {}
    by_model: list[dict[str, Any]] = []
    unpriced: list[str] = []
    total_usd_per_hour = 0.0
    priced_any = False

    # A non-positive window can't be projected; fall back to per-minute and report it as null.
    effective_elapsed = (
        elapsed_seconds if (elapsed_seconds is not None and elapsed_seconds > 0) else None
    )
    scale = 60.0 / effective_elapsed if effective_elapsed is not None else None

    for model, raw_usage in token_usage_by_model.items():
        usage = _token_usage_from_mapping(raw_usage or {})
        per_minute = usage.scaled(scale) if scale is not None else usage
        override = _price_override_from_mapping(overrides.get(model))
        usd_per_hour = usd_per_hour_for_usage(per_minute, model, override)
        entry = {
            "model": model,
            "normalized_model": normalize_model_name(model),
            "priced": usd_per_hour is not None,
            "usd_per_hour": None if usd_per_hour is None else round(float(usd_per_hour), 6),
        }
        if usd_per_hour is None:
            unpriced.append(model)
        else:
            priced_any = True
            total_usd_per_hour += float(usd_per_hour)
        by_model.append(entry)

    return {
        "run_label": run_label,
        "rates_verified_at": RATES_VERIFIED_AT,
        "priced": priced_any,
        "elapsed_seconds": effective_elapsed,
        "usage_interpreted_as": ("per_minute" if scale is None else "totals_over_elapsed_seconds"),
        "total_usd_per_hour": round(total_usd_per_hour, 6),
        "by_model": by_model,
        "unpriced_models": unpriced,
        "note": (
            "Projected burn from tokens x local price table; does not reflect a "
            "live provider credit balance."
        ),
    }


__all__ = ["estimate_spend", "summarize_burn_rate"]
