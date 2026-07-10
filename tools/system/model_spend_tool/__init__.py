"""Model spend + burn-rate estimation tools (issue #3231).

Pure calculators over token/model/lap metadata that reuse the fleet-monitoring
price table. They never call a provider billing API — estimates come only from
tokens x the local price table, and unknown models are surfaced as unpriced.
"""

from __future__ import annotations

from typing import Any

from core.tool_framework.tool_decorator import tool
from tools.system.model_spend_tool.estimation import estimate_spend, summarize_burn_rate

_TOKEN_USAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "Map of model name -> token buckets. Provide at least input_tokens and "
        "output_tokens; cache buckets are optional but improve accuracy."
    ),
    "additionalProperties": {
        "type": "object",
        "properties": {
            "input_tokens": {"type": "number"},
            "output_tokens": {"type": "number"},
            "cached_input_tokens": {"type": "number"},
            "cache_read_input_tokens": {"type": "number"},
            "cache_creation_input_tokens": {"type": "number"},
        },
    },
}

_PRICE_OVERRIDES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "Optional per-model rate overrides, in USD per 1M tokens.",
    "additionalProperties": {
        "type": "object",
        "properties": {
            "input_usd_per_million": {"type": "number"},
            "output_usd_per_million": {"type": "number"},
        },
    },
}


def _no_state_params(_sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    # Inputs are supplied by the caller/planner, not drawn from evidence state.
    return {}


@tool(
    name="estimate_model_spend",
    display_name="Model spend estimate",
    source="knowledge",
    description="Estimate AI spend (USD) from per-model token usage, with a per-lap breakdown.",
    use_cases=[
        "Estimating the USD cost of a benchmark or investigation run from token usage",
        "Explaining cost drivers: model choice, token buckets, and cost per investigation lap",
        "Comparing spend across models for the same workload",
    ],
    tags=("safe", "fast", "no-credentials"),
    surfaces=("investigation", "chat"),
    input_schema={
        "type": "object",
        "properties": {
            "token_usage_by_model": _TOKEN_USAGE_SCHEMA,
            "laps": {
                "type": "integer",
                "description": "Investigation loop iterations, used to derive cost per lap.",
            },
            "run_label": {"type": "string", "description": "Optional label for the run."},
            "price_overrides": _PRICE_OVERRIDES_SCHEMA,
        },
        "required": ["token_usage_by_model"],
    },
    extract_params=_no_state_params,
)
def estimate_model_spend(
    token_usage_by_model: dict[str, Any],
    laps: int | None = None,
    run_label: str | None = None,
    price_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate USD spend from per-model token usage."""
    return estimate_spend(
        token_usage_by_model,
        laps=laps,
        run_label=run_label,
        price_overrides=price_overrides,
    )


@tool(
    name="summarize_model_burn_rate",
    display_name="Model burn rate",
    source="knowledge",
    description="Project hourly AI burn ($/hr) from per-model token usage.",
    use_cases=[
        "Projecting $/hr burn from a per-minute token sample",
        "Projecting hourly burn from run totals plus elapsed wall-clock seconds",
        "Flagging which model dominates ongoing spend",
    ],
    tags=("safe", "fast", "no-credentials"),
    surfaces=("investigation", "chat"),
    input_schema={
        "type": "object",
        "properties": {
            "token_usage_by_model": _TOKEN_USAGE_SCHEMA,
            "elapsed_seconds": {
                "type": "number",
                "description": (
                    "If set, token usage is treated as totals over this window and "
                    "normalized to per-minute before projecting to an hour."
                ),
            },
            "run_label": {"type": "string", "description": "Optional label for the run."},
            "price_overrides": _PRICE_OVERRIDES_SCHEMA,
        },
        "required": ["token_usage_by_model"],
    },
    extract_params=_no_state_params,
)
def summarize_model_burn_rate(
    token_usage_by_model: dict[str, Any],
    elapsed_seconds: float | None = None,
    run_label: str | None = None,
    price_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project hourly burn from per-model token usage."""
    return summarize_burn_rate(
        token_usage_by_model,
        elapsed_seconds=elapsed_seconds,
        run_label=run_label,
        price_overrides=price_overrides,
    )


__all__ = ["estimate_model_spend", "summarize_model_burn_rate"]
