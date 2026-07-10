"""Tests for model_spend_tool (issue #3231)."""

from __future__ import annotations

from tests.tools.conftest import BaseToolContract
from tools.system.fleet_monitoring.meters import TokenUsage
from tools.system.fleet_monitoring.pricing import usd_for_usage
from tools.system.model_spend_tool import estimate_model_spend, summarize_model_burn_rate

_KNOWN = "claude-sonnet-4-6"
_UNKNOWN = "totally-made-up-model-xyz"


class TestEstimateModelSpendContract(BaseToolContract):
    def get_tool_under_test(self):  # type: ignore[no-untyped-def]
        return estimate_model_spend.__opensre_registered_tool__


class TestSummarizeBurnRateContract(BaseToolContract):
    def get_tool_under_test(self):  # type: ignore[no-untyped-def]
        return summarize_model_burn_rate.__opensre_registered_tool__


def test_known_model_pricing_matches_price_table() -> None:
    result = estimate_model_spend({_KNOWN: {"input_tokens": 1000, "output_tokens": 500}})
    expected = usd_for_usage(TokenUsage(input_tokens=1000, output_tokens=500), _KNOWN)
    assert expected is not None
    assert result["priced"] is True
    assert result["total_usd"] == round(float(expected), 6)
    assert result["by_model"][0]["model"] == _KNOWN
    assert result["by_model"][0]["priced"] is True


def test_unknown_model_is_unpriced_not_zero() -> None:
    result = estimate_model_spend({_UNKNOWN: {"input_tokens": 1000}})
    assert result["priced"] is False
    assert result["total_usd"] == 0.0
    assert result["unpriced_models"] == [_UNKNOWN]
    assert result["by_model"][0]["priced"] is False
    assert result["by_model"][0]["usd"] is None


def test_bucket_costs_sum_to_model_total() -> None:
    usage = {
        "input_tokens": 2000,
        "output_tokens": 800,
        "cache_read_input_tokens": 500,
        "cache_creation_input_tokens": 100,
    }
    entry = estimate_model_spend({_KNOWN: usage})["by_model"][0]
    assert round(sum(entry["cost_by_bucket"].values()), 6) == entry["usd"]


def test_usd_per_lap_divides_total_by_laps() -> None:
    result = estimate_model_spend({_KNOWN: {"input_tokens": 1000, "output_tokens": 500}}, laps=5)
    assert result["laps"] == 5
    assert result["usd_per_lap"] == round(result["total_usd"] / 5, 6)


def test_usd_per_lap_none_when_laps_zero_or_missing() -> None:
    assert estimate_model_spend({_KNOWN: {"input_tokens": 1000}}, laps=0)["usd_per_lap"] is None
    assert estimate_model_spend({_KNOWN: {"input_tokens": 1000}})["usd_per_lap"] is None


def test_price_override_prices_unknown_model() -> None:
    # 1M input tokens at an overridden $2/1M input rate = $2.00.
    result = estimate_model_spend(
        {"custom-model": {"input_tokens": 1_000_000, "output_tokens": 0}},
        price_overrides={
            "custom-model": {"input_usd_per_million": 2.0, "output_usd_per_million": 8.0}
        },
    )
    assert result["priced"] is True
    assert result["total_usd"] == 2.0


def test_multiple_models_aggregate_only_priced() -> None:
    mixed = estimate_model_spend(
        {
            _KNOWN: {"input_tokens": 1000, "output_tokens": 500},
            _UNKNOWN: {"input_tokens": 1000},
        }
    )
    known_only = estimate_model_spend({_KNOWN: {"input_tokens": 1000, "output_tokens": 500}})
    assert mixed["total_usd"] == known_only["total_usd"]
    assert _UNKNOWN in mixed["unpriced_models"]
    assert len(mixed["by_model"]) == 2


def test_burn_rate_per_minute_projection() -> None:
    result = summarize_model_burn_rate({_KNOWN: {"input_tokens": 1000, "output_tokens": 500}})
    per_min_cost = usd_for_usage(TokenUsage(input_tokens=1000, output_tokens=500), _KNOWN)
    assert per_min_cost is not None
    assert result["usage_interpreted_as"] == "per_minute"
    assert result["total_usd_per_hour"] == round(float(per_min_cost) * 60.0, 6)


def test_burn_rate_from_totals_over_elapsed_seconds() -> None:
    # 120s window -> per-minute halves the totals; $/hr = per-min cost * 60.
    result = summarize_model_burn_rate(
        {_KNOWN: {"input_tokens": 1000, "output_tokens": 500}}, elapsed_seconds=120.0
    )
    expected = usd_for_usage(TokenUsage(input_tokens=500, output_tokens=250), _KNOWN)
    assert expected is not None
    assert result["usage_interpreted_as"] == "totals_over_elapsed_seconds"
    assert result["total_usd_per_hour"] == round(float(expected) * 60.0, 6)


def test_burn_rate_unknown_model_unpriced() -> None:
    result = summarize_model_burn_rate({_UNKNOWN: {"input_tokens": 1000}})
    assert result["priced"] is False
    assert result["unpriced_models"] == [_UNKNOWN]


def test_bool_token_value_is_not_counted() -> None:
    # isinstance(True, int) is True, so a stray boolean must not add a token
    # (mirrors meters.safe_int). Otherwise "input_tokens": true would cost money.
    result = estimate_model_spend({_KNOWN: {"input_tokens": True, "output_tokens": 500}})
    assert result["by_model"][0]["tokens"]["input_tokens"] == 0.0
    expected = usd_for_usage(TokenUsage(output_tokens=500), _KNOWN)
    assert expected is not None
    assert result["total_usd"] == round(float(expected), 6)


def test_non_numeric_string_token_value_falls_back_to_zero() -> None:
    # A non-numeric string must not raise; it is treated as zero tokens.
    result = estimate_model_spend({_KNOWN: {"input_tokens": "1,000", "output_tokens": 500}})
    assert result["by_model"][0]["tokens"]["input_tokens"] == 0.0
    # A clean numeric string is still parsed.
    parsed = estimate_model_spend({_KNOWN: {"input_tokens": "1000"}})
    assert parsed["by_model"][0]["tokens"]["input_tokens"] == 1000.0


def test_non_mapping_usage_value_does_not_crash() -> None:
    # A bare total (a common LLM shorthand) is not a bucket dict; it must be
    # tolerated as empty usage rather than raising AttributeError mid-batch.
    result = estimate_model_spend({_KNOWN: 50000})  # type: ignore[dict-item]
    assert result["by_model"][0]["tokens"]["input_tokens"] == 0.0
    assert result["total_usd"] == 0.0


def test_non_positive_elapsed_seconds_is_consistent_per_minute() -> None:
    # A zero/negative window cannot be projected: fall back to per-minute and
    # report elapsed_seconds as null so the output is never self-contradictory.
    for bad in (0.0, -30.0):
        result = summarize_model_burn_rate({_KNOWN: {"input_tokens": 1000}}, elapsed_seconds=bad)
        assert result["usage_interpreted_as"] == "per_minute"
        assert result["elapsed_seconds"] is None
    baseline = summarize_model_burn_rate({_KNOWN: {"input_tokens": 1000}})
    zero = summarize_model_burn_rate({_KNOWN: {"input_tokens": 1000}}, elapsed_seconds=0.0)
    assert zero["total_usd_per_hour"] == baseline["total_usd_per_hour"]


def test_tool_sources_are_knowledge() -> None:
    for fn in (estimate_model_spend, summarize_model_burn_rate):
        assert fn.__opensre_registered_tool__.source == "knowledge"
