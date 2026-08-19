"""Unit tests for agent-tier sampling knobs and their per-transport application."""

from __future__ import annotations

import types
from typing import Any

import pytest

from config.llm_sampling import (
    AGENT_SEED_ENV,
    AGENT_TEMPERATURE_ENV,
    get_agent_seed,
    get_agent_temperature,
)
from core.llm.transports.litellm.clients import LiteLLMAgentClient
from core.llm.transports.sdk.agent_clients import OpenAIAgentClient


@pytest.fixture(autouse=True)
def _clear_sampling_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(AGENT_TEMPERATURE_ENV, raising=False)
    monkeypatch.delenv(AGENT_SEED_ENV, raising=False)


def test_defaults_are_deterministic() -> None:
    assert get_agent_temperature() == 0.0
    assert get_agent_seed() == 0


@pytest.mark.parametrize("value", ["none", "default", " DEFAULT "])
def test_opt_out_values_defer_to_the_provider(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(AGENT_TEMPERATURE_ENV, value)
    monkeypatch.setenv(AGENT_SEED_ENV, value)
    assert get_agent_temperature() is None
    assert get_agent_seed() is None


def test_env_overrides_are_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(AGENT_TEMPERATURE_ENV, "0.7")
    monkeypatch.setenv(AGENT_SEED_ENV, "42")
    assert get_agent_temperature() == 0.7
    assert get_agent_seed() == 42


@pytest.mark.parametrize("bad", ["hot", "nan", "inf", "-inf"])
def test_unusable_values_keep_determinism(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    """A typo must not silently restore provider sampling.

    nan/inf parse as floats but are not JSON-serializable, so letting them
    through would fail inside the provider SDK instead of here.
    """
    monkeypatch.setenv(AGENT_TEMPERATURE_ENV, bad)
    assert get_agent_temperature() == 0.0


def test_unparseable_seed_keeps_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(AGENT_SEED_ENV, "0.5")
    assert get_agent_seed() == 0


def _fake_completion_response() -> Any:
    message = types.SimpleNamespace(content="ok", tool_calls=[])
    message.model_dump = lambda **_: {"role": "assistant", "content": "ok"}
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=message, finish_reason="stop")], usage=None
    )


def _sdk_sampling(model: str, api_key_env: str, pinned: float | None) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _fake_completion_response()

    client = OpenAIAgentClient.__new__(OpenAIAgentClient)
    client._client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )
    client._model = model
    client._max_tokens = 1024
    client._api_key_env = api_key_env
    client._temperature = pinned
    client.invoke(messages=[{"role": "user", "content": "hi"}])
    return {k: v for k, v in captured.items() if k in ("temperature", "seed")}


def _litellm_sampling(model: str, api_key_env: str, pinned: float | None) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def completion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _fake_completion_response()

    LiteLLMAgentClient(
        litellm_model=f"openai/{model}",
        api_key_env=api_key_env,
        credential_resolver=lambda _env: "sk-key",
        temperature=pinned,
        completion_func=completion,
    ).invoke([{"role": "user", "content": "hi"}])
    return {k: v for k, v in captured.items() if k in ("temperature", "seed")}


_SAMPLING_CASES = [
    ("gpt-4o", "OPENAI_API_KEY", None, {"temperature": 0.0, "seed": 0}),
    ("gpt-5.4-mini", "OPENAI_API_KEY", None, {"seed": 0}),
    ("o3", "OPENAI_API_KEY", None, {"seed": 0}),
    ("deepseek-v4", "DEEPSEEK_API_KEY", None, {"temperature": 0.0}),
    ("MiniMax-M2", "MINIMAX_API_KEY", 1.0, {"temperature": 1.0}),
]


@pytest.mark.parametrize(("model", "api_key_env", "pinned", "expected"), _SAMPLING_CASES)
def test_both_transports_send_the_same_sampling_params(
    model: str, api_key_env: str, pinned: float | None, expected: dict[str, Any]
) -> None:
    """The SDK and LiteLLM routes must not disagree about one model's capabilities.

    They did, twice. Only the SDK knew o-series reject ``temperature``, so routing
    the agent tier through LiteLLM raised ``UnsupportedParamsError`` before the
    request left the process; and only LiteLLM honoured a provider-pinned
    temperature, so MiniMax lost its required 1.0 on the default transport.

    ``gpt-5.4-mini`` is the shipped OpenAI/Azure default, so its row is the one
    most users actually exercise.
    """
    assert _sdk_sampling(model, api_key_env, pinned) == expected
    assert _litellm_sampling(model, api_key_env, pinned) == expected
