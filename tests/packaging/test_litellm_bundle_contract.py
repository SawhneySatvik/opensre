"""Regression tests for LiteLLM package data in frozen release binaries.

``litellm`` is imported unconditionally now that
``tools/system/fleet_monitoring/pricing.py`` sources model rates from
``litellm.model_cost`` (#4035) — the always-on dashboard sampler imports it, on
top of the Azure OpenAI / ``OPENSRE_LLM_TRANSPORT=litellm`` transport paths.
LiteLLM reads JSON price/context files from its package directory on import, and
``pricing.py`` reads its ``model_prices_and_context_window_backup.json`` for the
offline cost map; the release PyInstaller build must bundle them under
``_internal/litellm/`` (via ``--collect-data litellm``) or the binary crashes
with ``FileNotFoundError`` (see issue #3631).
"""

from __future__ import annotations

from pathlib import Path

# LiteLLM loads this file during import; it must be present in frozen bundles.
_REQUIRED_LITELLM_DATA_FILE = "model_prices_and_context_window_backup.json"


def test_litellm_package_ships_required_price_context_backup() -> None:
    """Dev installs must still expose LiteLLM's price/context JSON (sanity check)."""
    import litellm

    data_path = Path(litellm.__file__).parent / _REQUIRED_LITELLM_DATA_FILE
    assert data_path.is_file(), f"expected LiteLLM data file at {data_path}"
