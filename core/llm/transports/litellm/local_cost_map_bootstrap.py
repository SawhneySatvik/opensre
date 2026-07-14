"""Keep LiteLLM's import-time price-map load offline (no network fetch).

On import, ``litellm`` populates its ``litellm.model_cost`` global via
``get_model_cost_map(url=...)`` — a live ``httpx.get`` against a GitHub-hosted
price map unless ``LITELLM_LOCAL_MODEL_COST_MAP`` is ``"true"``. ``litellm`` is
now imported unconditionally (the always-on dashboard sampler reaches it through
``tools/system/fleet_monitoring/pricing.py``), so an un-pinned import would do a
network round-trip at startup.

``pricing.py`` reads LiteLLM's bundled JSON snapshot *directly* for its rates and
does not depend on that global, so this is purely an optimization: calling
:func:`ensure_local_model_cost_map` before the first ``import litellm`` avoids
the wasted fetch. ``setdefault`` lets an operator opt back into the remote map
with ``LITELLM_LOCAL_MODEL_COST_MAP=false``.
"""

from __future__ import annotations

import os

_ENV_VAR = "LITELLM_LOCAL_MODEL_COST_MAP"


def ensure_local_model_cost_map() -> None:
    """Pin LiteLLM's import-time cost-map load to offline (idempotent).

    Must run before the first ``import litellm`` in the process to take effect.
    """
    os.environ.setdefault(_ENV_VAR, "True")
