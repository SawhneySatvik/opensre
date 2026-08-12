"""Dispatch a background-RCA completion notice to the channels a user chose.

The caller must have run ``bootstrap.adapters.install_notification_adapters``
first. This module cannot: ``bootstrap`` sits above ``platform`` in the layering
contract.
"""

from __future__ import annotations

from platform.common.background_investigation_types import BackgroundInvestigationRecord
from platform.notifications.outbound_registry import BACKGROUND_RCA, get_outbound_adapter


def dispatch_background_notifications(
    *,
    record: BackgroundInvestigationRecord,
    channels: tuple[str, ...],
) -> dict[str, str]:
    """Deliver ``record`` to each channel in ``channels``; return per-channel outcomes.

    Iterates the caller's tuple, not the registry, so the mapping keeps the order
    the user configured; ``/background show`` renders it in that order. Adapter
    exceptions propagate, matching the ``if channel == ...`` chain this replaces.
    """
    results: dict[str, str] = {}
    for channel in channels:
        adapter = get_outbound_adapter(channel)
        if adapter is None or BACKGROUND_RCA not in adapter.capabilities:
            results[channel] = "unsupported"
            continue
        results[channel] = adapter.deliver(record)
    return results


__all__ = ["dispatch_background_notifications"]
