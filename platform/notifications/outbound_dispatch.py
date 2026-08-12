"""Dispatch a background-RCA completion notice to the channels a user chose.

Resolves one registered adapter per requested channel. Vendor-free by
construction: everything channel-specific lives in the adapters, which register
themselves from ``integrations/``.

The caller is responsible for having run the registration bootstrap first
(``bootstrap.adapters.install_notification_adapters``). This module cannot do it
itself: ``bootstrap`` sits above ``platform`` in the layering contract.
"""

from __future__ import annotations

from platform.common.background_investigation_types import BackgroundInvestigationRecord
from platform.notifications.outbound_registry import (
    BACKGROUND_RCA,
    get_outbound_adapter,
    outbound_adapter_names_for,
)


def dispatch_background_notifications(
    *,
    record: BackgroundInvestigationRecord,
    channels: tuple[str, ...],
) -> dict[str, str]:
    """Deliver ``record`` to each channel in ``channels``; return per-channel outcomes.

    Iterates the caller's tuple rather than the registry, so the returned
    mapping preserves the order the user configured. ``/background show``
    renders it in that order.

    Exceptions from an adapter propagate, matching the behaviour of the
    ``if channel == ...`` chain this replaces.
    """
    results: dict[str, str] = {}
    for channel in channels:
        adapter = get_outbound_adapter(channel)
        if adapter is None or BACKGROUND_RCA not in adapter.capabilities:
            results[channel] = "unsupported"
            continue
        results[channel] = adapter.deliver(record)
    return results


def supported_notification_channels() -> tuple[str, ...]:
    """Return the sorted channels that advertise background-RCA delivery."""
    return outbound_adapter_names_for(BACKGROUND_RCA)


__all__ = [
    "dispatch_background_notifications",
    "supported_notification_channels",
]
