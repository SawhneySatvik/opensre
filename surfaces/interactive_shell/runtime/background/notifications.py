"""Background RCA notification helpers.

Channel selection moved to the outbound adapter registry in
``platform.notifications``. This stays as the REPL runtime's entry point:
it runs the registration bootstrap, which ``platform`` cannot do itself
because ``bootstrap`` sits above it in the layering contract.

Imports stay inside the function so the REPL boot path pays for none of it.
"""

from __future__ import annotations

from platform.common.background_investigation_types import BackgroundInvestigationRecord


def deliver_background_notifications(
    *,
    record: BackgroundInvestigationRecord,
    channels: tuple[str, ...],
) -> dict[str, str]:
    """Send configured notifications for a completed background RCA."""
    from bootstrap.adapters import install_notification_adapters
    from platform.notifications.outbound_dispatch import dispatch_background_notifications

    install_notification_adapters()
    return dispatch_background_notifications(record=record, channels=channels)
