"""Background RCA notification helpers."""

from __future__ import annotations

from platform.common.background_investigation_types import BackgroundInvestigationRecord


def deliver_background_notifications(
    *,
    record: BackgroundInvestigationRecord,
    channels: tuple[str, ...],
) -> dict[str, str]:
    """Send configured notifications for a completed background RCA."""
    results: dict[str, str] = {}

    for channel in channels:
        if channel == "telegram":
            # Imported lazily: telegram delivery only fires on background-RCA
            # completion, so the telegram client must not load into the base
            # REPL boot import path.
            from integrations.telegram.background_adapter import (
                deliver_telegram_notification,
            )

            results["telegram"] = deliver_telegram_notification(record)
            continue

        if channel == "rocketchat":
            # Imported lazily for the same reason as the telegram channel.
            from integrations.rocketchat.background_adapter import (
                deliver_rocketchat_notification,
            )

            results["rocketchat"] = deliver_rocketchat_notification(record)
            continue

        if channel == "buzz":
            # Imported lazily for the same reason as the telegram channel.
            from integrations.buzz.background_adapter import deliver_buzz_notification

            results["buzz"] = deliver_buzz_notification(record)
            continue

        if channel != "email":
            results[channel] = "unsupported"
            continue

        # Imported lazily for the same reason as the telegram channel.
        from integrations.smtp.background_adapter import deliver_email_notification

        results["email"] = deliver_email_notification(record)

    return results
