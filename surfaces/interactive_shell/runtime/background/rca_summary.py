"""Re-export of the bounded RCA summary used by chat-notification channels.

The implementation moved to ``platform.notifications.rca_summary`` so the
vendor notification adapters under ``integrations/`` can share it: the
layering contract forbids them importing this package.
"""

from __future__ import annotations

from platform.notifications.rca_summary import summary_sections

__all__ = ["summary_sections"]
