
from collections import deque
from datetime import datetime, timezone
from threading import Lock
from typing import Any


class SecurityAudit:
    """Store bounded security events without raw message content."""

    def __init__(self, max_events: int = 1_000) -> None:
        if max_events <= 0:
            raise ValueError("max_events must be positive.")

        self._events: deque[dict[str, Any]] = deque(
            maxlen=max_events
        )
        self._lock = Lock()

    def record(
        self,
        event_type: str,
        category: str,
        *,
        blocked: bool,
    ) -> None:
        """Record event metadata without storing conversation text."""

        if not event_type.strip():
            raise ValueError("event_type cannot be empty.")

        if not category.strip():
            raise ValueError("category cannot be empty.")

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "category": category,
            "blocked": blocked,
        }

        with self._lock:
            self._events.append(event)

    def report(self) -> dict[str, Any]:
        """Return aggregate statistics and recent event metadata."""

        with self._lock:
            events = list(self._events)

        return {
            "total_events": len(events),
            "total_blocked": sum(
                event["blocked"] for event in events
            ),
            "recent_events": events[-5:],
        }
