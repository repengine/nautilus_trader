"""
Event provider helpers for dataset building.

Centralizes initialization of EventScheduleProvider so dataset builders and
other components do not duplicate source selection logic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ml.data.providers.events import EventScheduleProvider


if TYPE_CHECKING:
    from ml.data.sources.events import EventSource
else:  # pragma: no cover - typing fallback
    EventSource = Any


logger = logging.getLogger(__name__)


def build_event_provider(events_base_dir: Path | None) -> EventScheduleProvider | None:
    """
    Build an event schedule provider from the configured events directory.

    Args:
        events_base_dir: Base directory that may contain ``events.parquet``.

    Returns:
        EventScheduleProvider instance when initialization succeeds, otherwise ``None``.

    Example:
        >>> provider = build_event_provider(Path("data/events"))
        >>> assert provider is None or provider.__class__.__name__.endswith("Provider")
    """
    try:
        from ml.data.sources.events import FileEventSource
        from ml.data.sources.events import SimpleEventSource

        events_path: Path | None = None
        if events_base_dir is not None:
            candidate = events_base_dir / "events.parquet"
            if candidate.exists():
                events_path = candidate

        source: EventSource = (
            cast(EventSource, FileEventSource(events_path)) if events_path else SimpleEventSource()
        )
        return EventScheduleProvider(source)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.debug("Event provider initialization failed: %s", exc, exc_info=True)
        return None
