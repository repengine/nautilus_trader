from __future__ import annotations

from typing import Callable, Sequence

import pytest

from ml.config.streaming_pipeline import AzureScheduledEventsConfig
from ml.training.event_driven.azure_events import AzureScheduledEventsWatcher
from ml.training.event_driven.azure_events import ScheduledEventNotice


def _notice(
    *,
    event_id: str,
    event_type: str = "Preempt",
    event_status: str = "Scheduled",
    resources: Sequence[str] = ("vm-1",),
    not_before: str | None = None,
) -> ScheduledEventNotice:
    return ScheduledEventNotice(
        event_id=event_id,
        event_type=event_type,
        event_status=event_status,
        resources=tuple(resources),
        not_before=not_before,
    )


def test_watcher_triggers_callback_for_matching_event() -> None:
    config = AzureScheduledEventsConfig(
        enabled=True,
        resource_filter=("vm-1",),
    )
    captured: list[str] = []

    def _callback(notice: ScheduledEventNotice) -> None:
        captured.append(notice.event_id)

    watcher = AzureScheduledEventsWatcher(
        config,
        callback=_callback,
        fetcher=lambda: (
            _notice(event_id="evt-1", resources=("vm-1",)),
            _notice(event_id="evt-2", event_type="Reboot", resources=("vm-1",)),
            _notice(event_id="evt-3", event_status="Completed", resources=("vm-1",)),
        ),
    )
    watcher.poll_once()
    assert captured == ["evt-1"]


def test_watcher_deduplicates_events() -> None:
    config = AzureScheduledEventsConfig(enabled=True)
    captured: list[str] = []

    def _callback(notice: ScheduledEventNotice) -> None:
        captured.append(notice.event_id)

    events = (_notice(event_id="evt-1"),)
    watcher = AzureScheduledEventsWatcher(
        config,
        callback=_callback,
        fetcher=lambda: events,
    )
    watcher.poll_once()
    watcher.poll_once()
    assert captured == ["evt-1"]


def test_watcher_respects_resource_filter() -> None:
    config = AzureScheduledEventsConfig(
        enabled=True,
        resource_filter=("vm-important",),
    )
    captured: list[str] = []

    watcher = AzureScheduledEventsWatcher(
        config,
        callback=lambda notice: captured.append(notice.event_id),
        fetcher=lambda: (
            _notice(event_id="evt-1", resources=("vm-other",)),
            _notice(event_id="evt-2", resources=("vm-important",)),
            _notice(event_id="evt-3", resources=("*",)),
        ),
    )
    watcher.poll_once()
    # event_id=2 matches resource filter; event_id=3 matches wildcard.
    assert captured == ["evt-2", "evt-3"]


def test_watcher_swallows_fetch_errors() -> None:
    config = AzureScheduledEventsConfig(enabled=True)
    watcher = AzureScheduledEventsWatcher(
        config,
        callback=lambda notice: pytest.fail(f"callback should not run: {notice}"),
        fetcher=_raise_error,
    )
    watcher.poll_once()  # Should not raise


def _raise_error() -> Sequence[ScheduledEventNotice]:
    raise RuntimeError("boom")
