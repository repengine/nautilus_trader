from __future__ import annotations

import json
from collections.abc import Sequence
from types import TracebackType
from typing import Any
from typing import Self
from urllib.error import URLError

import pytest

import ml.training.event_driven.azure_events as azure_events_module
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


class _ThreadStub:
    def __init__(self, *, target: Any, name: str, daemon: bool) -> None:
        self.target = target
        self.name = name
        self.daemon = daemon
        self.started = False
        self.join_calls: list[float] = []

    def start(self) -> None:
        self.started = True

    def join(self, timeout: float = 0.0) -> None:
        self.join_calls.append(timeout)


class _UrlopenResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        return False

    def read(self) -> bytes:
        return self._payload


def test_watcher_start_noops_when_disabled() -> None:
    config = AzureScheduledEventsConfig(enabled=False)
    watcher = AzureScheduledEventsWatcher(config, callback=lambda _: None)

    watcher.start()

    assert watcher._thread is None


def test_watcher_start_and_stop_manage_background_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    created_threads: list[_ThreadStub] = []

    def _thread_factory(*, target: Any, name: str, daemon: bool) -> _ThreadStub:
        thread = _ThreadStub(target=target, name=name, daemon=daemon)
        created_threads.append(thread)
        return thread

    monkeypatch.setattr(azure_events_module.threading, "Thread", _thread_factory)
    watcher = AzureScheduledEventsWatcher(
        AzureScheduledEventsConfig(enabled=True),
        callback=lambda _: None,
    )

    watcher.start()
    watcher.stop(timeout=0.25)

    assert len(created_threads) == 1
    assert created_threads[0].started is True
    assert created_threads[0].join_calls == [0.25]
    assert watcher._thread is None
    assert watcher._stop_event.is_set() is False


def test_watcher_poll_once_noops_when_disabled() -> None:
    fetch_calls = 0

    def _fetcher() -> Sequence[ScheduledEventNotice]:
        nonlocal fetch_calls
        fetch_calls += 1
        return ()

    watcher = AzureScheduledEventsWatcher(
        AzureScheduledEventsConfig(enabled=False),
        callback=lambda _: None,
        fetcher=_fetcher,
    )
    watcher.poll_once()

    assert fetch_calls == 0


def test_watcher_handles_callback_failure_and_deduplicates_event() -> None:
    callback_calls = 0

    def _callback(_: ScheduledEventNotice) -> None:
        nonlocal callback_calls
        callback_calls += 1
        raise RuntimeError("boom")

    watcher = AzureScheduledEventsWatcher(
        AzureScheduledEventsConfig(enabled=True),
        callback=_callback,
        fetcher=lambda: (_notice(event_id="evt-1"),),
    )

    watcher.poll_once()
    watcher.poll_once()

    assert callback_calls == 1


def test_fetch_events_parses_and_filters_payload_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    watcher = AzureScheduledEventsWatcher(
        AzureScheduledEventsConfig(enabled=True),
        callback=lambda _: None,
    )
    payload = {
        "Events": [
            {
                "EventId": "evt-1",
                "EventType": "Preempt",
                "EventStatus": "Scheduled",
                "Resources": ["vm-1", " ", 123],
                "NotBefore": "2026-02-08T12:00:00Z",
            },
            {
                "EventID": "evt-2",
                "EventType": "Preempt",
                "EventStatus": "InProgress",
                "Resources": "invalid",
                "NotBefore": "",
            },
            {"EventId": "", "EventType": "Preempt", "EventStatus": "Scheduled"},
            "invalid",
        ],
    }

    def _urlopen(_: Any, timeout: float) -> _UrlopenResponse:
        assert timeout == pytest.approx(float(watcher._config.request_timeout_seconds))
        return _UrlopenResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(azure_events_module, "urlopen", _urlopen)

    notices = watcher._fetch_events()

    assert len(notices) == 2
    assert notices[0].event_id == "evt-1"
    assert notices[0].resources == ("vm-1", "123")
    assert notices[0].not_before == "2026-02-08T12:00:00Z"
    assert notices[1].event_id == "evt-2"
    assert notices[1].resources == ()
    assert notices[1].not_before is None


def test_fetch_events_returns_empty_tuple_when_events_field_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watcher = AzureScheduledEventsWatcher(
        AzureScheduledEventsConfig(enabled=True),
        callback=lambda _: None,
    )

    def _urlopen(_: Any, timeout: float) -> _UrlopenResponse:
        assert timeout == pytest.approx(float(watcher._config.request_timeout_seconds))
        return _UrlopenResponse(b'{"Events": null}')

    monkeypatch.setattr(azure_events_module, "urlopen", _urlopen)

    assert watcher._fetch_events() == ()


def test_fetch_events_raises_runtime_error_for_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watcher = AzureScheduledEventsWatcher(
        AzureScheduledEventsConfig(enabled=True),
        callback=lambda _: None,
    )
    monkeypatch.setattr(
        azure_events_module,
        "urlopen",
        lambda _request, timeout: _UrlopenResponse(b"not-json"),
    )

    with pytest.raises(RuntimeError, match="Invalid JSON payload"):
        watcher._fetch_events()


def test_fetch_events_raises_runtime_error_for_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watcher = AzureScheduledEventsWatcher(
        AzureScheduledEventsConfig(enabled=True),
        callback=lambda _: None,
    )

    def _raising_urlopen(_request: Any, timeout: float) -> _UrlopenResponse:
        assert timeout == pytest.approx(float(watcher._config.request_timeout_seconds))
        raise URLError("unreachable")

    monkeypatch.setattr(azure_events_module, "urlopen", _raising_urlopen)

    with pytest.raises(RuntimeError, match="Failed to query Azure scheduled events"):
        watcher._fetch_events()
