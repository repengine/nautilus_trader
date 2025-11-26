"""
Event polling component for Dashboard Service.

This module provides event polling capabilities with TTL caching, background
polling, and filtering support. All operations are cold-path only.

"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from threading import Event as ThreadEvent
from threading import Lock
from threading import Thread
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ml.common.metrics_bootstrap import get_counter
from ml.config.bus import MessageBusConfig


if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Metrics
_EVENT_CACHE_HITS = get_counter(
    "ml_dashboard_events_cache_hits_total",
    "Dashboard events cache hits",
)
_EVENT_CACHE_MISSES = get_counter(
    "ml_dashboard_events_cache_misses_total",
    "Dashboard events cache misses",
)
_EVENT_POLLS_TOTAL = get_counter(
    "ml_dashboard_events_poll_total",
    "Dashboard events poll attempts",
)
_EVENT_FAILURES_TOTAL = get_counter(
    "ml_dashboard_events_failure_total",
    "Dashboard events polling failures",
    labels=["reason"],
)


@dataclass
class _EventCache:
    """
    Bounded TTL cache for dashboard event history.

    This cache stores event history with time-to-live semantics, ensuring
    stale data is automatically discarded.

    Parameters
    ----------
    ttl_seconds : float
        Time-to-live for cached events in seconds
    max_entries : int
        Maximum number of events to store
    _clock : Callable[[], float], optional
        Clock function for testing (default: time.monotonic)

    """

    ttl_seconds: float
    max_entries: int
    _clock: Callable[[], float] = time.monotonic
    _events: list[dict[str, Any]] = field(default_factory=list)
    _expires_at: float = 0.0
    _lock: Lock = field(default_factory=Lock)

    def snapshot(self) -> tuple[list[dict[str, Any]], bool]:
        """
        Return cached events and whether they remain fresh.

        Returns
        -------
        tuple[list[dict[str, Any]], bool]
            Cached events and freshness flag (True if still valid)

        """
        now = self._clock()
        with self._lock:
            is_fresh = bool(self._events) and now < self._expires_at
            return list(self._events), is_fresh

    def update(self, events: list[dict[str, Any]]) -> None:
        """
        Update cache with new events and reset TTL.

        Parameters
        ----------
        events : list[dict[str, Any]]
            New events to cache (trimmed to max_entries)

        """
        trimmed = list(events[: self.max_entries])
        with self._lock:
            self._events = trimmed
            self._expires_at = self._clock() + self.ttl_seconds

    def stale_snapshot(self) -> list[dict[str, Any]]:
        """
        Return cached events irrespective of expiry (best effort).

        Returns
        -------
        list[dict[str, Any]]
            Cached events regardless of TTL

        """
        with self._lock:
            return list(self._events)


@runtime_checkable
class EventPollingProtocol(Protocol):
    """
    Protocol for event polling operations.

    This protocol defines the contract for listing ML events with filtering
    and managing background polling threads.

    """

    def list_events(
        self,
        *,
        limit: int = 100,
        stage: str | None = None,
        source: str | None = None,
        instrument_substr: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List recent events with optional filtering.

        Parameters
        ----------
        limit : int, default 100
            Maximum number of events to return
        stage : str | None
            Filter by stage name (e.g., "ingestion", "training")
        source : str | None
            Filter by source (e.g., "orchestrator", "actor")
        instrument_substr : str | None
            Filter by instrument substring (e.g., "EURUSD")

        Returns
        -------
        list[dict[str, Any]]
            Filtered event list (empty if disabled or error)

        """
        ...

    def start_event_polling(self, interval_seconds: float) -> None:
        """
        Start background event polling thread.

        Parameters
        ----------
        interval_seconds : float
            Polling interval in seconds (must be > 0)

        """
        ...

    def stop_event_polling(self) -> None:
        """Stop background event polling thread."""
        ...


class EventPollingComponent:
    """
    Event polling component with TTL caching and background polling.

    This component manages event retrieval from Redis Streams (when enabled)
    with caching, filtering, and optional background polling.

    Parameters
    ----------
    ttl_seconds : float
        Cache time-to-live in seconds
    max_entries : int
        Maximum events to cache

    Example
    -------
    >>> component = EventPollingComponent(ttl_seconds=30.0, max_entries=500)
    >>> component.start_event_polling(interval_seconds=10.0)
    >>> events = component.list_events(limit=50, stage="training")
    >>> component.stop_event_polling()

    """

    def __init__(
        self,
        *,
        ttl_seconds: float,
        max_entries: int,
    ) -> None:
        """
        Initialize event polling component.

        Parameters
        ----------
        ttl_seconds : float
            Cache time-to-live in seconds
        max_entries : int
            Maximum events to cache

        """
        self._event_cache = _EventCache(
            ttl_seconds=ttl_seconds,
            max_entries=max_entries,
        )
        self._event_poll_thread: Thread | None = None
        self._event_poll_stop: ThreadEvent | None = None

    def list_events(
        self,
        *,
        limit: int = 100,
        stage: str | None = None,
        source: str | None = None,
        instrument_substr: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List recent events from Redis Streams with optional filtering.

        Best-effort operation: returns empty list if message bus is disabled
        or unavailable. Uses cache when fresh, polls on cache miss.

        Parameters
        ----------
        limit : int, default 100
            Maximum number of events to return
        stage : str | None
            Filter by stage name (topic contains stage)
        source : str | None
            Filter by source (payload.source == source)
        instrument_substr : str | None
            Filter by instrument substring in topic or payload

        Returns
        -------
        list[dict[str, Any]]
            Filtered event list (empty if disabled or error)

        """
        limit_value = max(1, int(limit))
        cached, is_fresh = self._event_cache.snapshot()
        events = cached

        if is_fresh:
            _EVENT_CACHE_HITS.inc()
        else:
            _EVENT_CACHE_MISSES.inc()
            try:
                polled = self._poll_events(
                    limit=max(limit_value, self._event_cache.max_entries)
                )
                _EVENT_POLLS_TOTAL.inc()
                self._event_cache.update(polled)
                events = polled
            except RuntimeError as exc:
                reason = "disabled" if str(exc) == "events_disabled" else "error"
                _EVENT_FAILURES_TOTAL.labels(reason=reason).inc()
                events = cached
            except Exception:
                _EVENT_FAILURES_TOTAL.labels(reason="error").inc()
                logger.debug("event polling failed", exc_info=True)
                events = cached

        # Apply filters
        out: list[dict[str, Any]] = []
        for entry in events:
            topic = entry.get("topic", "")
            payload = entry.get("payload", {})

            # Filter by source
            if source is not None and str(payload.get("source")) != source:
                continue

            # Filter by instrument substring
            if instrument_substr:
                instrument = None
                if isinstance(payload, dict):
                    params = payload.get("params")
                    if isinstance(params, dict):
                        instrument = params.get("instrument")
                if not instrument or instrument_substr not in str(instrument):
                    if instrument_substr not in topic:
                        continue

            # Filter by stage
            if stage and stage not in topic:
                continue

            out.append(entry)
            if len(out) >= limit_value:
                break

        return out

    def start_event_polling(self, interval_seconds: float) -> None:
        """
        Start background event polling thread.

        Polls events at specified interval and updates cache. No-op if already
        running or interval <= 0.

        Parameters
        ----------
        interval_seconds : float
            Polling interval in seconds (must be > 0)

        """
        if interval_seconds <= 0.0 or self._event_poll_thread is not None:
            return

        stop = ThreadEvent()
        self._event_poll_stop = stop

        def _run() -> None:
            while not stop.wait(interval_seconds):
                try:
                    events = self._poll_events(limit=self._event_cache.max_entries)
                    _EVENT_POLLS_TOTAL.inc()
                    self._event_cache.update(events)
                except RuntimeError as exc:
                    reason = "disabled" if str(exc) == "events_disabled" else "error"
                    _EVENT_FAILURES_TOTAL.labels(reason=reason).inc()
                except Exception:
                    _EVENT_FAILURES_TOTAL.labels(reason="error").inc()
                    logger.debug("background event poll failed", exc_info=True)

        thread = Thread(target=_run, name="ml-dashboard-event-poll", daemon=True)
        thread.start()
        self._event_poll_thread = thread

    def stop_event_polling(self) -> None:
        """
        Stop background event polling thread.

        Signals thread to stop and waits up to 1 second for clean shutdown.
        """
        stop = self._event_poll_stop
        if stop is not None:
            stop.set()
        thread = self._event_poll_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._event_poll_thread = None
        self._event_poll_stop = None

    def _poll_events(self, *, limit: int) -> list[dict[str, Any]]:
        """
        Poll events from Redis Streams.

        Parameters
        ----------
        limit : int
            Maximum number of events to retrieve

        Returns
        -------
        list[dict[str, Any]]
            Events from Redis Stream

        Raises
        ------
        RuntimeError
            If message bus is disabled or Redis unavailable

        """
        cfg = MessageBusConfig.from_env()
        if not cfg.enabled or cfg.backend != "redis":
            raise RuntimeError("events_disabled")

        try:
            import redis

            client: Any = redis.Redis.from_url(cfg.redis_url, decode_responses=True)
            rows: list[tuple[str, dict[str, str]]] = client.xrevrange(
                cfg.redis_stream,
                count=max(1, int(limit)),
            )
        except RuntimeError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError("events_error") from exc

        events: list[dict[str, Any]] = []
        for entry_id, fields in rows:
            topic = fields.get("topic", "")
            payload_raw = fields.get("payload", "{}")
            try:
                payload = json.loads(payload_raw)
            except Exception:
                payload = {"raw": payload_raw}
            events.append({"id": entry_id, "topic": topic, "payload": payload})
        return events


__all__ = [
    "EventPollingComponent",
    "EventPollingProtocol",
]
