"""
Async message bus bridge for non-blocking publish paths.

This module provides a generic, shared domain-event bridge that enqueues events
from hot paths and flushes them on a background thread. It is safe for actors
and strategies to use without adding blocking I/O to critical loops.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

from ml.common.events_util import validate_bus_payload
from ml.common.message_bus import MessagePublisherProtocol
from ml.common.metrics_manager import MetricsManager
from ml.common.throttler import Throttler


class _QueuedEvent(NamedTuple):
    topic: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TopicThrottleConfig:
    """
    Configuration for per-topic rate limiting.
    """

    rate_per_sec: float
    burst: int


class ThrottleStats(NamedTuple):
    """
    Statistics for throttling behavior.
    """

    total_events: int
    throttled_events: int
    queue_full_drops: int
    last_queue_depth: int


class DomainEventBridge:
    """
    Non-blocking domain event publisher bridge.

    Provides O(1) enqueue on the caller thread and offloads publishing to a
    background worker. Drops are recorded under backpressure (throttled or
    queue_full) and a queue depth gauge is updated opportunistically.

    Parameters
    ----------
    publisher : MessagePublisherProtocol
        Concrete publisher to send events to the configured bus.
    max_queue : int, default 4096
        Maximum queue size before dropping new events.
    throttler : Throttler | None, optional
        Optional global token bucket throttler applied to all topics.
    per_topic_throttles : Mapping[str, TopicThrottleConfig] | None, optional
        Per-topic throttling configuration for fine-grained rate limiting.
    component_id : str, default "ml_actor"
        Component label for metrics (kept low-cardinality).

    Usage
    -----
    >>> per_topic_config = {
    ...     "ml.features.computed": TopicThrottleConfig(rate_per_sec=10.0, burst=5),
    ...     "ml.predictions.made": TopicThrottleConfig(rate_per_sec=100.0, burst=20)
    ... }
    >>> bridge = DomainEventBridge(
    ...     publisher,
    ...     max_queue=1024,
    ...     per_topic_throttles=per_topic_config,
    ... )
    >>> bridge.start()
    >>> bridge.publish("ml.features.computed.EURUSD.SIM", {"count": 100})
    True
    >>> bridge.stop()

    """

    def __init__(
        self,
        publisher: MessagePublisherProtocol,
        *,
        max_queue: int = 4096,
        throttler: Throttler | None = None,
        per_topic_throttles: Mapping[str, TopicThrottleConfig] | None = None,
        component_id: str = "ml_actor",
    ) -> None:
        self._publisher = publisher
        self._queue: queue.Queue[_QueuedEvent] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._throttler = throttler
        self._component_id = component_id

        # Per-topic throttlers initialized lazily
        self._per_topic_throttles: dict[str, Throttler] = {}
        self._topic_throttle_configs = dict(per_topic_throttles) if per_topic_throttles else {}

        # Statistics tracking
        self._stats = ThrottleStats(
            total_events=0,
            throttled_events=0,
            queue_full_drops=0,
            last_queue_depth=0,
        )

    def start(self) -> None:
        """
        Start the background worker thread (idempotent).
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ml-domain-events", daemon=True)
        self._thread.start()

    def stop(self, *, drain: bool = True, timeout: float | None = 1.0) -> None:
        """
        Stop the background worker thread.

        Args:
            drain: Whether to drain queued events before stopping.
            timeout: Maximum wait in seconds for the thread to join.
        """
        if self._thread is None:
            return
        if drain:
            # Signal stop and allow run loop to drain queue
            self._stop.set()
            self._thread.join(timeout=timeout)
        else:
            self._stop.set()
            self._thread.join(timeout=timeout)
        self._thread = None

    def get_throttle_stats(self) -> ThrottleStats:
        """
        Get current throttling statistics.

        Returns
        -------
        ThrottleStats
            Current statistics including event counts and queue depth.
        """
        return self._stats._replace(last_queue_depth=self._queue.qsize())

    def get_topic_throttlers(self) -> dict[str, Throttler]:
        """
        Get copy of per-topic throttlers for inspection.

        Returns
        -------
        dict[str, Throttler]
            Mapping of topic patterns to their throttlers.
        """
        return dict(self._per_topic_throttles)

    def _get_throttler_for_topic(self, topic: str, now_ns: int) -> Throttler | None:
        """
        Get or create throttler for a specific topic.

        Parameters
        ----------
        topic : str
            The topic name to check throttling for.
        now_ns : int
            Current timestamp in nanoseconds.

        Returns
        -------
        Throttler | None
            Topic-specific throttler if configured, None otherwise.
        """
        # Check for exact topic match first
        if topic in self._topic_throttle_configs:
            if topic not in self._per_topic_throttles:
                config = self._topic_throttle_configs[topic]
                self._per_topic_throttles[topic] = Throttler(
                    rate_per_sec=config.rate_per_sec,
                    burst=config.burst,
                )
            return self._per_topic_throttles[topic]

        # Check for prefix matches (e.g., "ml.features" matches "ml.features.computed.EURUSD")
        for pattern, config in self._topic_throttle_configs.items():
            if topic.startswith(pattern):
                if pattern not in self._per_topic_throttles:
                    self._per_topic_throttles[pattern] = Throttler(
                        rate_per_sec=config.rate_per_sec,
                        burst=config.burst,
                    )
                return self._per_topic_throttles[pattern]

        return None

    def _record_drop_metric(self, reason: str) -> None:
        """
        Record a drop metric with detailed reason labeling.

        Parameters
        ----------
        reason : str
            Reason for drop: "throttled", "queue_full", "topic_throttled".
        """
        try:
            mm = MetricsManager.default()
            mm.inc(
                "nautilus_ml_backpressure_drops_total",
                "Total events dropped due to backpressure",
                labels={"component": self._component_id, "reason": reason},
                labelnames=("component", "reason"),
            )
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Backpressure drop metric emit failed: %s",
                exc,
                exc_info=True,
            )

    def _record_queue_depth_metric(self) -> None:
        """
        Record queue depth metric.
        """
        try:
            mm = MetricsManager.default()
            mm.set_gauge(
                "nautilus_ml_backpressure_queue_depth",
                "Current depth of actor-side domain event queue",
                value=float(self._queue.qsize()),
                labels={"component": self._component_id},
                labelnames=("component",),
            )
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Queue depth metric emit failed: %s",
                exc,
                exc_info=True,
            )

    def _record_topic_metric(self, topic: str, status: str) -> None:
        """
        Record per-topic metrics for published/dropped events.
        """
        try:
            mm = MetricsManager.default()
            mm.inc(
                "nautilus_ml_domain_event_total",
                "Total domain events by topic and status",
                labels={"topic": topic, "status": status},
                labelnames=("topic", "status"),
            )
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Per-topic metric emit failed: %s",
                exc,
                exc_info=True,
            )

    def _record_throttle_efficiency_metrics(self) -> None:
        """
        Record throttling efficiency metrics (periodic).
        """
        try:
            if self._stats.total_events <= 0:
                return
            throttled_pct = (
                float(self._stats.throttled_events) / float(self._stats.total_events)
            )
            mm = MetricsManager.default()
            mm.set_gauge(
                "nautilus_ml_domain_event_throttle_rate",
                "Throttle ratio for domain events (throttled / total)",
                value=throttled_pct,
                labels={"component": self._component_id},
                labelnames=("component",),
            )
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Throttle efficiency metric emit failed: %s",
                exc,
                exc_info=True,
            )

    def _record_invalid_payload_metric(self, reason: str) -> None:
        """
        Record invalid payload metrics.

        Parameters
        ----------
        reason : str
            Reason for payload rejection (e.g., "validation_failed").
        """
        try:
            mm = MetricsManager.default()
            mm.inc(
                "nautilus_ml_invalid_payload_total",
                "Total events dropped due to payload validation failures",
                labels={"component": self._component_id, "reason": reason},
                labelnames=("component", "reason"),
            )
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Invalid payload metric emit failed: %s",
                exc,
                exc_info=True,
            )

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        """
        Enqueue an event for background publishing.

        Applies both global and per-topic throttling rules, with enhanced metrics
        recording for different drop reasons.

        Returns
        -------
        bool
            True when enqueued successfully; False when dropped due to backpressure.
        """
        # Update total events counter
        self._stats = self._stats._replace(total_events=self._stats.total_events + 1)

        try:
            # Extract timestamp for throttling
            now_ns = (
                payload.get("ts_max")
                or payload.get("created_at")
                or int(time.time() * 1_000_000_000)
            )
            try:
                now_ns_int = int(now_ns)
            except Exception:
                now_ns_int = int(time.time() * 1_000_000_000)

            # Check global throttler first
            if self._throttler is not None:
                if not self._throttler.should_publish(topic, now_ns_int):
                    self._stats = self._stats._replace(
                        throttled_events=self._stats.throttled_events + 1,
                    )
                    self._record_drop_metric("throttled")
                    self._record_topic_metric(topic, "throttled")
                    return False

            # Check per-topic throttler
            topic_throttler = self._get_throttler_for_topic(topic, now_ns_int)
            if topic_throttler is not None:
                if not topic_throttler.should_publish(topic, now_ns_int):
                    self._stats = self._stats._replace(
                        throttled_events=self._stats.throttled_events + 1,
                    )
                    self._record_drop_metric("topic_throttled")
                    self._record_topic_metric(topic, "topic_throttled")
                    return False

            # Enqueue the event
            self._queue.put_nowait(_QueuedEvent(topic, payload))

            # Update metrics
            self._record_queue_depth_metric()
            self._record_topic_metric(topic, "published")

            # Periodically record efficiency metrics (every 100 events)
            if self._stats.total_events % 100 == 0:
                self._record_throttle_efficiency_metrics()

            return True

        except queue.Full:
            self._stats = self._stats._replace(queue_full_drops=self._stats.queue_full_drops + 1)
            self._record_drop_metric("queue_full")
            self._record_topic_metric(topic, "queue_full")
            return False

    def _run(self) -> None:
        # Drain until stop is set and queue is empty
        while not self._stop.is_set() or not self._queue.empty():
            try:
                evt = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                if "strategy_id" not in evt.payload:
                    is_valid, errors = validate_bus_payload(evt.payload)
                    if not is_valid:
                        self._record_invalid_payload_metric("validation_failed")
                        logging.getLogger(__name__).warning(
                            "ml_domain_event_invalid_payload",
                            extra={
                                "topic": evt.topic,
                                "errors": errors,
                            },
                        )
                        continue
                self._publisher.publish(evt.topic, evt.payload)
            finally:
                self._queue.task_done()


def parse_topic_throttles_from_env() -> dict[str, TopicThrottleConfig]:
    """
    Parse per-topic throttling configuration from environment variables.

    Expected format: ML_BUS_TOPIC_THROTTLES="topic1:rate1:burst1,topic2:rate2:burst2"
    Example: ML_BUS_TOPIC_THROTTLES="ml.features:10.0:5,ml.predictions:100.0:20"

    Returns
    -------
    dict[str, TopicThrottleConfig]
        Mapping of topic patterns to their throttling configurations.
    """
    import os

    throttle_config_str = os.getenv("ML_BUS_TOPIC_THROTTLES", "").strip()
    if not throttle_config_str:
        return {}

    topic_throttles: dict[str, TopicThrottleConfig] = {}
    try:
        for entry in throttle_config_str.split(","):
            entry = entry.strip()
            if not entry:
                continue

            parts = entry.split(":")
            if len(parts) != 3:
                continue

            topic_pattern, rate_str, burst_str = parts
            try:
                rate = float(rate_str)
                burst = int(burst_str)
                if rate > 0 and burst > 0:
                    topic_throttles[topic_pattern.strip()] = TopicThrottleConfig(
                        rate_per_sec=rate,
                        burst=burst,
                    )
            except ValueError:
                continue

    except Exception as exc:
        # Don't fail initialization on config parsing errors — record warning metric
        try:
            MetricsManager.default().inc(
                "ml_pipeline_warnings_total",
                "Pipeline warnings",
                labels={
                    "component": "domain_events",
                    "op": "parse_topic_throttles",
                    "error_type": "exception",
                },
                labelnames=("component", "op", "error_type"),
            )
        except Exception as metric_exc:
            logging.getLogger(__name__).debug(
                "Warning metric emit failed (parse_topic_throttles): %s",
                metric_exc,
            )
        logging.getLogger(__name__).debug(
            "Failed to parse ML_BUS_TOPIC_THROTTLES: %s",
            exc,
        )

    return topic_throttles


__all__ = [
    "DomainEventBridge",
    "ThrottleStats",
    "TopicThrottleConfig",
    "parse_topic_throttles_from_env",
]
