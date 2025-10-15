"""
Actor-side domain events bridge (non-blocking enqueue + background flusher).

This bridge ensures the actor thread performs only an O(1) enqueue per event, while a
background worker drains a bounded queue and calls the configured publisher. Dropped
events due to backpressure are logged via counters in the observability pipeline
(optional; best-effort).

Features:
- Per-topic rate limiting with configurable throttling rules
- Enhanced drop metrics with reason labeling (capacity, rate_limit, etc.)
- Queue depth monitoring and backpressure visibility
- Stress testing support for capacity and metrics verification

"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

from ml.common.message_bus import MessagePublisherProtocol
from ml.common.message_bus import publisher_from_config  # re-exported for tests
from ml.common.metrics_manager import MetricsManager
from ml.common.throttler import Throttler


class _QueuedEvent(NamedTuple):
    topic: str
    payload: dict[str, Any]


@dataclass(frozen=True)
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
    Actor-side non-blocking domain event publisher bridge.

    Provides O(1) enqueue on the actor thread and offloads publishing to a
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
    ...     per_topic_throttles=per_topic_config
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
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ml-domain-events", daemon=True)
        self._thread.start()

    def stop(self, *, drain: bool = True, timeout: float | None = 1.0) -> None:
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
            # Never allow metrics errors to affect behavior — record debug only
            logging.getLogger(__name__).debug(
                "Drop metric emit failed: %s",
                exc,
                exc_info=True,
            )

    def _record_queue_depth_metric(self) -> None:
        """
        Record current queue depth metric.
        """
        try:
            current_depth = self._queue.qsize()
            mm = MetricsManager.default()
            mm.set_gauge(
                "nautilus_ml_backpressure_queue_depth",
                "Current depth of actor-side domain event queue",
                float(current_depth),
                labels={"component": self._component_id},
                labelnames=("component",),
            )
            # Update stats
            self._stats = self._stats._replace(last_queue_depth=current_depth)
        except Exception as exc:
            logger = logging.getLogger(__name__)
            logger.debug("Domain event queue depth gauge update failed (ignored): %s", exc)

    def _record_topic_metric(self, topic: str, action: str) -> None:
        """
        Record per-topic metrics for enhanced telemetry.

        Parameters
        ----------
        topic : str
            Topic name for labeling.
        action : str
            Action taken: "published", "throttled", "topic_throttled".

        """
        try:
            # Extract topic prefix for lower cardinality (e.g., "ml.features" from "ml.features.computed.EURUSD")
            topic_prefix = ".".join(topic.split(".")[:2]) if "." in topic else topic

            mm = MetricsManager.default()
            mm.inc(
                "nautilus_ml_topic_events_total",
                "Total events processed per topic",
                labels={
                    "component": self._component_id,
                    "topic_prefix": topic_prefix,
                    "action": action,
                },
                labelnames=("component", "topic_prefix", "action"),
            )
        except Exception as exc:
            # Never allow metrics errors to affect behavior — record debug only
            logging.getLogger(__name__).debug(
                "Topic metric emit failed: %s",
                exc,
                exc_info=True,
            )

    def _record_throttle_efficiency_metrics(self) -> None:
        """
        Record throttling efficiency metrics periodically.
        """
        try:
            if self._stats.total_events == 0:
                return

            throttle_rate = self._stats.throttled_events / self._stats.total_events
            queue_utilization = self._stats.last_queue_depth / self._queue.maxsize

            mm = MetricsManager.default()
            mm.set_gauge(
                "nautilus_ml_throttle_efficiency_ratio",
                "Ratio of throttled events to total events",
                throttle_rate,
                labels={"component": self._component_id},
                labelnames=("component",),
            )
            mm.set_gauge(
                "nautilus_ml_queue_utilization_ratio",
                "Queue utilization ratio (depth/maxsize)",
                queue_utilization,
                labels={"component": self._component_id},
                labelnames=("component",),
            )
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Throttle efficiency metric emit failed: %s",
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
                self._publisher.publish(evt.topic, evt.payload)
            finally:
                self._queue.task_done()


def _parse_per_topic_throttles() -> dict[str, TopicThrottleConfig]:
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

    topic_throttles = {}
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
            from ml.common.metrics_manager import MetricsManager as _MM

            _MM.default().inc(
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
                exc_info=True,
            )
        logging.getLogger(__name__).debug(
            "Failed to parse ML_BUS_TOPIC_THROTTLES: %s",
            exc,
            exc_info=True,
        )

    return topic_throttles


def init_actor_bus_bridge(actor: Any) -> tuple[DomainEventBridge | None, str, str]:
    """
    Initialize an actor-side domain event bridge from environment configuration.

    Returns
    -------
    tuple[DomainEventBridge | None, str, str]
        (bridge, topic_scheme, topic_prefix). Bridge is None when disabled.

    """
    # Default topic configuration
    topic_scheme = "domain_op"
    topic_prefix = "events.ml"

    try:
        # Lazy import to avoid cycles in module graph
        from ml.common.throttler import Throttler as _Throttler
        from ml.config.actor_bus import ActorBusConfig
        from ml.config.bus import MessageBusConfig

        actor_bus_cfg = ActorBusConfig.from_env()
        bus_cfg = MessageBusConfig.from_env()
        if not (actor_bus_cfg.from_actor and bus_cfg.enabled):
            return None, topic_scheme, topic_prefix

        publisher = publisher_from_config(bus_cfg)
        throttler = (
            _Throttler(
                rate_per_sec=float(actor_bus_cfg.throttle_rate_per_sec),
                burst=int(actor_bus_cfg.throttle_burst),
            )
            if actor_bus_cfg.throttle_enabled
            else None
        )
        # Parse per-topic throttling configuration
        per_topic_throttles = _parse_per_topic_throttles()

        bridge = DomainEventBridge(
            publisher,
            max_queue=4096,
            throttler=throttler,
            per_topic_throttles=per_topic_throttles,
        )
        bridge.start()

        # Update topic config from actor bus settings
        topic_scheme = str(actor_bus_cfg.scheme)
        topic_prefix = str(actor_bus_cfg.prefix)

        # Mutual exclusion: disable store-path publishers to avoid duplicates
        try:
            stores = [
                getattr(actor, "_feature_store", None),
                getattr(actor, "_model_store", None),
                getattr(actor, "_strategy_store", None),
                getattr(actor, "_data_store", None),
            ]
            for st in stores:
                if st is None:
                    continue
                if hasattr(st, "publisher"):
                    setattr(st, "publisher", None)
                if hasattr(st, "_enable_publishing"):
                    try:
                        setattr(st, "_enable_publishing", False)
                    except Exception as set_exc:
                        logging.getLogger(__name__).debug(
                            "Failed to disable store-level publishing: %s",
                            set_exc,
                            exc_info=True,
                        )
        except Exception as exc:
            # Never impact initialization on optional convenience — log debug
            logging.getLogger(__name__).debug(
                "Actor bus mutual exclusion setup failed: %s",
                exc,
                exc_info=True,
            )

        return bridge, topic_scheme, topic_prefix
    except Exception as exc:
        # Best-effort helper; keep actor hot path clean — record warning metric
        try:
            from ml.common.metrics_manager import MetricsManager as _MM

            _MM.default().inc(
                "ml_pipeline_warnings_total",
                "Pipeline warnings",
                labels={
                    "component": "domain_events",
                    "op": "init_actor_bus_bridge",
                    "error_type": "exception",
                },
                labelnames=("component", "op", "error_type"),
            )
        except Exception as metric_exc:
            logging.getLogger(__name__).debug(
                "Warning metric emit failed (init_actor_bus_bridge): %s",
                metric_exc,
                exc_info=True,
            )
        logging.getLogger(__name__).debug(
            "init_actor_bus_bridge failed: %s",
            exc,
            exc_info=True,
        )
        return None, topic_scheme, topic_prefix
