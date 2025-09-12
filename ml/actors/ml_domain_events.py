"""
Actor-side domain events bridge (non-blocking enqueue + background flusher).

This bridge ensures the actor thread performs only an O(1) enqueue per event, while a
background worker drains a bounded queue and calls the configured publisher. Dropped
events due to backpressure are logged via counters in the observability pipeline
(optional; best-effort).

"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, NamedTuple

from ml.common.message_bus import MessagePublisherProtocol
from ml.common.message_bus import publisher_from_config  # re-exported for tests
from ml.common.metrics_manager import MetricsManager
from ml.common.throttler import Throttler


class _QueuedEvent(NamedTuple):
    topic: str
    payload: dict[str, Any]


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
        Optional token bucket throttler applied per-topic.
    component_id : str, default "ml_actor"
        Component label for metrics (kept low-cardinality).

    Usage
    -----
    >>> bridge = DomainEventBridge(publisher, max_queue=1024)
    >>> bridge.start()
    >>> bridge.publish("ml.data.created.EURUSD.SIM", {"k": 1})
    True
    >>> bridge.stop()

    """

    def __init__(
        self,
        publisher: MessagePublisherProtocol,
        *,
        max_queue: int = 4096,
        throttler: Throttler | None = None,
        component_id: str = "ml_actor",
    ) -> None:
        self._publisher = publisher
        self._queue: queue.Queue[_QueuedEvent] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._throttler = throttler
        self._component_id = component_id

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

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        """
        Enqueue an event for background publishing.

        Returns
        -------
        bool
            True when enqueued successfully; False when dropped due to backpressure.

        """
        try:
            if self._throttler is not None:
                now_ns = payload.get("ts_max") or payload.get("created_at") or 0
                try:
                    now_ns_int = int(now_ns)
                except Exception:
                    now_ns_int = 0
                if not self._throttler.should_publish(topic, now_ns_int):
                    mm = MetricsManager.default()
                    mm.inc(
                        "nautilus_ml_backpressure_drops_total",
                        "Total events dropped due to backpressure",
                        labels={"component": self._component_id, "reason": "throttled"},
                        labelnames=("component", "reason"),
                    )
                    return False
            self._queue.put_nowait(_QueuedEvent(topic, payload))
            try:
                mm = MetricsManager.default()
                mm.set_gauge(
                    "nautilus_ml_backpressure_queue_depth",
                    "Current depth of actor-side domain event queue",
                    float(self._queue.qsize()),
                    labels={"component": self._component_id},
                    labelnames=("component",),
                )
            except Exception as exc:
                logger = logging.getLogger(__name__)
                logger.debug("Domain event queue depth gauge update failed (ignored): %s", exc)
            return True
        except queue.Full:
            mm = MetricsManager.default()
            mm.inc(
                "nautilus_ml_backpressure_drops_total",
                "Total events dropped due to backpressure",
                labels={"component": self._component_id, "reason": "queue_full"},
                labelnames=("component", "reason"),
            )
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
        bridge = DomainEventBridge(publisher, max_queue=4096, throttler=throttler)
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
                    except Exception:
                        pass
        except Exception:
            # Never impact initialization on optional convenience
            pass

        return bridge, topic_scheme, topic_prefix
    except Exception:
        # Best-effort helper; keep actor hot path clean
        return None, topic_scheme, topic_prefix
