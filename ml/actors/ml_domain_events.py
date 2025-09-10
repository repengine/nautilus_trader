"""
Actor-side domain events bridge (non-blocking enqueue + background flusher).

This bridge ensures the actor thread performs only an O(1) enqueue per event, while a
background worker drains a bounded queue and calls the configured publisher. Dropped
events due to backpressure are logged via counters in the observability pipeline
(optional; best-effort).

"""

from __future__ import annotations

import queue
import threading
from typing import Any, NamedTuple

from ml.common.message_bus import MessagePublisherProtocol
from ml.common.metrics import backpressure_drops_total
from ml.common.metrics import backpressure_queue_depth
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
                    backpressure_drops_total.labels(
                        component=self._component_id,
                        reason="throttled",
                    ).inc()
                    return False
            self._queue.put_nowait(_QueuedEvent(topic, payload))
            try:
                backpressure_queue_depth.labels(component=self._component_id).set(
                    float(self._queue.qsize()),
                )
            except Exception:
                pass
            return True
        except queue.Full:
            backpressure_drops_total.labels(
                component=self._component_id,
                reason="queue_full",
            ).inc()
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
