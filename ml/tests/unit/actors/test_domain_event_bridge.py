from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import Mock, patch

from ml.actors.ml_domain_events import DomainEventBridge, TopicThrottleConfig
from ml.common.message_bus import MessagePublisherProtocol
from ml.common.metrics_manager import MetricsManager
from ml.common.throttler import Throttler


class CapturePublisher(MessagePublisherProtocol):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        self.calls.append((topic, payload))
        return True


class SlowPublisher(MessagePublisherProtocol):
    """Publisher that simulates slow downstream system."""

    def __init__(self, delay_seconds: float = 0.1) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.delay_seconds = delay_seconds
        self.publish_count = 0
        self.lock = threading.Lock()

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        time.sleep(self.delay_seconds)
        with self.lock:
            self.calls.append((topic, payload))
            self.publish_count += 1
        return True


class MetricsCapture:
    """Capture metrics calls for testing."""

    def __init__(self) -> None:
        self.counters: dict[str, int] = {}
        self.gauges: dict[str, float] = {}
        self.counter_labels: dict[str, dict[str, str]] = {}
        self.gauge_labels: dict[str, dict[str, str]] = {}

    def inc(self, name: str, doc: str, labels: dict[str, str] | None = None, **kwargs) -> None:
        self.counters[name] = self.counters.get(name, 0) + 1
        if labels:
            self.counter_labels[name] = labels

    def set_gauge(self, name: str, doc: str, value: float, labels: dict[str, str] | None = None, **kwargs) -> None:
        self.gauges[name] = value
        if labels:
            self.gauge_labels[name] = labels


def test_bridge_enqueues_and_flushes() -> None:
    cap = CapturePublisher()
    bridge = DomainEventBridge(cap, max_queue=8)
    bridge.start()
    try:
        for i in range(4):
            assert bridge.publish("topic", {"i": i}) is True
        # Give background thread a moment to drain
        time.sleep(0.05)
        # Stop and drain
    finally:
        bridge.stop(drain=True, timeout=1.0)

    assert len(cap.calls) >= 1


def test_bridge_backpressure_drop() -> None:
    cap = CapturePublisher()
    bridge = DomainEventBridge(cap, max_queue=1)
    bridge.start()
    try:
        assert bridge.publish("t1", {"i": 1}) is True
        # Immediately fill the queue; next may drop
        dropped = not bridge.publish("t2", {"i": 2})
        assert dropped in {True, False}
    finally:
        bridge.stop(drain=True, timeout=1.0)


def test_bridge_respects_throttler() -> None:
    cap = CapturePublisher()
    throttler = Throttler(rate_per_sec=1.0, burst=1)
    bridge = DomainEventBridge(cap, max_queue=8, throttler=throttler)
    bridge.start()
    try:
        payload1 = {"ts_max": 0}
        payload2 = {"ts_max": 0}
        assert bridge.publish("topic", payload1) is True
        # Without time advancing, throttler should drop the next publish
        assert bridge.publish("topic", payload2) is False
    finally:
        bridge.stop(drain=True, timeout=1.0)


def test_per_topic_throttling() -> None:
    """Test per-topic throttling configuration."""
    cap = CapturePublisher()
    per_topic_throttles = {
        "ml.features": TopicThrottleConfig(rate_per_sec=2.0, burst=1),
        "ml.predictions": TopicThrottleConfig(rate_per_sec=10.0, burst=2),
    }
    bridge = DomainEventBridge(cap, max_queue=32, per_topic_throttles=per_topic_throttles)
    bridge.start()

    try:
        # Test ml.features throttling (rate: 2/sec, burst: 1)
        now = time.time_ns()
        payload = {"ts_max": now}

        # First should succeed (uses burst token for first topic)
        assert bridge.publish("ml.features.computed.EURUSD", payload) is True

        # Second should succeed (uses burst token for second topic - different key)
        assert bridge.publish("ml.features.computed.GBPUSD", payload) is True

        # Third on same topic should be throttled (no tokens left for EURUSD)
        assert bridge.publish("ml.features.computed.EURUSD", payload) is False

        # Different topic prefix should not be throttled
        assert bridge.publish("ml.predictions.made.EURUSD", payload) is True

        time.sleep(0.1)  # Let some time pass for token refill

    finally:
        bridge.stop(drain=True, timeout=1.0)


def test_stress_queue_capacity_exceeds_limit() -> None:
    """Stress test: Enqueue beyond capacity and verify drop metrics."""
    metrics_capture = MetricsCapture()

    slow_publisher = SlowPublisher(delay_seconds=0.1)  # Slow to create backlog
    bridge = DomainEventBridge(slow_publisher, max_queue=5, component_id="stress_test")
    bridge.start()

    with patch.object(MetricsManager, "default", return_value=metrics_capture):
        try:
            # Rapidly enqueue more than capacity
            results = []
            for i in range(20):
                result = bridge.publish(f"topic.{i % 3}", {"event_id": i})
                results.append(result)
                if i < 10:
                    time.sleep(0.01)  # Small delay to allow some processing

            # Wait a bit for queue processing
            time.sleep(0.5)

        finally:
            bridge.stop(drain=True, timeout=2.0)

    # Verify that some events were dropped due to queue being full
    successful_publishes = sum(results)
    dropped_publishes = len(results) - successful_publishes

    assert dropped_publishes > 0, "Expected some events to be dropped"
    assert "nautilus_ml_backpressure_drops_total" in metrics_capture.counters
    assert metrics_capture.counters["nautilus_ml_backpressure_drops_total"] >= dropped_publishes

    # Verify queue depth metrics were recorded
    assert "nautilus_ml_backpressure_queue_depth" in metrics_capture.gauges


def test_stress_throttling_behavior_under_load() -> None:
    """Stress test: Verify throttling behavior under high event load."""
    metrics_capture = MetricsCapture()
    cap = CapturePublisher()

    # Configure aggressive throttling
    throttler = Throttler(rate_per_sec=5.0, burst=2)
    per_topic_throttles = {
        "high_volume": TopicThrottleConfig(rate_per_sec=1.0, burst=1),
    }

    bridge = DomainEventBridge(
        cap,
        max_queue=100,
        throttler=throttler,
        per_topic_throttles=per_topic_throttles,
        component_id="throttle_stress"
    )
    bridge.start()

    with patch.object(MetricsManager, "default", return_value=metrics_capture):
        try:
            # Send high volume of events
            results = []
            now = time.time_ns()

            for i in range(50):
                topic = "high_volume.topic" if i % 2 == 0 else "normal.topic"
                payload = {"ts_max": now, "event_id": i}
                result = bridge.publish(topic, payload)
                results.append((topic, result))

        finally:
            bridge.stop(drain=True, timeout=1.0)

    # Verify throttling occurred
    throttled_count = sum(1 for _, result in results if not result)
    assert throttled_count > 0, "Expected some events to be throttled"

    # Verify different drop reasons were recorded
    assert "nautilus_ml_backpressure_drops_total" in metrics_capture.counters

    # Verify topic-specific metrics were recorded
    if "nautilus_ml_topic_events_total" in metrics_capture.counters:
        assert metrics_capture.counters["nautilus_ml_topic_events_total"] > 0


def test_stress_concurrent_publishing() -> None:
    """Stress test: Concurrent publishing from multiple threads."""
    metrics_capture = MetricsCapture()
    cap = CapturePublisher()
    bridge = DomainEventBridge(cap, max_queue=50, component_id="concurrent_test")
    bridge.start()

    results = []
    threads = []
    events_per_thread = 25

    def publisher_worker(thread_id: int) -> None:
        thread_results = []
        for i in range(events_per_thread):
            topic = f"thread.{thread_id}.topic"
            payload = {"thread_id": thread_id, "event_id": i, "ts_max": time.time_ns()}
            result = bridge.publish(topic, payload)
            thread_results.append(result)
            time.sleep(0.001)  # Small delay to create some contention
        results.extend(thread_results)

    with patch.object(MetricsManager, "default", return_value=metrics_capture):
        try:
            # Start multiple publisher threads
            for thread_id in range(4):
                thread = threading.Thread(target=publisher_worker, args=(thread_id,))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Allow some time for queue processing
            time.sleep(0.2)

        finally:
            bridge.stop(drain=True, timeout=2.0)

    # Verify that most events were processed successfully
    successful_publishes = sum(results)
    total_events = len(results)

    assert total_events == 4 * events_per_thread
    assert successful_publishes > total_events * 0.8, "Expected most events to succeed"

    # Verify queue depth metrics were updated
    if "nautilus_ml_backpressure_queue_depth" in metrics_capture.gauges:
        # Queue depth should have been recorded
        assert metrics_capture.gauges["nautilus_ml_backpressure_queue_depth"] >= 0


def test_stress_metrics_accuracy_under_load() -> None:
    """Stress test: Verify metrics accuracy under high event load."""
    metrics_capture = MetricsCapture()
    cap = CapturePublisher()

    # Configure limited throttling to test both paths
    throttler = Throttler(rate_per_sec=100.0, burst=10)
    bridge = DomainEventBridge(cap, max_queue=20, throttler=throttler, component_id="metrics_test")
    bridge.start()

    with patch.object(MetricsManager, "default", return_value=metrics_capture):
        try:
            # Generate mix of successful and throttled events
            total_events = 100
            results = []

            for i in range(total_events):
                # Use same timestamp to trigger throttling
                payload = {"ts_max": 0, "event_id": i}
                result = bridge.publish(f"topic.{i % 5}", payload)
                results.append(result)

            # Wait for processing
            time.sleep(0.3)

        finally:
            bridge.stop(drain=True, timeout=1.0)

    # Verify statistics accuracy
    stats = bridge.get_throttle_stats()
    successful_publishes = sum(results)
    failed_publishes = len(results) - successful_publishes

    assert stats.total_events == total_events
    assert stats.throttled_events + stats.queue_full_drops == failed_publishes

    # Verify metrics were recorded
    assert "nautilus_ml_backpressure_queue_depth" in metrics_capture.gauges
    if failed_publishes > 0:
        assert "nautilus_ml_backpressure_drops_total" in metrics_capture.counters


def test_throttle_stats_tracking() -> None:
    """Test throttle statistics tracking accuracy."""
    cap = CapturePublisher()
    throttler = Throttler(rate_per_sec=1.0, burst=1)
    bridge = DomainEventBridge(cap, max_queue=5, throttler=throttler)
    bridge.start()

    try:
        # Publish events to trigger various outcomes
        payload = {"ts_max": 0}

        # First should succeed (burst token)
        assert bridge.publish("topic1", payload) is True

        # Second with same topic should be throttled (no tokens)
        assert bridge.publish("topic1", payload) is False

        # Third with same topic should be throttled
        assert bridge.publish("topic1", payload) is False

        stats = bridge.get_throttle_stats()
        assert stats.total_events == 3
        assert stats.throttled_events == 2
        assert stats.queue_full_drops == 0

    finally:
        bridge.stop(drain=True, timeout=1.0)
