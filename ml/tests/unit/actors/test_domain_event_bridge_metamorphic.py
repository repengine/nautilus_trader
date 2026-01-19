from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from ml.actors.ml_domain_events import DomainEventBridge
from ml.common.events_util import build_bus_payload
from ml.common.message_bus import MessagePublisherProtocol
from ml.common.metrics_manager import MetricsManager
from ml.common.throttler import Throttler
from ml.config.events import EventStatus, Source, Stage


def _make_payload(
    *,
    ts_min: int = 1,
    ts_max: int = 1,
    count: int = 1,
    dataset_id: str = "dataset",
    instrument_id: str = "EURUSD",
    run_id: str = "run",
    metadata: dict[str, object] | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    payload = build_bus_payload(
        dataset_id=dataset_id,
        instrument_id=instrument_id,
        stage=Stage.FEATURE_COMPUTED,
        source=Source.LIVE,
        run_id=run_id,
        ts_min=ts_min,
        ts_max=ts_max,
        count=count,
        status=EventStatus.SUCCESS,
        metadata=metadata,
    )
    if extra:
        payload.update(extra)
    return payload


class CapturePublisher(MessagePublisherProtocol):
    def __init__(self, delay_ms: float = 0.0) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._delay = delay_ms
        self._condition = threading.Condition()

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        if self._delay > 0:
            time.sleep(self._delay / 1000.0)
        with self._condition:
            self.calls.append((topic, payload))
            self._condition.notify_all()
        return True

    def wait_for_calls(self, expected: int, timeout: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while len(self.calls) < expected:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
        return len(self.calls) >= expected


class FakeMM:
    def __init__(self) -> None:
        self.incs: list[tuple[str, dict[str, Any]]] = []
        self.gauges: list[tuple[str, dict[str, Any], float]] = []

    def inc(
        self,
        name: str,
        desc: str,
        *,
        labels: dict[str, Any] | None = None,
        amount: float = 1.0,
        labelnames: tuple[str, ...] | None = None,
    ) -> None:
        self.incs.append((name, dict(labels or {})))

    def set_gauge(
        self,
        name: str,
        desc: str,
        value: float,
        *,
        labels: dict[str, Any] | None = None,
        labelnames: tuple[str, ...] | None = None,
    ) -> None:
        self.gauges.append((name, dict(labels or {}), float(value)))


@pytest.fixture()
def metrics_patch(monkeypatch: pytest.MonkeyPatch) -> FakeMM:
    fake = FakeMM()
    monkeypatch.setattr(MetricsManager, "default", classmethod(lambda cls: fake))
    return fake


def test_throttled_duplicates_publish_once(metrics_patch: FakeMM) -> None:
    cap = CapturePublisher()
    # Allow 1 event per second with burst=1
    throttler = Throttler(rate_per_sec=1.0, burst=1)
    bridge = DomainEventBridge(cap, throttler=throttler, max_queue=16)
    bridge.start()
    try:
        now_ns = time.time_ns()
        payload = _make_payload(ts_min=now_ns, ts_max=now_ns)
        assert bridge.publish("topic", payload) is True
        # Immediate duplicates should be throttled and not published again
        assert bridge.publish("topic", payload) is False
        assert bridge.publish("topic", payload) is False
        assert cap.wait_for_calls(1, timeout=1.0)
    finally:
        bridge.stop(drain=True)

    # Only one publish should be processed
    assert len(cap.calls) == 1
    # And throttled drops recorded
    assert any(
        labels.get("reason") == "throttled"
        for name, labels in metrics_patch.incs
        if name == "nautilus_ml_backpressure_drops_total"
    )


def test_bounded_queue_backpressure_and_drain(metrics_patch: FakeMM) -> None:
    # Small queue with slow publisher to trigger queue_full reliably
    cap = CapturePublisher(delay_ms=20)
    bridge = DomainEventBridge(cap, max_queue=1)
    bridge.start()
    try:
        # Enqueue one; next immediate enqueue should drop due to full queue
        now = time.time_ns()
        assert bridge.publish("topic", _make_payload(ts_min=now, ts_max=now)) is True
        drops = 0
        # Attempt several more while first is still processing
        for _ in range(5):
            now = time.time_ns()
            ok = bridge.publish("topic", _make_payload(ts_min=now, ts_max=now))
            if not ok:
                drops += 1
        # Some should have dropped due to queue_full
        assert drops >= 1
        assert cap.wait_for_calls(1, timeout=1.0)
    finally:
        bridge.stop(drain=True)

    # At least the first succeeded and was processed
    assert len(cap.calls) >= 1
    # And queue_full drop counter recorded
    assert any(
        labels.get("reason") == "queue_full"
        for name, labels in metrics_patch.incs
        if name == "nautilus_ml_backpressure_drops_total"
    )
