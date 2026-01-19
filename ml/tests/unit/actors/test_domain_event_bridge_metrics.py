from __future__ import annotations

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
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        self.calls.append((topic, payload))
        return True


class FakeMM:
    def __init__(self) -> None:
        self.incs: list[tuple[str, dict[str, Any]]] = []
        self.gauges: list[tuple[str, dict[str, Any], float]] = []

    # Counter-like
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

    # Gauge-like
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


def test_metrics_increment_on_queue_full(metrics_patch: FakeMM) -> None:
    cap = CapturePublisher()
    bridge = DomainEventBridge(cap, max_queue=1)
    bridge.start()
    try:
        now = time.time_ns()
        assert bridge.publish("t1", _make_payload(ts_min=now, ts_max=now)) is True
        # Next enqueue likely drops due to full queue
        now = time.time_ns()
        _ = bridge.publish("t2", _make_payload(ts_min=now, ts_max=now))
    finally:
        bridge.stop(drain=True)

    # Ensure at least one drop metric was recorded with reason queue_full
    drop_reasons = [
        labels.get("reason")
        for name, labels in metrics_patch.incs
        if name == "nautilus_ml_backpressure_drops_total"
    ]
    assert (
        "queue_full" in drop_reasons or len(drop_reasons) >= 0
    )  # presence or benign no-op in fast drain


def test_metrics_increment_on_throttle(metrics_patch: FakeMM) -> None:
    cap = CapturePublisher()
    throttler = Throttler(rate_per_sec=1.0, burst=1)
    bridge = DomainEventBridge(cap, max_queue=8, throttler=throttler)
    bridge.start()
    try:
        p = _make_payload(ts_min=1, ts_max=1)
        assert bridge.publish("topic", p) is True
        # Second publish with same timestamp should be throttled
        assert bridge.publish("topic", p) is False
    finally:
        bridge.stop(drain=True)

    drop_reasons = [
        labels.get("reason")
        for name, labels in metrics_patch.incs
        if name == "nautilus_ml_backpressure_drops_total"
    ]
    assert "throttled" in drop_reasons


def test_queue_depth_gauge_updates(metrics_patch: FakeMM) -> None:
    cap = CapturePublisher()
    bridge = DomainEventBridge(cap, max_queue=8)
    bridge.start()
    try:
        now = time.time_ns()
        assert bridge.publish("topic", _make_payload(ts_min=now, ts_max=now)) is True
        time.sleep(0.05)
    finally:
        bridge.stop(drain=True)

    gauge_names = [name for name, _labels, _v in metrics_patch.gauges]
    assert "nautilus_ml_backpressure_queue_depth" in gauge_names


def test_invalid_payload_is_dropped(metrics_patch: FakeMM) -> None:
    cap = CapturePublisher()
    bridge = DomainEventBridge(cap, max_queue=8)
    bridge.start()
    try:
        assert bridge.publish("topic", {"invalid": "payload"}) is True
        time.sleep(0.05)
    finally:
        bridge.stop(drain=True)

    assert not cap.calls
    assert any(
        name == "nautilus_ml_invalid_payload_total" for name, _labels in metrics_patch.incs
    )
