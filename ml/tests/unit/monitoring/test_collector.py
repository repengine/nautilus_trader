from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

import ml.common.metrics_manager as metrics_manager_module
import ml.monitoring.collector as collector_module
import ml.monitoring.collectors.base as base_module
from ml.monitoring._config import MonitoringConfig
from ml.monitoring.collector import MLMetricsCollector


@dataclass
class _MetricEvent:
    operation: str
    labels: dict[str, str]
    value: float


class _BoundMetric:
    def __init__(self, metric: _FakeMetric, labels: dict[str, str]) -> None:
        self._metric = metric
        self._labels = labels

    def inc(self, value: float = 1.0) -> None:
        self._metric.events.append(_MetricEvent("inc", dict(self._labels), value))

    def observe(self, value: float) -> None:
        self._metric.events.append(_MetricEvent("observe", dict(self._labels), value))

    def set(self, value: float) -> None:
        self._metric.events.append(_MetricEvent("set", dict(self._labels), value))


class _FakeMetric:
    def __init__(self) -> None:
        self.events: list[_MetricEvent] = []
        self.label_calls: list[dict[str, str]] = []

    def labels(self, **labels: str) -> _BoundMetric:
        self.label_calls.append(dict(labels))
        return _BoundMetric(self, dict(labels))


class _FakeMetricsManager:
    def __init__(self) -> None:
        self.metrics_by_name: dict[str, _FakeMetric] = {}

    def _create(self, name: str) -> _FakeMetric:
        metric = _FakeMetric()
        self.metrics_by_name[name] = metric
        return metric

    def counter(self, name: str, _description: str, _labelnames: list[str]) -> _FakeMetric:
        return self._create(name)

    def histogram(
        self,
        name: str,
        _description: str,
        _labelnames: list[str],
        *,
        buckets: list[float],
    ) -> _FakeMetric:
        assert buckets
        return self._create(name)

    def gauge(self, name: str, _description: str, _labelnames: list[str]) -> _FakeMetric:
        return self._create(name)


@pytest.fixture
def collector_with_fakes(monkeypatch: pytest.MonkeyPatch) -> tuple[MLMetricsCollector, _FakeMetricsManager]:
    fake_manager = _FakeMetricsManager()
    monkeypatch.setattr(base_module, "HAS_PROMETHEUS", True)
    monkeypatch.setattr(collector_module, "HAS_PROMETHEUS", True)
    monkeypatch.setattr(
        metrics_manager_module.MetricsManager,
        "default",
        staticmethod(lambda: fake_manager),
    )

    collector = MLMetricsCollector(MonitoringConfig(enabled=True))
    collector._initialize_metrics()
    return collector, fake_manager


def test_collector_initialization_registers_all_core_metrics(
    collector_with_fakes: tuple[MLMetricsCollector, _FakeMetricsManager],
) -> None:
    collector, _fake_manager = collector_with_fakes

    assert collector.enabled is True
    assert collector.get_metric_count() == 5
    assert collector.config.metrics_prefix == "nautilus_ml"
    assert set(collector.metrics.keys()) == {
        "ml_predictions_total",
        "ml_prediction_latency_seconds",
        "ml_model_confidence",
        "ml_feature_computation_latency_seconds",
        "ml_model_errors_total",
    }


def test_record_prediction_updates_all_prediction_metrics(
    collector_with_fakes: tuple[MLMetricsCollector, _FakeMetricsManager],
) -> None:
    collector, fake_manager = collector_with_fakes

    collector.record_prediction(
        model="tft",
        instrument="SPY",
        prediction_class="buy",
        latency_seconds=0.25,
        confidence=0.85,
        success=False,
    )

    predictions = fake_manager.metrics_by_name["nautilus_ml_predictions_total"]
    latency = fake_manager.metrics_by_name["nautilus_ml_prediction_latency_seconds"]
    confidence = fake_manager.metrics_by_name["nautilus_ml_model_confidence"]

    assert predictions.events[0].operation == "inc"
    assert predictions.events[0].labels["status"] == "error"
    assert latency.events[0] == _MetricEvent("observe", {"model": "tft", "instrument": "SPY"}, 0.25)
    assert confidence.events[0] == _MetricEvent("set", {"model": "tft", "instrument": "SPY"}, 0.85)


def test_record_feature_computation_and_error(
    collector_with_fakes: tuple[MLMetricsCollector, _FakeMetricsManager],
) -> None:
    collector, fake_manager = collector_with_fakes

    collector.record_feature_computation("AAPL", "technical", 0.12)
    collector.record_error("xgb", "AAPL", "timeout")

    feature_latency = fake_manager.metrics_by_name["nautilus_ml_feature_computation_latency_seconds"]
    model_errors = fake_manager.metrics_by_name["nautilus_ml_model_errors_total"]

    assert feature_latency.events[0] == _MetricEvent(
        "observe",
        {"instrument": "AAPL", "feature_type": "technical"},
        0.12,
    )
    assert model_errors.events[0] == _MetricEvent(
        "inc",
        {"model": "xgb", "instrument": "AAPL", "error_type": "timeout"},
        1.0,
    )


def test_prediction_timer_records_success_and_error_paths(
    collector_with_fakes: tuple[MLMetricsCollector, _FakeMetricsManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector, fake_manager = collector_with_fakes
    values = iter([10.0, 10.2, 20.0, 20.8])
    monkeypatch.setattr(collector_module.time, "perf_counter", lambda: next(values))

    with collector.time_prediction("ensemble", "QQQ") as timer:
        timer.set_prediction("hold", 0.6)

    with pytest.raises(RuntimeError, match="boom"):
        with collector.time_prediction("ensemble", "QQQ") as timer:
            timer.set_prediction("sell", 0.3)
            raise RuntimeError("boom")

    predictions = fake_manager.metrics_by_name["nautilus_ml_predictions_total"]
    latency = fake_manager.metrics_by_name["nautilus_ml_prediction_latency_seconds"]

    assert [event.labels["status"] for event in predictions.events] == ["success", "error"]
    assert [event.value for event in latency.events] == pytest.approx([0.2, 0.8])


def test_feature_timer_records_latency(
    collector_with_fakes: tuple[MLMetricsCollector, _FakeMetricsManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector, fake_manager = collector_with_fakes
    values = iter([1.0, 1.05])
    monkeypatch.setattr(collector_module.time, "perf_counter", lambda: next(values))

    with collector.time_feature_computation("NVDA", "microstructure"):
        pass

    feature_latency = fake_manager.metrics_by_name["nautilus_ml_feature_computation_latency_seconds"]
    assert feature_latency.events[0].operation == "observe"
    assert feature_latency.events[0].labels == {"instrument": "NVDA", "feature_type": "microstructure"}
    assert feature_latency.events[0].value == pytest.approx(0.05)


def test_disabled_collector_operations_are_noop() -> None:
    collector = MLMetricsCollector(MonitoringConfig(enabled=False))
    fake_metric = _FakeMetric()
    collector._ml_predictions_total = fake_metric
    collector._ml_prediction_latency_seconds = fake_metric
    collector._ml_model_confidence = fake_metric
    collector._ml_feature_computation_latency_seconds = fake_metric
    collector._ml_model_errors_total = fake_metric

    collector.record_prediction("m", "i", "buy", 0.1, 0.9, success=True)
    collector.record_feature_computation("i", "technical", 0.1)
    collector.record_error("m", "i", "timeout")

    assert collector.enabled is False
    assert collector.config.enabled is False
    assert fake_metric.events == []


def test_initialize_metrics_returns_early_when_module_prometheus_flag_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_state = {"called": False}

    def _default() -> Any:
        call_state["called"] = True
        return _FakeMetricsManager()

    monkeypatch.setattr(base_module, "HAS_PROMETHEUS", True)
    monkeypatch.setattr(collector_module, "HAS_PROMETHEUS", False)
    monkeypatch.setattr(
        metrics_manager_module.MetricsManager,
        "default",
        staticmethod(_default),
    )

    collector = MLMetricsCollector(MonitoringConfig(enabled=True))

    assert collector.enabled is True
    assert collector.get_metric_count() == 0
    assert call_state["called"] is False
