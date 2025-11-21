from __future__ import annotations

import numpy as np
import pytest

from ml.features.earnings import earnings_features as ef
from ml.features.earnings import reset_earnings_metrics_state


@pytest.fixture(autouse=True)
def reset_metrics_cache() -> None:
    reset_earnings_metrics_state()


def test_earnings_metrics_enabled_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ML_EARNINGS_ENABLE_METRICS", raising=False)
    assert ef.earnings_metrics_enabled() is False


def test_earnings_metrics_enabled_from_mapping() -> None:
    env = {"ML_EARNINGS_ENABLE_METRICS": "true"}
    assert ef.earnings_metrics_enabled(env=env) is True


def test_metrics_enabled_handles_false_mapping() -> None:
    env = {"ML_EARNINGS_ENABLE_METRICS": "0"}
    assert ef.earnings_metrics_enabled(env=env) is False


class _StubMetric:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def labels(self, **kwargs: object) -> _StubMetric:
        self.calls.append("labels")
        return self

    def inc(self, amount: float = 1.0) -> None:
        self.calls.append(f"inc:{amount}")

    def observe(self, value: float) -> None:
        self.calls.append(f"observe:{value}")


class _StubMetricsManager:
    def __init__(self) -> None:
        self.counter_calls = 0
        self.histogram_calls = 0
        self.counter_metric = _StubMetric()
        self.hist_metric = _StubMetric()

    def counter(self, *args: object, **kwargs: object) -> _StubMetric:
        self.counter_calls += 1
        return self.counter_metric

    def histogram(self, *args: object, **kwargs: object) -> _StubMetric:
        self.histogram_calls += 1
        return self.hist_metric


def test_compute_surprise_incremental_without_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_EARNINGS_ENABLE_METRICS", "0")
    reset_earnings_metrics_state()

    stub_manager = _StubMetricsManager()
    from ml.common.metrics_manager import MetricsManager

    monkeypatch.setattr(MetricsManager, "_DEFAULT", stub_manager, raising=False)

    result = ef.compute_earnings_surprise_incremental(2.5, 2.0)

    assert pytest.approx(result["eps_surprise_q0"], rel=1e-9) == 0.5
    assert stub_manager.counter_calls == 0
    assert stub_manager.histogram_calls == 0


def test_compute_surprise_batch_with_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_EARNINGS_ENABLE_METRICS", "1")
    reset_earnings_metrics_state()

    stub_manager = _StubMetricsManager()
    from ml.common.metrics_manager import MetricsManager

    monkeypatch.setattr(MetricsManager, "_DEFAULT", stub_manager, raising=False)

    ef.compute_earnings_surprise_batch(
        np.array([2.5, 2.6]),
        np.array([2.0, 2.4]),
    )

    assert stub_manager.counter_calls > 0
    assert stub_manager.histogram_calls > 0


def test_metrics_flag_reflects_env_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_EARNINGS_ENABLE_METRICS", "1")
    reset_earnings_metrics_state()
    assert ef.earnings_metrics_enabled() is True

    monkeypatch.setenv("ML_EARNINGS_ENABLE_METRICS", "0")
    assert ef.earnings_metrics_enabled() is False
