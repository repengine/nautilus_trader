"""
MetricsManager histogram observe test.

Patches the bootstrap get_histogram to return a fake with labels().observe(), then
verifies that observe flows through with expected labels and value.

"""

from __future__ import annotations

from typing import Any
import pytest

from ml.common.metrics_manager import MetricsManager


def test_metrics_manager_histogram_observe(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {"get_hist_count": 0}

    class _FakeHistogram:
        def __init__(self) -> None:
            self.labeled: dict[str, Any] | None = None
            self.observed: float | None = None

        def labels(self, **kwargs: object) -> _FakeHistogram:
            self.labeled = dict(kwargs)
            return self

        def observe(self, amount: float) -> None:
            self.observed = float(amount)

    def _fake_get_histogram(
        name: str,
        description: str,
        labelnames: list[str] | None = None,
        *,
        buckets: tuple[float, ...] | None = None,
    ) -> _FakeHistogram:  # noqa: E501
        calls["get_hist_count"] += 1
        return _FakeHistogram()

    # Patch bootstrap getter that MetricsManager.histogram delegates to
    monkeypatch.setattr("ml.common.metrics_manager._get_histogram", _fake_get_histogram)

    mm = MetricsManager()
    labels = {"component": "ml_actor", "stage": "PREDICTION_EMITTED"}
    mm.observe(
        "nautilus_ml_latency_ms",
        "Latency in ms",
        12.3,
        labels=labels,
        labelnames=("component", "stage"),
    )

    # The patched get_histogram is called once
    assert calls["get_hist_count"] == 1

    # Retrieve the cached histogram and inspect the fake's state
    hist = mm._cache["hist::nautilus_ml_latency_ms::('component', 'stage')"]
    assert isinstance(hist, _FakeHistogram)
    assert hist.labeled == labels
    assert hist.observed == 12.3
