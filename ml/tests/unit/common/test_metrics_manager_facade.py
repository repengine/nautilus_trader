from __future__ import annotations

from typing import Any

import pytest

from ml.common.metrics_manager import MetricsManager


class _FakeCtr:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], float]] = []

    def labels(self, **kwargs: object) -> _FakeCtr:
        self._labels = dict(kwargs)
        return self

    def inc(self, amount: float = 1.0) -> None:
        self.calls.append((getattr(self, "_labels", {}), float(amount)))


class _FakeGauge:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], float]] = []

    def labels(self, **kwargs: object) -> _FakeGauge:
        self._labels = dict(kwargs)
        return self

    def set(self, value: float) -> None:
        self.calls.append((getattr(self, "_labels", {}), float(value)))


def test_metrics_manager_inc_and_gauge_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ctr = _FakeCtr()
    fake_gauge = _FakeGauge()

    # Patch the imported aliases inside metrics_manager module
    import ml.common.metrics_manager as mm_mod

    monkeypatch.setattr(mm_mod, "_get_counter", lambda *a, **k: fake_ctr)
    monkeypatch.setattr(mm_mod, "_get_gauge", lambda *a, **k: fake_gauge)

    mm = MetricsManager()

    # First inc creates and caches counter; second uses cache
    mm.inc(
        "nautilus_ml_test_counter",
        "desc",
        labels={"component": "x", "reason": "y"},
        amount=2.0,
    )
    mm.inc(
        "nautilus_ml_test_counter",
        "desc",
        labels={"component": "x", "reason": "y"},
        amount=3.0,
    )

    assert len(fake_ctr.calls) == 2
    assert fake_ctr.calls[0][0] == {"component": "x", "reason": "y"}

    mm.set_gauge(
        "nautilus_ml_test_gauge",
        "desc",
        5.0,
        labels={"component": "z"},
    )
    assert len(fake_gauge.calls) == 1
    assert fake_gauge.calls[0][0] == {"component": "z"}
