from __future__ import annotations

from typing import Any
import pytest

from ml.common.metrics_manager import MetricsManager


def test_metrics_manager_counter_and_gauge(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeCtr:
        def __init__(self) -> None:
            self.labeled: dict[str, Any] | None = None
            self.count: float = 0.0

        def labels(self, **kwargs: object) -> _FakeCtr:
            self.labeled = dict(kwargs)
            return self

        def inc(self, amount: float = 1.0) -> None:
            self.count += float(amount)

    class _FakeGauge:
        def __init__(self) -> None:
            self.labeled: dict[str, Any] | None = None
            self.value: float = 0.0

        def labels(self, **kwargs: object) -> _FakeGauge:
            self.labeled = dict(kwargs)
            return self

        def set(self, value: float) -> None:
            self.value = float(value)

    monkeypatch.setattr("ml.common.metrics_manager._get_counter", lambda n, d, l: _FakeCtr())
    monkeypatch.setattr("ml.common.metrics_manager._get_gauge", lambda n, d, l: _FakeGauge())

    mm = MetricsManager()
    mm.inc(
        "ctr",
        "desc",
        labels={"a": "b"},
        amount=2.0,
        labelnames=("a",),
    )
    mm.set_gauge(
        "g",
        "desc",
        3.14,
        labels={"x": 1},
        labelnames=("x",),
    )

    ctr = mm._cache["ctr::ctr::('a',)"]
    assert isinstance(ctr, _FakeCtr) and ctr.count == 2.0 and ctr.labeled == {"a": "b"}
    g = mm._cache["gauge::g::('x',)"]
    assert isinstance(g, _FakeGauge) and g.value == 3.14 and g.labeled == {"x": 1}
