"""
Exercise feature-time metric path in MLSignalActor._compute_features.

Stubs indicator manager and feature engineer to avoid heavy runtime and monkeypatches
the module-level histogram to assert observation.

"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np

import ml.actors.signal as signal_mod
from ml.actors.signal import MLSignalActor
from ml.actors.signal import ThresholdSignalStrategy
from ml.tests.utils.stubs import SignalActorHarness


class _DummyHist:
    def __init__(self) -> None:
        self.observed: list[tuple[dict[str, str], float]] = []

    def labels(self, **labels: str) -> _DummyHist:
        """
        Return self to allow .observe() chaining.
        """
        self._labels = labels
        return self

    def observe(self, value: float) -> None:
        self.observed.append((getattr(self, "_labels", {}), value))


def _stub_bar() -> Any:
    # Minimal bar with floats accepted by _compute_features
    return SimpleNamespace(close=1.0, volume=100.0, high=1.1, low=0.9)


def test_compute_features_records_feature_time() -> None:
    indicator_manager = SimpleNamespace(
        update_from_bar=lambda bar: None,
        all_initialized=lambda: True,
    )
    feature_engineer = SimpleNamespace(
        calculate_features_online=lambda **kwargs: np.zeros(2, dtype=np.float32),
    )

    harness = SignalActorHarness(
        _signal_strategy=ThresholdSignalStrategy(threshold=0.0),
        _signal_config=SimpleNamespace(signal_strategy="threshold", min_signal_separation_bars=0),
        _config=SimpleNamespace(max_feature_latency_ms=1_000.0),
        id="actor-1",
        _feature_store=None,
        _indicator_manager=indicator_manager,
        _feature_engineer=feature_engineer,
        _feature_buffer=np.zeros(2, dtype=np.float32),
        _feature_set_id="fs1",
    )
    actor = cast(MLSignalActor, harness)

    # Monkeypatch histogram
    hist = _DummyHist()
    setattr(signal_mod, "_feature_time_by_feature_set_metric", hist)

    # Call unbound method with typed harness
    res = MLSignalActor._compute_features(actor, _stub_bar())
    assert isinstance(res, np.ndarray)
    assert hist.observed, "Expected feature-time histogram observation"
    labels, _value = hist.observed[-1]
    assert labels.get("feature_set_id") == "fs1"
