from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from ml.actors.adapters import build_strategy_from_policy
from ml.actors.signal import ThresholdSignalStrategy
from ml.tests.utils.stubs import SignalActorHarness


def test_manifest_adapter_class_dynamic_threshold() -> None:
    harness = SignalActorHarness(
        _signal_strategy=ThresholdSignalStrategy(threshold=0.0),
        _signal_config=SimpleNamespace(signal_strategy="threshold", min_signal_separation_bars=0),
        _config=SimpleNamespace(prediction_threshold=0.42),
        id=SimpleNamespace(value="actor-1"),
        _adaptive_threshold=0.42,
    )

    strat = build_strategy_from_policy(
        policy_path="ml.actors.adapters.DynamicThresholdAdapter",
        actor=harness.as_actor(),
        config={},
    )
    assert isinstance(strat, ThresholdSignalStrategy)

    # Ensure adapter used actor threshold
    # The strategy exposes `.threshold` in our implementation
    assert np.isclose(strat.threshold, 0.42)
