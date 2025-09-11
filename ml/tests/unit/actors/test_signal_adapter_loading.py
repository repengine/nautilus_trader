from __future__ import annotations

from typing import Any

import numpy as np

from ml.actors.adapters import build_strategy_from_policy
from ml.actors.signal import ThresholdSignalStrategy


def test_manifest_adapter_class_dynamic_threshold() -> None:
    class _FakeConfig:
        prediction_threshold = 0.42

    class _FakeActor:
        _adaptive_threshold = 0.42
        _config = _FakeConfig()

    strat = build_strategy_from_policy(
        policy_path="ml.actors.adapters.DynamicThresholdAdapter",
        actor=_FakeActor(),
        config={},
    )
    assert isinstance(strat, ThresholdSignalStrategy)

    # Ensure adapter used actor threshold
    # The strategy exposes `.threshold` in our implementation
    assert np.isclose(strat.threshold, 0.42)
