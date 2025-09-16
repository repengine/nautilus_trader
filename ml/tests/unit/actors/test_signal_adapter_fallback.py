from __future__ import annotations

from typing import Any

from ml.actors.signal import MLSignalActor
from ml.actors.signal import ThresholdSignalStrategy
from ml.actors.signal import MLSignalActorConfig as _MLSignalActorConfig
from ml.tests.builders import MLConfigBuilder
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId


def test_invalid_policy_falls_back_to_threshold(
    tmp_path: Any,
    default_instrument_id,
    default_bar_type,
) -> None:
    cfg = MLConfigBuilder.signal_config(
        model_path=str(tmp_path / "model.onnx"),
        model_id="demo",
        bar_type=default_bar_type,
        instrument_id=default_instrument_id,
        prediction_threshold=0.33,
        signal_strategy="threshold",
    )
    actor = MLSignalActor(cfg)
    # invalid policy path should be ignored and actor should use built-ins
    actor._model_metadata = {"decision_policy": "not.a.real.Adapter", "decision_config": {}}
    strat = actor._create_strategy()
    assert isinstance(strat, ThresholdSignalStrategy)
    assert abs(strat.threshold - 0.33) < 1e-9
