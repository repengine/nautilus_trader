from __future__ import annotations

from typing import Any

from ml.actors.signal import MLSignalActor
from ml.actors.signal import ThresholdSignalStrategy
from ml.actors.signal import MLSignalActorConfig as _MLSignalActorConfig
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId


def test_invalid_policy_falls_back_to_threshold(tmp_path: Any) -> None:
    inst = InstrumentId.from_str("EURUSD.SIM")
    bar_type = BarType.from_str("EURUSD.SIM-1-MINUTE-MID-EXTERNAL")
    cfg = _MLSignalActorConfig(
        model_path=str(tmp_path / "model.onnx"),
        model_id="demo",
        bar_type=bar_type,
        instrument_id=inst,
        use_dummy_stores=True,
        prediction_threshold=0.33,
        signal_strategy="threshold",
    )
    actor = MLSignalActor(cfg)
    # invalid policy path should be ignored and actor should use built-ins
    actor._model_metadata = {"decision_policy": "not.a.real.Adapter", "decision_config": {}}
    strat = actor._create_strategy()
    assert isinstance(strat, ThresholdSignalStrategy)
    assert abs(strat.threshold - 0.33) < 1e-9
