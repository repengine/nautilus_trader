from __future__ import annotations

from typing import Any

import pytest

from ml.stores.base import FeatureData
from ml.stores.base import ModelPrediction
from ml.stores.base import StrategySignal
from ml.stores.common.data_frame_converters import data_frame_to_feature_data
from ml.stores.common.data_frame_converters import data_frame_to_predictions
from ml.stores.common.data_frame_converters import data_frame_to_signals


class _IterRowsFrame:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def iter_rows(self, named: bool = True):
        return iter(self._rows)


class _IterrowsFrame:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def iterrows(self):
        yield from enumerate(self._rows)


def test_data_frame_to_feature_data_from_iter_rows() -> None:
    frame = _IterRowsFrame([{"ts_event": 10, "values": {"alpha": 1.5}}])
    features = data_frame_to_feature_data(frame, instrument_id="SPY")

    assert len(features) == 1
    feature = features[0]
    assert isinstance(feature, FeatureData)
    assert feature.instrument_id == "SPY"
    assert feature.feature_set_id == "default"
    assert feature.values == {"alpha": 1.5}
    assert feature._ts_event == 10
    assert feature._ts_init == 10


def test_data_frame_to_feature_data_requires_ts_event() -> None:
    frame = _IterrowsFrame([{"values": {"alpha": 1.0}}])

    with pytest.raises(ValueError, match="ts_event"):
        data_frame_to_feature_data(frame, instrument_id="SPY")


def test_data_frame_to_predictions_falls_back_to_value_and_features() -> None:
    rows = [
        {
            "model_id": "m1",
            "instrument_id": "AAPL",
            "value": 0.42,
            "features": {"x": 2.0},
            "ts_event": 5,
        },
    ]
    preds = data_frame_to_predictions(rows)

    assert len(preds) == 1
    pred = preds[0]
    assert isinstance(pred, ModelPrediction)
    assert pred.prediction == 0.42
    assert pred.features_used == {"x": 2.0}
    assert pred._ts_init == 5


def test_data_frame_to_predictions_requires_ts_event() -> None:
    with pytest.raises(ValueError, match="ts_event"):
        data_frame_to_predictions([{"model_id": "m1", "instrument_id": "AAPL"}])


def test_data_frame_to_signals_from_iterrows_normalizes_metadata() -> None:
    frame = _IterrowsFrame(
        [
            {
                "strategy_id": "strat-1",
                "instrument_id": "EUR/USD",
                "signal_type": "BUY",
                "signal_value": 0.25,
                "ts_event": 7,
                "decision_metadata": {"policy": "basic"},
            },
        ],
    )
    signals = data_frame_to_signals(frame)

    assert len(signals) == 1
    signal = signals[0]
    assert isinstance(signal, StrategySignal)
    assert signal.strength == 0.25
    assert signal.decision_metadata["policy"] == "basic"
    assert signal.decision_metadata["version"] == "v1"


def test_data_frame_to_signals_requires_decision_metadata() -> None:
    rows = [
        {
            "strategy_id": "strat-2",
            "instrument_id": "AAPL",
            "signal_type": "SELL",
            "ts_event": 9,
        },
    ]

    with pytest.raises(ValueError, match="decision_metadata"):
        data_frame_to_signals(rows)
