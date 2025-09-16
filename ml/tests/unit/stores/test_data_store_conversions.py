"""
Unit tests for DataStore conversion helpers (_df_to_predictions/_df_to_signals).

Ensures correct mapping from dict/pandas (and polars when available) to dataclasses.

"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from ml.stores.base import ModelPrediction, StrategySignal
from ml.stores.data_store import DataStore


def _mk_store() -> DataStore:
    reg = MagicMock()
    from ml.stores.base import DummyStore

    return DataStore(
        connection_string="sqlite:///:memory:",
        registry=reg,
        feature_store=DummyStore(),
        model_store=DummyStore(),
        strategy_store=DummyStore(),
        fail_on_validation_error=False,
    )


def test_df_to_predictions_from_dicts() -> None:
    ds = _mk_store()
    rows: list[dict[str, Any]] = [
        {
            "model_id": "m1",
            "instrument_id": "EUR/USD",
            "prediction": 0.5,
            "confidence": 0.8,
            "ts_event": 123,
            "ts_init": 123,
            "features_used": {"f": 1.0},
        },
    ]
    preds = ds._df_to_predictions(rows)
    assert len(preds) == 1
    p = preds[0]
    assert isinstance(p, ModelPrediction)
    assert p.model_id == "m1"
    assert p.instrument_id == "EUR/USD"
    assert p.prediction == 0.5


def test_df_to_predictions_from_pandas() -> None:
    ds = _mk_store()
    df = pd.DataFrame(
        [
            {
                "model_id": "m2",
                "instrument_id": "SPY",
                "prediction": 0.6,
                "confidence": 0.9,
                "ts_event": 456,
                "ts_init": 456,
            },
        ],
    )
    preds = ds._df_to_predictions(df)
    assert len(preds) == 1
    assert preds[0].model_id == "m2"
    assert preds[0].instrument_id == "SPY"


def test_df_to_signals_from_dicts() -> None:
    ds = _mk_store()
    rows: list[dict[str, Any]] = [
        {
            "strategy_id": "s1",
            "instrument_id": "EUR/USD",
            "signal_type": "BUY",
            "strength": 0.7,
            "ts_event": 789,
            "ts_init": 789,
            "risk_metrics": {"r": 1},
        },
    ]
    sigs = ds._df_to_signals(rows)
    assert len(sigs) == 1
    s = sigs[0]
    assert isinstance(s, StrategySignal)
    assert s.strategy_id == "s1"
    assert s.instrument_id == "EUR/USD"
    assert s.signal_type == "BUY"


@pytest.mark.skipif(
    pytest.importorskip("polars", reason="polars not installed") is None,
    reason="polars missing",
)
def test_df_to_predictions_from_polars() -> None:
    import polars as pl

    ds = _mk_store()
    pdf = pl.DataFrame(
        {
            "model_id": ["mP"],
            "instrument_id": ["AAPL"],
            "prediction": [0.4],
            "confidence": [0.6],
            "ts_event": [999],
            "ts_init": [999],
        },
    )
    preds = ds._df_to_predictions(pdf)
    assert len(preds) == 1
    assert preds[0].model_id == "mP"
    assert preds[0].instrument_id == "AAPL"
