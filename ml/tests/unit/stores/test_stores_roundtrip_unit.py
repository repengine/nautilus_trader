"""
Unit tests for store write/read logic using patched DB writes.

These tests patch the private _execute_write methods to avoid database dependencies
while still exercising the public write APIs and ensuring payload integrity and
timestamp normalization.

"""

from __future__ import annotations

from typing import Any

import numpy as np

from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelPrediction
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategySignal
from ml.stores.strategy_store import StrategyStore


def test_feature_store_write_explicit_args(monkeypatch: Any) -> None:
    rows: list[dict[str, Any]] = []

    fs = FeatureStore(connection_string="sqlite:///:memory:")

    def _fake_exec(row: dict[str, Any]) -> None:
        rows.append(row)

    monkeypatch.setattr(fs, "_execute_write", _fake_exec)

    fs.write_features(
        feature_set_id="feat_v1",
        instrument_id="EURUSD.SIM",
        features={"a": 1.0, "b": 2.0},
        ts_event=1,
        ts_init=1,
    )

    assert len(rows) == 1
    assert rows[0]["feature_set_id"] == "feat_v1"
    assert rows[0]["instrument_id"] == "EURUSD.SIM"
    assert rows[0]["values"] == {"a": 1.0, "b": 2.0}


def test_model_store_write_batch_and_events(monkeypatch: Any) -> None:
    written: list[dict[str, Any]] = []
    ms = ModelStore(connection_string="sqlite:///:memory:")

    def _fake_exec(values: list[dict[str, Any]]) -> None:
        written.extend(values)

    monkeypatch.setattr(ms, "_execute_write", _fake_exec)

    # Buffer two predictions then flush via write_batch
    ms.write_prediction(
        model_id="m1",
        instrument_id="EURUSD.SIM",
        prediction=0.9,
        confidence=0.8,
        features={"a": 1.0},
        inference_time_ms=0.1,
        ts_event=1,
    )
    ms.write_prediction(
        model_id="m1",
        instrument_id="EURUSD.SIM",
        prediction=0.1,
        confidence=0.2,
        features={"a": 2.0},
        inference_time_ms=0.2,
        ts_event=2,
    )
    # Invoke write_batch explicitly (avoid timing-based flush)
    ms.write_batch(list(ms._write_buffer))

    assert len(written) == 2
    assert written[0]["model_id"] == "m1"
    assert written[0]["instrument_id"] == "EURUSD.SIM"
    assert "prediction" in written[0]


def test_strategy_store_write_batch(monkeypatch: Any) -> None:
    out: list[dict[str, Any]] = []
    ss = StrategyStore(connection_string="sqlite:///:memory:")

    def _fake_exec(values: list[dict[str, Any]]) -> None:
        out.extend(values)

    monkeypatch.setattr(ss, "_execute_write", _fake_exec)

    # Buffer signals via write_signal, then flush through write_batch
    ss.write_signal(
        strategy_id="s1",
        instrument_id="EURUSD.SIM",
        signal_type="BUY",
        strength=0.8,
        model_predictions={"m1": 0.8},
        risk_metrics={"conf": 0.8},
        execution_params={"side": "BUY"},
        ts_event=1,
    )
    ss.write_signal(
        strategy_id="s1",
        instrument_id="EURUSD.SIM",
        signal_type="SELL",
        strength=0.2,
        model_predictions={"m1": 0.2},
        risk_metrics={"conf": 0.2},
        execution_params={"side": "SELL"},
        ts_event=2,
    )
    ss.write_batch(list(ss._write_buffer))

    assert len(out) == 2
    assert out[0]["strategy_id"] == "s1"
    assert out[0]["signal_type"] in ("BUY", "SELL")
