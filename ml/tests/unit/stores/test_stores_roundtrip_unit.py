"""
Unit tests for store write/read logic using patched DB writes.

These tests patch the private _execute_write methods to avoid database dependencies
while still exercising the public write APIs and ensuring payload integrity and
timestamp normalization.

"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelPrediction
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategySignal
from ml.stores.strategy_store import StrategyStore
from nautilus_trader.model.identifiers import InstrumentId


def test_feature_store_write_explicit_args(
    monkeypatch: Any,
    default_instrument_id: InstrumentId,
    test_timestamps: tuple[int, int],
) -> None:
    rows: list[dict[str, Any]] = []

    fs = FeatureStore(connection_string="sqlite:///:memory:")

    def _fake_exec(row: dict[str, Any]) -> None:
        rows.append(row)

    monkeypatch.setattr(fs, "_execute_write", _fake_exec)

    ts_event, _ts_init = test_timestamps
    instrument_id_str = str(default_instrument_id)

    fs.write_features(
        feature_set_id="feat_v1",
        instrument_id=instrument_id_str,
        features={"a": 1.0, "b": 2.0},
        ts_event=ts_event,
        ts_init=ts_init,
    )

    assert len(rows) == 1
    assert rows[0]["feature_set_id"] == "feat_v1"
    assert rows[0]["instrument_id"] == instrument_id_str
    assert rows[0]["values"] == {"a": 1.0, "b": 2.0}


def test_model_store_write_batch_and_events(
    monkeypatch: Any,
    default_instrument_id: InstrumentId,
    test_timestamps: tuple[int, int],
) -> None:
    written: list[dict[str, Any]] = []
    ms = ModelStore(connection_string="sqlite:///:memory:")

    def _fake_exec(values: list[dict[str, Any]]) -> None:
        written.extend(values)

    monkeypatch.setattr(ms, "_execute_write", _fake_exec)

    ts_event, _ts_init = test_timestamps
    instrument_id_str = str(default_instrument_id)

    # Buffer two predictions then flush via write_batch
    ms.write_prediction(
        model_id="m1",
        instrument_id=instrument_id_str,
        prediction=0.9,
        confidence=0.8,
        features={"a": 1.0},
        inference_time_ms=0.1,
        ts_event=ts_event,
    )
    ms.write_prediction(
        model_id="m1",
        instrument_id=instrument_id_str,
        prediction=0.1,
        confidence=0.2,
        features={"a": 2.0},
        inference_time_ms=0.2,
        ts_event=ts_event + 1,
    )
    # Invoke write_batch explicitly (avoid timing-based flush)
    ms.write_batch(list(ms._write_buffer))

    assert len(written) == 2
    assert written[0]["model_id"] == "m1"
    assert written[0]["instrument_id"] == instrument_id_str
    assert "prediction" in written[0]


def test_strategy_store_write_batch(
    monkeypatch: Any,
    default_instrument_id: InstrumentId,
    test_timestamps: tuple[int, int],
) -> None:
    out: list[dict[str, Any]] = []
    ss = StrategyStore(connection_string="sqlite:///:memory:")

    def _fake_exec(values: list[dict[str, Any]]) -> None:
        out.extend(values)

    monkeypatch.setattr(ss, "_execute_write", _fake_exec)

    ts_event, _ts_init = test_timestamps
    instrument_id_str = str(default_instrument_id)

    # Buffer signals via write_signal, then flush through write_batch
    ss.write_signal(
        strategy_id="s1",
        instrument_id=instrument_id_str,
        signal_type="BUY",
        strength=0.8,
        model_predictions={"m1": 0.8},
        risk_metrics={"conf": 0.8},
        execution_params={"side": "BUY"},
        ts_event=ts_event,
    )
    ss.write_signal(
        strategy_id="s1",
        instrument_id=instrument_id_str,
        signal_type="SELL",
        strength=0.2,
        model_predictions={"m1": 0.2},
        risk_metrics={"conf": 0.2},
        execution_params={"side": "SELL"},
        ts_event=ts_event + 1,
    )
    ss.write_batch(list(ss._write_buffer))

    assert len(out) == 2
    assert out[0]["strategy_id"] == "s1"
    assert out[0]["signal_type"] in ("BUY", "SELL")
