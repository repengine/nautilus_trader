"""Tests for the file-backed store implementations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml.actors.base import MLSignal
from ml.stores.file_backed import FileDataStore
from ml.stores.file_backed import FileFeatureStore
from ml.stores.file_backed import FileModelStore
from ml.stores.file_backed import FileStrategyStore
from nautilus_trader.model.identifiers import InstrumentId


@pytest.fixture()
def file_root(tmp_path: Path) -> Path:
    root = tmp_path / "file_store"
    root.mkdir()
    return root


def test_file_feature_store_round_trip(file_root: Path) -> None:
    store = FileFeatureStore(base_path=file_root)
    store.write_features(
        feature_set_id="fs1",
        instrument_id="EURUSD",
        features={"v1": 1.0},
        ts_event=100,
        ts_init=90,
    )
    store.flush()

    reloaded = FileFeatureStore(base_path=file_root)
    snapshot = reloaded.get_latest_at_or_before("EURUSD", 100)
    assert snapshot == {"v1": 1.0}


def test_file_model_store_read_predictions(file_root: Path) -> None:
    store = FileModelStore(base_path=file_root)
    store.write_prediction(
        model_id="modelA",
        instrument_id="EURUSD",
        prediction=0.7,
        confidence=0.8,
        features={"v1": 1.0},
        inference_time_ms=0.5,
        ts_event=200,
    )
    store.flush()

    reloaded = FileModelStore(base_path=file_root)
    frame = reloaded.read_predictions("modelA", "EURUSD", start_ns=0, end_ns=300)
    assert not frame.empty
    assert frame.iloc[0]["prediction"] == pytest.approx(0.7)


def test_file_strategy_store_signal_distribution(file_root: Path) -> None:
    store = FileStrategyStore(base_path=file_root)
    store.write_signal(
        strategy_id="strat",
        instrument_id="EURUSD",
        signal_type="BUY",
        strength=1.0,
        model_predictions={"modelA": 0.7},
        risk_metrics={"confidence": 0.8},
        execution_params={"target": "enter"},
        ts_event=300,
    )
    store.flush()

    reloaded = FileStrategyStore(base_path=file_root)
    distribution = reloaded.get_signal_distribution("strat")
    assert distribution["BUY"] == 1


def test_file_data_store_writes_jsonl(file_root: Path) -> None:
    store = FileDataStore(base_path=file_root)
    frame = pd.DataFrame(
        {
            "ts_event": [10, 20, 30],
            "price": [1.0, 1.1, 1.2],
        },
    )
    written = store.write_ingestion(
        dataset_id="bars",
        records=frame,
        source="historical",
        run_id="run-1",
        instrument_id="EURUSD",
    )
    assert written == 3
    store.flush()

    events_path = file_root / "events.jsonl"
    assert events_path.exists()
    content = events_path.read_text(encoding="utf-8")
    assert "correlation_id" in content


def test_portfolio_allocation_integration(tmp_path: Path) -> None:
    # Smoke-test portfolio allocation helper: ensure scaling happens without errors.
    from ml.strategies.portfolio import PortfolioManager

    # Minimal strategy double with portfolio manager to ensure allocate_signals accepts MLSignal
    signal = MLSignal(
        instrument_id=InstrumentId.from_str("EURUSD.NYSE"),
        model_id="modelA",
        prediction=0.6,
        confidence=0.7,
        ts_event=100,
        ts_init=100,
    )
    pm = PortfolioManager()
    allocations = pm.allocate_signals([signal], available_capital=1000.0)
    assert signal.instrument_id in allocations
