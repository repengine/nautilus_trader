"""Tests for the file-backed store implementations."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ml._imports import HAS_POLARS
from ml.actors.base import MLSignal
from ml.stores.base import StrategyReplaySummary
from ml.stores.file_backed import FileEarningsStore
from ml.stores.file_backed import FileDataStore
from ml.stores.file_backed import FileFeatureStore
from ml.stores.file_backed import FileModelStore
from ml.stores.file_backed import FileStrategyStore
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.test_kit.stubs.events import TestEventStubs
from nautilus_trader.test_kit.stubs.execution import TestExecStubs


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
        decision_metadata={"version": "v1"},
        ts_event=300,
    )
    store.flush()

    reloaded = FileStrategyStore(base_path=file_root)
    distribution = reloaded.get_signal_distribution("strat")
    assert distribution["BUY"] == 1


def test_file_strategy_store_writes_order_events(file_root: Path) -> None:
    store = FileStrategyStore(base_path=file_root)
    order = TestExecStubs.limit_order()
    ts_event = 1_700_000_000_000_000_000
    event = TestEventStubs.order_submitted(order, ts_event=ts_event)

    store.write_order_event(event, is_live=True, run_id="run-1")
    store.flush()

    events_path = file_root / "order_events.jsonl"
    assert events_path.exists()
    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    payload = json.loads(lines[-1])
    assert payload["event_type"] == "OrderSubmitted"
    assert payload["instrument_id"] == str(order.instrument_id)
    assert payload["ts_event"] == ts_event
    assert payload["is_live"] is True
    assert payload["run_id"] == "run-1"
    assert payload["ingested_at_ns"] is not None


def test_file_strategy_store_writes_risk_halt_events(file_root: Path) -> None:
    store = FileStrategyStore(base_path=file_root)
    ts_event = 1_700_000_000_000_000_000

    store.write_risk_halt_event(
        strategy_id="strat",
        instrument_id="EURUSD",
        event_type="halted",
        reason="daily_loss_limit",
        detail="Daily loss limit reached",
        ts_event=ts_event,
        is_live=True,
        run_id="run-2",
    )
    store.flush()

    events_path = file_root / "risk_halt_events.jsonl"
    assert events_path.exists()
    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    payload = json.loads(lines[-1])
    assert payload["event_type"] == "halted"
    assert payload["reason"] == "daily_loss_limit"
    assert payload["instrument_id"] == "EURUSD"
    assert payload["ts_event"] == ts_event
    assert payload["is_live"] is True
    assert payload["run_id"] == "run-2"
    assert payload["ingested_at_ns"] is not None


def test_file_strategy_store_writes_replay_summary(file_root: Path) -> None:
    store = FileStrategyStore(base_path=file_root)
    summary = StrategyReplaySummary(
        run_id="run-3",
        instrument_ids=["EURUSD"],
        total_orders=4,
        total_fills=3,
        total_halts=1,
        total_sizing_rejects=2,
        total_positions=1,
        started_ns=10,
        finished_ns=20,
        _ts_event=30,
        _ts_init=30,
    )
    store.write_replay_summary(summary)
    store.flush()

    summary_path = file_root / "replay_summary.jsonl"
    assert summary_path.exists()
    lines = summary_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    payload = json.loads(lines[-1])
    assert payload["run_id"] == "run-3"
    assert payload["total_orders"] == 4
    assert payload["total_fills"] == 3
    assert payload["total_halts"] == 1
    assert payload["total_sizing_rejects"] == 2
    assert payload["instrument_ids"] == ["EURUSD"]


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


@pytest.mark.skipif(not HAS_POLARS, reason="polars required for file earnings store")
def test_file_earnings_store_round_trip(file_root: Path) -> None:
    store = FileEarningsStore(base_path=file_root)
    ts_event = 1_700_000_000_000_000_000
    store.write_actuals(
        ticker="AAPL",
        period_end="2024-09-30",
        filing_date="2024-11-01",
        eps_diluted=1.64,
        revenue=94.9e9,
        ts_event=ts_event,
        ts_init=ts_event,
        eps_basic=1.60,
    )
    store.write_estimates(
        ticker="AAPL",
        estimate_date="2024-09-15",
        period_end="2024-09-30",
        eps_consensus=1.55,
        ts_event=ts_event - 1_000,
        ts_init=ts_event - 1_000,
        revenue_consensus=92.0e9,
        num_analysts=32,
    )
    store.flush()

    reloaded = FileEarningsStore(base_path=file_root)
    actuals = reloaded.get_actuals("AAPL", as_of_ts=ts_event + 1)
    assert actuals and actuals[0]["eps_diluted"] == pytest.approx(1.64)
    assert reloaded.get_actuals("AAPL", as_of_ts=ts_event) == []

    estimate = reloaded.get_estimates("AAPL", "2024-09-30", as_of_ts=ts_event)
    assert estimate is not None
    assert estimate["eps_consensus"] == pytest.approx(1.55)


@pytest.mark.skipif(not HAS_POLARS, reason="polars required for file earnings store")
def test_file_data_store_earnings_methods(file_root: Path) -> None:
    earnings_root = file_root / "earnings"
    datastore_root = file_root / "datastore"
    earnings_store = FileEarningsStore(base_path=earnings_root)
    store = FileDataStore(base_path=datastore_root, earnings_store=earnings_store)

    ts_event = 1_800_000_000_000_000_000
    event_actual = store.write_earnings_actual(
        ticker="MSFT",
        period_end="2024-12-31",
        filing_date="2025-02-01",
        eps_diluted=2.75,
        revenue=120.4e9,
        ts_event=ts_event,
        ts_init=ts_event,
    )
    assert event_actual.dataset_id == "ml.earnings_actuals"

    event_estimate = store.write_earnings_estimate(
        ticker="MSFT",
        estimate_date="2024-12-15",
        period_end="2024-12-31",
        eps_consensus=2.60,
        ts_event=ts_event - 5_000,
        ts_init=ts_event - 5_000,
    )
    assert event_estimate.dataset_id == "ml.earnings_estimates"

    store.flush()

    reloaded = FileDataStore(
        base_path=datastore_root,
        earnings_store=FileEarningsStore(base_path=earnings_root),
    )
    actuals = reloaded.get_earnings_actuals_at_or_before(
        ticker="MSFT",
        ts_event=ts_event + 10,
        limit=5,
    )
    assert len(actuals) == 1
    assert actuals[0]["revenue"] == pytest.approx(120.4e9)

    estimate = reloaded.get_earnings_estimate_at_or_before(
        ticker="MSFT",
        period_end="2024-12-31",
        ts_event=ts_event,
    )
    assert estimate is not None
    assert estimate["eps_consensus"] == pytest.approx(2.60)


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
