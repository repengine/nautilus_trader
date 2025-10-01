"""Tests for store integration service metrics aggregation."""

from __future__ import annotations

import time
from dataclasses import dataclass
import statistics

import pytest
from sqlalchemy import text

from ml.dashboard.services.metrics_service import StoreIntegrationService
from ml.tests.conftest import TestDatabase


@dataclass(slots=True)
class _StubIntegrationManager:
    """Minimal integration manager providing a database connection."""

    db_connection: str

    data_store: object | None = None
    model_store: object | None = None
    strategy_store: object | None = None
    feature_store: object | None = None
    strategy_registry: object | None = None


def _ns_timestamp(offset_seconds: int = 0) -> int:
    """Return a timestamp in nanoseconds relative to now."""

    return time.time_ns() + offset_seconds * 1_000_000_000


def _seed_metrics_data(database: TestDatabase) -> None:
    """Populate the database with deterministic metrics fixtures."""

    now_ns = _ns_timestamp()
    five_minutes_ns = 300 * 1_000_000_000

    with database.engine.begin() as conn:
        # Ensure deterministic state across repeated test runs
        for table_name in (
            "ml_positions",
            "ml_data_events",
            "ml_model_predictions",
            "ml_strategy_signals",
            "ml_risk_limits",
        ):
            conn.execute(text(f"DELETE FROM {table_name}"))

        # Strategy signals for Sharpe/win rate computations
        conn.execute(
            text(
                """
                INSERT INTO ml_strategy_signals (
                    strategy_id,
                    instrument_id,
                    ts_event,
                    ts_init,
                    signal_type,
                    strength,
                    model_predictions,
                    risk_metrics,
                    execution_params,
                    is_live
                ) VALUES
                    ('strat-alpha', 'EUR/USD', :ts1, :ts1, 'BUY', 0.6, '{}'::jsonb, '{}'::jsonb, '{}'::jsonb, TRUE),
                    ('strat-alpha', 'EUR/USD', :ts2, :ts2, 'SELL', -0.3, '{}'::jsonb, '{}'::jsonb, '{}'::jsonb, TRUE),
                    ('strat-beta', 'AAPL', :ts3, :ts3, 'BUY', 0.4, '{}'::jsonb, '{}'::jsonb, '{}'::jsonb, TRUE)
                """
            ),
            {
                "ts1": now_ns - 2_000_000_000,
                "ts2": now_ns - 1_500_000_000,
                "ts3": now_ns - 1_000_000_000,
            },
        )

        # Model predictions to mark active models
        conn.execute(
            text(
                """
                INSERT INTO ml_model_predictions (
                    model_id,
                    instrument_id,
                    ts_event,
                    ts_init,
                    prediction,
                    confidence,
                    features_used,
                    inference_time_ms,
                    is_live
                ) VALUES
                    ('model-alpha', 'EUR/USD', :ts1, :ts1, 0.45, 0.9, '{}'::jsonb, 5.0, TRUE),
                    ('model-beta', 'AAPL', :ts2, :ts2, -0.12, 0.7, '{}'::jsonb, 8.0, FALSE)
                """
            ),
            {"ts1": now_ns - 2_000_000_000, "ts2": now_ns - 1_500_000_000},
        )

        # Portfolio positions contributing to PnL/portfolio snapshot
        conn.execute(
            text(
                """
                INSERT INTO ml_positions (
                    strategy_id,
                    instrument_id,
                    quantity,
                    side,
                    entry_price,
                    current_price,
                    unrealized_pnl,
                    realized_pnl,
                    position_value,
                    exposure,
                    var_95,
                    entry_time,
                    last_update
                ) VALUES
                    ('strat-alpha', 'EUR/USD', 1.0, 'LONG', 100.0, 110.0, 25.0, 15.0, 10000.0, 5000.0, 100.0, :ts1, :ts1),
                    ('strat-beta', 'AAPL', 2.0, 'LONG', 200.0, 195.0, -5.0, 20.0, 15000.0, 8000.0, 200.0, :ts2, :ts2)
                """
            ),
            {"ts1": now_ns - 2_000_000_000, "ts2": now_ns - 1_500_000_000},
        )

        # Risk limits to surface maximum drawdown
        conn.execute(
            text(
                """
                INSERT INTO ml_risk_limits (
                    strategy_id,
                    max_exposure,
                    max_position_size,
                    max_positions,
                    max_drawdown,
                    max_var,
                    max_leverage,
                    max_daily_trades,
                    max_order_size,
                    is_active
                ) VALUES ('strat-alpha', 100000.0, 50000.0, 10, 0.15, 20000.0, 5.0, 50, 10000.0, TRUE)
                ON CONFLICT (strategy_id) DO UPDATE SET max_drawdown = EXCLUDED.max_drawdown
                """
            )
        )

        # Data events for ingestion metrics (window = 5 minutes)
        conn.execute(
            text(
                """
                INSERT INTO ml_data_events (
                    dataset_id,
                    instrument_id,
                    stage,
                    source,
                    run_id,
                    ts_min,
                    ts_max,
                    ts_event,
                    count,
                    seq_min,
                    seq_max,
                    status
                ) VALUES
                    ('EQUS.MINI.BARS', 'AAPL', 'INGESTED', 'live', 'run-bars', :ts_base, :ts_base, :ts_now, 600, NULL, NULL, 'success'),
                    ('EQUS.MINI.QUOTES', 'AAPL', 'INGESTED', 'live', 'run-quotes', :ts_base, :ts_base, :ts_now, 300, NULL, NULL, 'success'),
                    ('EQUS.MINI.BOOK', 'AAPL', 'INGESTED', 'live', 'run-book', :ts_base, :ts_base, :ts_now, 150, NULL, NULL, 'failed')
                """
            ),
            {"ts_base": now_ns - five_minutes_ns, "ts_now": now_ns - 1_000_000_000},
        )


@pytest.mark.asyncio
async def test_store_metrics_snapshot_aggregates_real_data(test_database: TestDatabase) -> None:
    _seed_metrics_data(test_database)

    integration = _StubIntegrationManager(db_connection=test_database.connection_string)
    service = StoreIntegrationService(integration)

    snapshot = await service.get_metrics_snapshot()

    # Portfolio-driven metrics
    assert snapshot.daily_pnl == pytest.approx(55.0)
    assert snapshot.portfolio.total_value == pytest.approx(25000.0)
    assert snapshot.portfolio.margin_used == pytest.approx(13000.0)
    assert snapshot.portfolio.cash == pytest.approx(12000.0)
    assert snapshot.portfolio.positions == 2

    # Performance metrics derived from strategy/model activity
    assert snapshot.active_models == 2
    assert snapshot.max_drawdown == pytest.approx(0.15)
    assert snapshot.win_rate == pytest.approx(2 / 3)
    strengths = [0.6, -0.3, 0.4]
    expected_sharpe = statistics.mean(strengths) / (statistics.pstdev(strengths) or 1.0)
    assert snapshot.sharpe_ratio == pytest.approx(expected_sharpe)

    # Ingestion metrics over the five minute window
    assert snapshot.ingestion_rate.bars_per_sec == pytest.approx(600 / 300.0)
    assert snapshot.ingestion_rate.quotes_per_sec == pytest.approx(300 / 300.0)
    assert snapshot.ingestion_rate.l2_updates_per_sec == pytest.approx(150 / 300.0)
    assert snapshot.ingestion_rate.data_quality == pytest.approx(900 / 1050.0)
