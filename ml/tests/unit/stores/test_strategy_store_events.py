"""
Test StrategyStore event emission functionality.

This test verifies that SIGNAL_EMITTED events are properly emitted after flush operations.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from ml.stores.strategy_store import StrategyStore


# Apply module-scoped cleanup once for all tests in this module to reduce overhead
pytestmark = pytest.mark.usefixtures("clean_postgres_db_module")


@pytest.mark.database
@pytest.mark.serial
def test_strategy_store_emits_signal_events(test_database):
    """Test that StrategyStore emits SIGNAL_EMITTED events after flush."""
    # Create store with PostgreSQL connection
    store = StrategyStore(
        connection_string=test_database.connection_string,
        batch_size=10
    )

    # Mock the DataRegistry
    mock_registry = MagicMock()
    mock_registry.emit_event = MagicMock()
    mock_registry.update_watermark = MagicMock()

    # Patch the _get_data_registry method to return our mock
    with patch.object(store, "_get_data_registry", return_value=mock_registry):
        # Add some signals
        for i in range(3):
            store.write_signal(
                strategy_id="test_strategy",
                instrument_id="EUR/USD",
                signal_type="BUY" if i % 2 == 0 else "SELL",
                strength=0.5 + i * 0.1,
                model_predictions={"model1": 0.7 + i * 0.05},
                risk_metrics={"risk_score": 0.2 + i * 0.1},
                execution_params={"stop_loss": 1.05 + i * 0.01},
                ts_event=1700000000000000000 + i * 1000000000,
                is_live=False
            )

        # Flush should trigger event emission
        store.flush()

        # Verify emit_event was called
        assert mock_registry.emit_event.called

        # Check the call arguments
        call_args = mock_registry.emit_event.call_args[1]
        assert call_args["dataset_id"] == "signals"
        assert call_args["instrument_id"] == "EUR/USD"
        assert call_args["stage"] == "SIGNAL_EMITTED"
        assert call_args["source"] == "realtime"
        assert call_args["count"] == 3
        assert call_args["status"] == "success"
        assert call_args["ts_min"] == 1700000000000000000
        assert call_args["ts_max"] == 1700000002000000000

        # Verify update_watermark was called
        assert mock_registry.update_watermark.called
        watermark_args = mock_registry.update_watermark.call_args[1]
        assert watermark_args["dataset_id"] == "signals"
        assert watermark_args["instrument_id"] == "EUR/USD"
        assert watermark_args["source"] == "realtime"
        assert watermark_args["last_success_ns"] == 1700000002000000000
        assert watermark_args["count"] == 3
        assert watermark_args["completeness_pct"] == 100.0


@pytest.mark.database
@pytest.mark.serial
def test_strategy_store_groups_signals_by_strategy_and_instrument(test_database):
    """Test that signals are grouped by strategy_id and instrument_id for event emission."""
    store = StrategyStore(
        connection_string=test_database.connection_string,
        batch_size=100
    )

    mock_registry = MagicMock()
    mock_registry.emit_event = MagicMock()
    mock_registry.update_watermark = MagicMock()

    with patch.object(store, "_get_data_registry", return_value=mock_registry):
        # Add signals for different strategies and instruments
        store.write_signal(
            strategy_id="strategy1",
            instrument_id="EUR/USD",
            signal_type="BUY",
            strength=0.8,
            model_predictions={"model1": 0.75},
            risk_metrics={"risk_score": 0.3},
            execution_params={"stop_loss": 1.05},
            ts_event=1700000000000000000,
            is_live=True
        )

        store.write_signal(
            strategy_id="strategy1",
            instrument_id="GBP/USD",
            signal_type="SELL",
            strength=0.6,
            model_predictions={"model1": 0.65},
            risk_metrics={"risk_score": 0.4},
            execution_params={"stop_loss": 1.35},
            ts_event=1700000001000000000,
            is_live=True
        )

        store.write_signal(
            strategy_id="strategy2",
            instrument_id="EUR/USD",
            signal_type="HOLD",
            strength=0.5,
            model_predictions={"model2": 0.55},
            risk_metrics={"risk_score": 0.2},
            execution_params={},
            ts_event=1700000002000000000,
            is_live=False
        )

        # Flush should emit 3 separate events (one for each strategy/instrument combo)
        store.flush()

        # Should have 3 emit_event calls
        assert mock_registry.emit_event.call_count == 3
        assert mock_registry.update_watermark.call_count == 3

        # Count occurrences per instrument_id (two for EUR/USD, one for GBP/USD)
        from collections import Counter as _Counter
        instrument_counts = _Counter(call[1]["instrument_id"] for call in mock_registry.emit_event.call_args_list)
        assert instrument_counts["EUR/USD"] == 2
        assert instrument_counts["GBP/USD"] == 1
        # All events should use canonical dataset id
        assert all(call[1]["dataset_id"] == "signals" for call in mock_registry.emit_event.call_args_list)


@pytest.mark.database
@pytest.mark.serial
def test_strategy_store_handles_event_emission_failure_gracefully(test_database):
    """Test that event emission failures don't break signal storage."""
    store = StrategyStore(
        connection_string=test_database.connection_string,
        batch_size=10
    )

    # Mock registry that raises an exception
    mock_registry = MagicMock()
    mock_registry.emit_event.side_effect = Exception("Test failure")

    with patch.object(store, "_get_data_registry", return_value=mock_registry):
        # This should not raise even though event emission fails
        store.write_signal(
            strategy_id="test_strategy",
            instrument_id="EUR/USD",
            signal_type="BUY",
            strength=0.8,
            model_predictions={"model1": 0.75},
            risk_metrics={"risk_score": 0.3},
            execution_params={"stop_loss": 1.05},
            ts_event=1700000000000000000,
            is_live=False
        )

        # Flush should complete successfully despite event emission failure
        store.flush()

        # Buffer should be cleared
        assert len(store._write_buffer) == 0


@pytest.mark.database
@pytest.mark.serial
def test_strategy_store_no_events_when_registry_unavailable(test_database):
    """Test that store works normally when DataRegistry is unavailable."""
    store = StrategyStore(
        connection_string=test_database.connection_string,
        batch_size=10
    )

    # Mock _get_data_registry to return None (simulating unavailable registry)
    with patch.object(store, "_get_data_registry", return_value=None):
        # Add and flush signals
        store.write_signal(
            strategy_id="test_strategy",
            instrument_id="EUR/USD",
            signal_type="BUY",
            strength=0.8,
            model_predictions={"model1": 0.75},
            risk_metrics={"risk_score": 0.3},
            execution_params={"stop_loss": 1.05},
            ts_event=1700000000000000000,
            is_live=False
        )

        # Should complete without issues
        store.flush()

        # Buffer should be cleared
        assert len(store._write_buffer) == 0


# Remove main block - use pytest to run tests
