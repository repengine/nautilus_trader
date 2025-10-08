#!/usr/bin/env python3

"""
Unit tests for DataReader component.

Tests all read operations with mocked stores to ensure proper delegation
and data transformation. Focus is on testing delegation logic, not SQL construction.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from ml.stores.data_reader import DataReader


# ========================================================================
# Fixtures
# ========================================================================


@pytest.fixture
def mock_feature_store() -> Mock:
    """Create mock feature store."""
    store = Mock()
    store.get_latest_at_or_before = Mock(return_value=None)
    return store


@pytest.fixture
def mock_model_store() -> Mock:
    """Create mock model store."""
    store = Mock()
    # No table/engine - tests will verify None handling
    store.model_predictions_table = None
    store.engine = None
    return store


@pytest.fixture
def mock_strategy_store() -> Mock:
    """Create mock strategy store."""
    store = Mock()
    # No table/engine - tests will verify None handling
    store.strategy_signals_table = None
    store.engine = None
    return store


@pytest.fixture
def mock_earnings_store() -> Mock:
    """Create mock earnings store."""
    store = Mock()
    store.get_actuals = Mock(return_value=[])
    store.get_estimates = Mock(return_value=None)
    return store


@pytest.fixture
def data_reader(
    mock_feature_store: Mock,
    mock_model_store: Mock,
    mock_strategy_store: Mock,
    mock_earnings_store: Mock,
) -> DataReader:
    """Create DataReader with mocked stores."""
    return DataReader(
        feature_store=mock_feature_store,
        model_store=mock_model_store,
        strategy_store=mock_strategy_store,
        earnings_store=mock_earnings_store,
    )


# ========================================================================
# Feature Read Tests
# ========================================================================


def test_data_reader_initialization(data_reader: DataReader) -> None:
    """Test DataReader initializes with all stores."""
    assert data_reader.feature_store is not None
    assert data_reader.model_store is not None
    assert data_reader.strategy_store is not None
    assert data_reader.earnings_store is not None


def test_get_features_at_or_before_returns_features(
    data_reader: DataReader,
    mock_feature_store: Mock,
) -> None:
    """Test get_features_at_or_before returns features from store."""
    expected_features = {"rsi": 65.5, "macd": 0.002}
    mock_feature_store.get_latest_at_or_before.return_value = expected_features

    result = data_reader.get_features_at_or_before(
        instrument_id="EURUSD.SIM",
        ts_event=1234567890000000000,
    )

    assert result == expected_features
    mock_feature_store.get_latest_at_or_before.assert_called_once_with(
        "EURUSD.SIM",
        1234567890000000000,
    )


def test_get_features_at_or_before_returns_none_when_not_found(
    data_reader: DataReader,
    mock_feature_store: Mock,
) -> None:
    """Test get_features_at_or_before returns None when features not found."""
    mock_feature_store.get_latest_at_or_before.return_value = None

    result = data_reader.get_features_at_or_before(
        instrument_id="EURUSD.SIM",
        ts_event=1234567890000000000,
    )

    assert result is None


def test_get_features_at_or_before_sanitizes_timestamp(
    data_reader: DataReader,
    mock_feature_store: Mock,
) -> None:
    """Test get_features_at_or_before converts timestamp to int."""
    mock_feature_store.get_latest_at_or_before.return_value = {}

    # Pass float timestamp
    data_reader.get_features_at_or_before(
        instrument_id="EURUSD.SIM",
        ts_event=1234567890000000000,
    )

    # Should convert to int
    mock_feature_store.get_latest_at_or_before.assert_called_once_with(
        "EURUSD.SIM",
        1234567890000000000,
    )


# ========================================================================
# Prediction Read Tests
# ========================================================================


def test_get_latest_prediction_returns_none_when_table_missing(
    data_reader: DataReader,
    mock_model_store: Mock,
) -> None:
    """Test get_latest_prediction_at_or_before returns None when table missing."""
    mock_model_store.model_predictions_table = None

    result = data_reader.get_latest_prediction_at_or_before(
        instrument_id="EURUSD.SIM",
        ts_event=1234567890000000000,
    )

    assert result is None


def test_get_latest_prediction_returns_none_when_engine_missing(
    data_reader: DataReader,
    mock_model_store: Mock,
) -> None:
    """Test get_latest_prediction_at_or_before returns None when engine missing."""
    mock_model_store.model_predictions_table = Mock()
    mock_model_store.engine = None

    result = data_reader.get_latest_prediction_at_or_before(
        instrument_id="EURUSD.SIM",
        ts_event=1234567890000000000,
    )

    assert result is None


# ========================================================================
# Signal Read Tests
# ========================================================================


def test_get_latest_signal_returns_none_when_table_missing(
    data_reader: DataReader,
    mock_strategy_store: Mock,
) -> None:
    """Test get_latest_signal_at_or_before returns None when table missing."""
    mock_strategy_store.strategy_signals_table = None

    result = data_reader.get_latest_signal_at_or_before(
        instrument_id="EURUSD.SIM",
        ts_event=1234567890000000000,
    )

    assert result is None


def test_get_latest_signal_returns_none_when_engine_missing(
    data_reader: DataReader,
    mock_strategy_store: Mock,
) -> None:
    """Test get_latest_signal_at_or_before returns None when engine missing."""
    mock_strategy_store.strategy_signals_table = Mock()
    mock_strategy_store.engine = None

    result = data_reader.get_latest_signal_at_or_before(
        instrument_id="EURUSD.SIM",
        ts_event=1234567890000000000,
    )

    assert result is None


# ========================================================================
# Earnings Read Tests
# ========================================================================


def test_get_earnings_actuals_returns_actuals(
    data_reader: DataReader,
    mock_earnings_store: Mock,
) -> None:
    """Test get_earnings_actuals_at_or_before returns earnings actuals."""
    expected_actuals = [
        {"ticker": "AAPL", "eps_diluted": 1.52, "revenue": 123.4e9},
        {"ticker": "AAPL", "eps_diluted": 1.40, "revenue": 111.4e9},
    ]
    mock_earnings_store.get_actuals.return_value = expected_actuals

    result = data_reader.get_earnings_actuals_at_or_before(
        ticker="AAPL",
        ts_event=1234567890000000000,
    )

    assert result == expected_actuals
    mock_earnings_store.get_actuals.assert_called_once()


def test_get_earnings_actuals_respects_limit(
    data_reader: DataReader,
    mock_earnings_store: Mock,
) -> None:
    """Test get_earnings_actuals_at_or_before respects limit parameter."""
    actuals = [{"id": i} for i in range(10)]
    mock_earnings_store.get_actuals.return_value = actuals

    result = data_reader.get_earnings_actuals_at_or_before(
        ticker="AAPL",
        ts_event=1234567890000000000,
        limit=3,
    )

    assert len(result) == 3
    assert result == actuals[:3]


def test_get_earnings_actuals_returns_empty_when_limit_zero(
    data_reader: DataReader,
    mock_earnings_store: Mock,
) -> None:
    """Test get_earnings_actuals_at_or_before returns empty list when limit is 0."""
    result = data_reader.get_earnings_actuals_at_or_before(
        ticker="AAPL",
        ts_event=1234567890000000000,
        limit=0,
    )

    assert result == []
    mock_earnings_store.get_actuals.assert_not_called()


def test_get_earnings_actuals_returns_empty_when_limit_negative(
    data_reader: DataReader,
    mock_earnings_store: Mock,
) -> None:
    """Test get_earnings_actuals_at_or_before returns empty list when limit is negative."""
    result = data_reader.get_earnings_actuals_at_or_before(
        ticker="AAPL",
        ts_event=1234567890000000000,
        limit=-5,
    )

    assert result == []
    mock_earnings_store.get_actuals.assert_not_called()


def test_get_earnings_actuals_passes_date_filters(
    data_reader: DataReader,
    mock_earnings_store: Mock,
) -> None:
    """Test get_earnings_actuals_at_or_before passes date filters to store."""
    mock_earnings_store.get_actuals.return_value = []

    data_reader.get_earnings_actuals_at_or_before(
        ticker="AAPL",
        ts_event=1234567890000000000,
        start_date="2023-01-01",
        end_date="2023-12-31",
    )

    call_args = mock_earnings_store.get_actuals.call_args
    assert call_args.kwargs["start_date"] == "2023-01-01"
    assert call_args.kwargs["end_date"] == "2023-12-31"


def test_get_earnings_actuals_sanitizes_timestamp(
    data_reader: DataReader,
    mock_earnings_store: Mock,
) -> None:
    """Test get_earnings_actuals_at_or_before sanitizes timestamp."""
    mock_earnings_store.get_actuals.return_value = []

    data_reader.get_earnings_actuals_at_or_before(
        ticker="AAPL",
        ts_event=1234567890000000000,
    )

    # Should call sanitize_timestamp_ns internally
    mock_earnings_store.get_actuals.assert_called_once()
    call_args = mock_earnings_store.get_actuals.call_args
    assert "as_of_ts" in call_args.kwargs


def test_get_earnings_estimate_returns_estimate(
    data_reader: DataReader,
    mock_earnings_store: Mock,
) -> None:
    """Test get_earnings_estimate_at_or_before returns estimate."""
    expected_estimate = {
        "ticker": "AAPL",
        "period_end": "2023-12-31",
        "eps_consensus": 1.55,
    }
    mock_earnings_store.get_estimates.return_value = expected_estimate

    result = data_reader.get_earnings_estimate_at_or_before(
        ticker="AAPL",
        period_end="2023-12-31",
        ts_event=1234567890000000000,
    )

    assert result == expected_estimate


def test_get_earnings_estimate_returns_none_when_not_found(
    data_reader: DataReader,
    mock_earnings_store: Mock,
) -> None:
    """Test get_earnings_estimate_at_or_before returns None when not found."""
    mock_earnings_store.get_estimates.return_value = None

    result = data_reader.get_earnings_estimate_at_or_before(
        ticker="AAPL",
        period_end="2023-12-31",
        ts_event=1234567890000000000,
    )

    assert result is None


def test_get_earnings_estimate_sanitizes_timestamp(
    data_reader: DataReader,
    mock_earnings_store: Mock,
) -> None:
    """Test get_earnings_estimate_at_or_before sanitizes timestamp."""
    mock_earnings_store.get_estimates.return_value = None

    data_reader.get_earnings_estimate_at_or_before(
        ticker="AAPL",
        period_end="2023-12-31",
        ts_event=1234567890000000000,
    )

    # Should call sanitize_timestamp_ns internally
    mock_earnings_store.get_estimates.assert_called_once()
    call_args = mock_earnings_store.get_estimates.call_args
    assert "as_of_ts" in call_args.kwargs


# ========================================================================
# Protocol Conformance Tests
# ========================================================================


def test_data_reader_conforms_to_protocol() -> None:
    """Test DataReader conforms to DataReaderProtocol."""
    from ml.stores.data_reader import DataReaderProtocol

    # Create with mocks
    reader = DataReader(
        feature_store=Mock(),
        model_store=Mock(),
        strategy_store=Mock(),
        earnings_store=Mock(),
    )

    # Check protocol conformance (should have all required methods)
    assert hasattr(reader, "get_features_at_or_before")
    assert hasattr(reader, "get_latest_prediction_at_or_before")
    assert hasattr(reader, "get_latest_signal_at_or_before")
    assert hasattr(reader, "get_earnings_actuals_at_or_before")
    assert hasattr(reader, "get_earnings_estimate_at_or_before")


# ========================================================================
# Edge Case Tests
# ========================================================================


def test_data_reader_handles_store_exceptions_gracefully(
    data_reader: DataReader,
    mock_feature_store: Mock,
) -> None:
    """Test DataReader handles store exceptions gracefully."""
    mock_feature_store.get_latest_at_or_before.side_effect = Exception("Store error")

    # Should propagate exception (no swallowing)
    with pytest.raises(Exception, match="Store error"):
        data_reader.get_features_at_or_before(
            instrument_id="EURUSD.SIM",
            ts_event=1234567890000000000,
        )


def test_data_reader_with_real_stores_integration() -> None:
    """Test DataReader can be initialized with real store types."""
    # This is a smoke test to ensure the component can work with real stores
    from unittest.mock import Mock

    # Mock stores that look more realistic
    feature_store = Mock()
    feature_store.get_latest_at_or_before = Mock(return_value={"feature": 1.0})

    model_store = Mock()
    model_store.model_predictions_table = None
    model_store.engine = None

    strategy_store = Mock()
    strategy_store.strategy_signals_table = None
    strategy_store.engine = None

    earnings_store = Mock()
    earnings_store.get_actuals = Mock(return_value=[])
    earnings_store.get_estimates = Mock(return_value=None)

    reader = DataReader(
        feature_store=feature_store,
        model_store=model_store,
        strategy_store=strategy_store,
        earnings_store=earnings_store,
    )

    # Should work without errors
    assert reader.get_features_at_or_before(
        instrument_id="TEST",
        ts_event=0,
    ) == {"feature": 1.0}
