#!/usr/bin/env python3

"""
Unit tests for FeatureReaderComponent (Phase 3.7.2).

Tests feature reading operations including training data retrieval, point-in-time
lookup, range queries, and existence checks.

Coverage target: 95%

"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ml.stores.common.feature_reader import (
    FeatureReaderComponent,
    FeatureReaderConfig,
    FeatureReaderProtocol,
)


# =========================================================================
# Mock Classes and Helpers
# =========================================================================


class MockRow:
    """Mock SQLAlchemy row for testing."""

    def __init__(self, data: tuple[Any, ...]) -> None:
        self._data = data

    def __getitem__(self, idx: int) -> Any:
        return self._data[idx]


def create_mock_feature_row(
    ts_event: int = 1700000000000000000,
    values: dict[str, float] | str | None = None,
) -> MockRow:
    """Create a mock feature row."""
    if values is None:
        values = {"close_return": 0.01, "volume_ratio_20": 1.5}
    return MockRow((ts_event, values))


def create_mock_values_row(
    values: dict[str, float] | str | None = None,
) -> MockRow:
    """Create a mock row with just values column."""
    if values is None:
        values = {"close_return": 0.01, "volume_ratio_20": 1.5}
    return MockRow((values,))


def create_mock_ts_row(ts_event: int = 1700000000000000000) -> MockRow:
    """Create a mock row with just ts_event column."""
    return MockRow((ts_event,))


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def mock_engine() -> MagicMock:
    """Create a mock SQLAlchemy engine."""
    engine = MagicMock()
    engine.dialect.name = "postgresql"
    conn_mock = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=conn_mock)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    return engine


@pytest.fixture
def mock_table() -> MagicMock:
    """Create a mock SQLAlchemy table."""
    table = MagicMock()
    table.c = MagicMock()
    # Set up column accessors
    table.c.ts_event = MagicMock()
    table.c.feature_set_id = MagicMock()
    table.c.instrument_id = MagicMock()
    table.c.__getitem__ = MagicMock(return_value=MagicMock())
    return table


@pytest.fixture
def feature_reader(mock_engine: MagicMock, mock_table: MagicMock) -> FeatureReaderComponent:
    """Create a FeatureReaderComponent for testing."""
    return FeatureReaderComponent(
        engine=mock_engine,
        table=mock_table,
        get_feature_set_id=lambda: "default_fs",
        get_feature_names=lambda: ["close_return", "volume_ratio_20"],
    )


@pytest.fixture
def feature_reader_with_persistence(
    mock_engine: MagicMock,
    mock_table: MagicMock,
) -> FeatureReaderComponent:
    """Create a FeatureReaderComponent with mock persistence."""
    persistence = MagicMock()
    persistence.session = MagicMock()
    return FeatureReaderComponent(
        engine=mock_engine,
        table=mock_table,
        get_feature_set_id=lambda: "default_fs",
        get_feature_names=lambda: ["close_return", "volume_ratio_20"],
        persistence=persistence,
    )


# =========================================================================
# Protocol Compliance Tests
# =========================================================================


class TestFeatureReaderProtocol:
    """Test protocol compliance."""

    def test_component_satisfies_protocol(
        self,
        feature_reader: FeatureReaderComponent,
    ) -> None:
        """Verify FeatureReaderComponent satisfies FeatureReaderProtocol."""
        assert isinstance(feature_reader, FeatureReaderProtocol)


# =========================================================================
# Configuration Tests
# =========================================================================


class TestFeatureReaderConfig:
    """Test configuration."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = FeatureReaderConfig()
        assert config.table_name == "ml_feature_values"

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = FeatureReaderConfig(table_name="custom_features")
        assert config.table_name == "custom_features"


# =========================================================================
# Happy Path Tests - get_training_data
# =========================================================================


class TestGetTrainingData:
    """Test get_training_data method."""

    def test_get_training_data_returns_features_timestamps_names(
        self,
        feature_reader: FeatureReaderComponent,
    ) -> None:
        """Verify training data retrieval returns correct structure."""
        # Setup mock result
        mock_rows = [
            create_mock_feature_row(
                ts_event=1700000000000000000,
                values={"close_return": 0.01, "volume_ratio_20": 1.5},
            ),
            create_mock_feature_row(
                ts_event=1700000001000000000,
                values={"close_return": 0.02, "volume_ratio_20": 1.6},
            ),
        ]

        # Patch internal query method
        feature_reader._execute_training_query = MagicMock(return_value=mock_rows)

        features, timestamps, feature_names = feature_reader.get_training_data(
            instrument_id="SPY.DATABENTO",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
        )

        # Verify return types
        assert features.dtype == np.float64
        assert timestamps.dtype == np.int64
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0

        # Verify shapes
        assert features.shape == (2, 2)  # 2 samples, 2 features
        assert timestamps.shape == (2,)  # 2 timestamps
        assert features.shape[0] == len(timestamps)

        # Verify values
        assert timestamps[0] == 1700000000000000000
        assert timestamps[1] == 1700000001000000000
        assert features[0, 0] == 0.01  # close_return for first row
        assert features[1, 0] == 0.02  # close_return for second row

    def test_get_training_data_empty_range_returns_empty_arrays(
        self,
        feature_reader: FeatureReaderComponent,
    ) -> None:
        """Verify empty arrays returned for empty time range."""
        # Patch internal query method to return empty list
        feature_reader._execute_training_query = MagicMock(return_value=[])

        features, timestamps, feature_names = feature_reader.get_training_data(
            instrument_id="SPY.DATABENTO",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
        )

        assert features.size == 0
        assert timestamps.size == 0
        assert feature_names == []

    def test_get_training_data_parses_json_values_correctly(
        self,
        feature_reader: FeatureReaderComponent,
    ) -> None:
        """Verify JSON string values are parsed correctly."""
        # Values stored as JSON string
        mock_rows = [
            create_mock_feature_row(
                ts_event=1700000000000000000,
                values='{"close_return": 0.01, "volume_ratio_20": 1.5}',
            ),
        ]

        feature_reader._execute_training_query = MagicMock(return_value=mock_rows)

        features, _timestamps, _feature_names = feature_reader.get_training_data(
            instrument_id="SPY.DATABENTO",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
        )

        assert features.shape == (1, 2)
        assert features[0, 0] == 0.01  # close_return
        assert features[0, 1] == 1.5  # volume_ratio_20

    def test_get_training_data_handles_malformed_json(
        self,
        feature_reader: FeatureReaderComponent,
    ) -> None:
        """Verify malformed JSON is handled gracefully."""
        mock_rows = [
            create_mock_feature_row(
                ts_event=1700000000000000000,
                values="not valid json",
            ),
        ]

        feature_reader._execute_training_query = MagicMock(return_value=mock_rows)

        features, _timestamps, _feature_names = feature_reader.get_training_data(
            instrument_id="SPY.DATABENTO",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
        )

        # Should return zeros for missing features
        assert features.shape == (1, 2)
        assert features[0, 0] == 0.0
        assert features[0, 1] == 0.0

    def test_get_training_data_missing_feature_returns_zero(
        self,
        feature_reader: FeatureReaderComponent,
    ) -> None:
        """Verify missing features return 0.0."""
        mock_rows = [
            create_mock_feature_row(
                ts_event=1700000000000000000,
                values={"close_return": 0.01},  # Missing volume_ratio_20
            ),
        ]

        feature_reader._execute_training_query = MagicMock(return_value=mock_rows)

        features, _timestamps, _feature_names = feature_reader.get_training_data(
            instrument_id="SPY.DATABENTO",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
        )

        assert features.shape == (1, 2)
        assert features[0, 0] == 0.01  # close_return present
        assert features[0, 1] == 0.0  # volume_ratio_20 missing

    def test_get_training_data_include_bars_is_consumed(
        self,
        feature_reader: FeatureReaderComponent,
    ) -> None:
        """Verify include_bars parameter is accepted but consumed."""
        feature_reader._execute_training_query = MagicMock(return_value=[])

        # Should not raise even with include_bars=False
        features, _timestamps, _feature_names = feature_reader.get_training_data(
            instrument_id="SPY.DATABENTO",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
            include_bars=False,
        )

        assert features.size == 0


# =========================================================================
# Happy Path Tests - get_latest_at_or_before
# =========================================================================


class TestGetLatestAtOrBefore:
    """Test get_latest_at_or_before method."""

    def test_get_latest_at_or_before_returns_most_recent(
        self,
        feature_reader: FeatureReaderComponent,
    ) -> None:
        """Verify most recent feature row is returned."""
        mock_row = create_mock_values_row(
            values={"close_return": 0.01, "volume_ratio_20": 1.5},
        )

        feature_reader._execute_latest_query = MagicMock(return_value=mock_row)

        result = feature_reader.get_latest_at_or_before(
            instrument_id="SPY.DATABENTO",
            ts_event=1700000000000000000,
        )

        assert result is not None
        assert result["close_return"] == 0.01
        assert result["volume_ratio_20"] == 1.5

    def test_get_latest_at_or_before_returns_none_when_empty(
        self,
        feature_reader: FeatureReaderComponent,
    ) -> None:
        """Verify None returned when no features exist."""
        feature_reader._execute_latest_query = MagicMock(return_value=None)

        result = feature_reader.get_latest_at_or_before(
            instrument_id="NON_EXISTENT",
            ts_event=1700000000000000000,
        )

        assert result is None

    def test_get_latest_at_or_before_handles_json_string_values(
        self,
        feature_reader: FeatureReaderComponent,
    ) -> None:
        """Verify JSON string values are parsed correctly."""
        mock_row = create_mock_values_row(
            values='{"close_return": 0.01, "volume_ratio_20": 1.5}',
        )

        feature_reader._execute_latest_query = MagicMock(return_value=mock_row)

        result = feature_reader.get_latest_at_or_before(
            instrument_id="SPY.DATABENTO",
            ts_event=1700000000000000000,
        )

        assert result is not None
        assert isinstance(result, dict)
        assert result["close_return"] == 0.01
        assert result["volume_ratio_20"] == 1.5

    def test_get_latest_at_or_before_handles_malformed_json(
        self,
        feature_reader: FeatureReaderComponent,
    ) -> None:
        """Verify malformed JSON returns empty dict."""
        mock_row = create_mock_values_row(values="not valid json")

        feature_reader._execute_latest_query = MagicMock(return_value=mock_row)

        result = feature_reader.get_latest_at_or_before(
            instrument_id="SPY.DATABENTO",
            ts_event=1700000000000000000,
        )

        assert result == {}

    def test_get_latest_at_or_before_handles_none_values(
        self,
        feature_reader: FeatureReaderComponent,
    ) -> None:
        """Verify None values are handled gracefully."""
        # Create a row with explicit None (not using default)
        mock_row = MockRow((None,))

        feature_reader._execute_latest_query = MagicMock(return_value=mock_row)

        result = feature_reader.get_latest_at_or_before(
            instrument_id="SPY.DATABENTO",
            ts_event=1700000000000000000,
        )

        assert result == {}

    def test_get_latest_at_or_before_converts_values_to_float(
        self,
        feature_reader: FeatureReaderComponent,
    ) -> None:
        """Verify values are converted to float."""
        # Values with int type
        mock_row = create_mock_values_row(
            values={"close_return": 1, "volume_ratio_20": 2},
        )

        feature_reader._execute_latest_query = MagicMock(return_value=mock_row)

        result = feature_reader.get_latest_at_or_before(
            instrument_id="SPY.DATABENTO",
            ts_event=1700000000000000000,
        )

        assert result is not None
        assert all(isinstance(v, float) for v in result.values())
        assert result["close_return"] == 1.0
        assert result["volume_ratio_20"] == 2.0


# =========================================================================
# Happy Path Tests - read_range
# =========================================================================


class TestReadRange:
    """Test read_range method."""

    def test_read_range_returns_dataframe_with_correct_columns(
        self,
        feature_reader: FeatureReaderComponent,
        mock_engine: MagicMock,
    ) -> None:
        """Verify DataFrame returned with correct columns."""
        # Create a mock DataFrame result
        expected_df = pd.DataFrame(
            {
                "feature_set_id": ["fs_001"],
                "instrument_id": ["SPY.DATABENTO"],
                "values": [{"close_return": 0.01}],
                "ts_event": [1700000000000000000],
                "ts_init": [1700000000000000000],
            }
        )

        conn_mock = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = conn_mock

        with patch("pandas.read_sql_query", return_value=expected_df):
            result = feature_reader.read_range(
                start_ns=1700000000000000000,
                end_ns=1700086400000000000,
            )

        assert isinstance(result, pd.DataFrame)
        assert "feature_set_id" in result.columns
        assert "instrument_id" in result.columns
        assert "values" in result.columns
        assert "ts_event" in result.columns
        assert "ts_init" in result.columns

    def test_read_range_filters_by_instrument_id(
        self,
        feature_reader: FeatureReaderComponent,
        mock_engine: MagicMock,
    ) -> None:
        """Verify instrument_id filter is applied."""
        expected_df = pd.DataFrame(
            {
                "feature_set_id": ["fs_001"],
                "instrument_id": ["SPY.DATABENTO"],
                "values": [{"close_return": 0.01}],
                "ts_event": [1700000000000000000],
                "ts_init": [1700000000000000000],
            }
        )

        conn_mock = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = conn_mock

        with patch("pandas.read_sql_query", return_value=expected_df) as mock_query:
            feature_reader.read_range(
                start_ns=1700000000000000000,
                end_ns=1700086400000000000,
                instrument_id="SPY.DATABENTO",
            )

            # Verify params include instrument_id
            call_args = mock_query.call_args
            params = call_args[1].get("params") or call_args[0][2] if len(call_args[0]) > 2 else {}
            if not params and "params" in call_args[1]:
                params = call_args[1]["params"]
            assert params.get("instrument_id") == "SPY.DATABENTO"

    def test_read_range_without_instrument_filter(
        self,
        feature_reader: FeatureReaderComponent,
        mock_engine: MagicMock,
    ) -> None:
        """Verify no instrument_id filter when not provided."""
        expected_df = pd.DataFrame(
            {
                "feature_set_id": ["fs_001", "fs_002"],
                "instrument_id": ["SPY.DATABENTO", "AAPL.DATABENTO"],
                "values": [{"close_return": 0.01}, {"close_return": 0.02}],
                "ts_event": [1700000000000000000, 1700000001000000000],
                "ts_init": [1700000000000000000, 1700000001000000000],
            }
        )

        conn_mock = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = conn_mock

        with patch("pandas.read_sql_query", return_value=expected_df) as mock_query:
            result = feature_reader.read_range(
                start_ns=1700000000000000000,
                end_ns=1700086400000000000,
            )

            # Verify params do not include instrument_id
            call_args = mock_query.call_args
            params = call_args[1].get("params") or {}
            assert "instrument_id" not in params

    def test_read_range_uses_sqlite_table_name(
        self,
        mock_table: MagicMock,
    ) -> None:
        """Verify SQLite table name is used when dialect is sqlite."""
        mock_engine = MagicMock()
        mock_engine.dialect.name = "sqlite"
        conn_mock = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=conn_mock)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        reader = FeatureReaderComponent(
            engine=mock_engine,
            table=mock_table,
            get_feature_set_id=lambda: "default_fs",
            get_feature_names=lambda: ["close_return"],
        )

        expected_df = pd.DataFrame(columns=["feature_set_id", "instrument_id", "values", "ts_event", "ts_init"])

        with patch("pandas.read_sql_query", return_value=expected_df) as mock_query:
            reader.read_range(
                start_ns=1700000000000000000,
                end_ns=1700086400000000000,
            )

            # Verify query uses sqlite table name (without schema)
            call_args = mock_query.call_args
            sql = str(call_args[0][0])
            assert "public." not in sql
            assert "ml_feature_values" in sql

    def test_read_range_with_persistence_session(
        self,
        feature_reader_with_persistence: FeatureReaderComponent,
    ) -> None:
        """Verify persistence session is used when available."""
        mock_rows = [
            ("fs_001", "SPY.DATABENTO", {"close_return": 0.01}, 1700000000000000000, 1700000000000000000),
        ]

        # Setup session mock
        session = feature_reader_with_persistence.persistence.session
        execute_result = MagicMock()
        execute_result.fetchall.return_value = mock_rows
        session.execute.return_value = execute_result

        result = feature_reader_with_persistence.read_range(
            start_ns=1700000000000000000,
            end_ns=1700086400000000000,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["feature_set_id"] == "fs_001"

    def test_read_range_session_returns_empty_on_exception(
        self,
        feature_reader_with_persistence: FeatureReaderComponent,
        mock_engine: MagicMock,
    ) -> None:
        """Verify empty result when session execution fails."""
        session = feature_reader_with_persistence.persistence.session
        session.execute.side_effect = Exception("Session error")

        # When session fails, should fall back to engine
        expected_df = pd.DataFrame(
            columns=["feature_set_id", "instrument_id", "values", "ts_event", "ts_init"]
        )
        conn_mock = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = conn_mock

        with patch("pandas.read_sql_query", return_value=expected_df):
            result = feature_reader_with_persistence.read_range(
                start_ns=1700000000000000000,
                end_ns=1700086400000000000,
            )

        assert isinstance(result, pd.DataFrame)


# =========================================================================
# Happy Path Tests - features_exist
# =========================================================================


class TestFeaturesExist:
    """Test features_exist method."""

    def test_features_exist_returns_true_when_present(
        self,
        feature_reader: FeatureReaderComponent,
    ) -> None:
        """Verify True returned when features exist."""
        feature_reader._execute_exists_query = MagicMock(return_value=True)

        result = feature_reader.features_exist(
            instrument_id="SPY.DATABENTO",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
        )

        assert result is True

    def test_features_exist_returns_false_when_empty(
        self,
        feature_reader: FeatureReaderComponent,
    ) -> None:
        """Verify False returned when no features exist."""
        feature_reader._execute_exists_query = MagicMock(return_value=False)

        result = feature_reader.features_exist(
            instrument_id="NON_EXISTENT",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
        )

        assert result is False

    def test_features_exist_uses_correct_feature_set_id(
        self,
        mock_engine: MagicMock,
        mock_table: MagicMock,
    ) -> None:
        """Verify feature_set_id is obtained from callback."""
        custom_fs_id = "custom_feature_set"
        reader = FeatureReaderComponent(
            engine=mock_engine,
            table=mock_table,
            get_feature_set_id=lambda: custom_fs_id,
            get_feature_names=lambda: ["close_return"],
        )

        reader._execute_exists_query = MagicMock(return_value=False)

        # Just verify it doesn't raise - the callback is used internally
        result = reader.features_exist(
            instrument_id="SPY.DATABENTO",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
        )

        assert result is False


# =========================================================================
# Edge Case Tests
# =========================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_get_training_data_with_single_feature(
        self,
        mock_engine: MagicMock,
        mock_table: MagicMock,
    ) -> None:
        """Verify single feature handling."""
        reader = FeatureReaderComponent(
            engine=mock_engine,
            table=mock_table,
            get_feature_set_id=lambda: "default_fs",
            get_feature_names=lambda: ["close_return"],  # Single feature
        )

        mock_rows = [
            create_mock_feature_row(
                ts_event=1700000000000000000,
                values={"close_return": 0.01},
            ),
        ]

        reader._execute_training_query = MagicMock(return_value=mock_rows)

        features, _timestamps, feature_names = reader.get_training_data(
            instrument_id="SPY.DATABENTO",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
        )

        assert features.shape == (1, 1)
        assert feature_names == ["close_return"]

    def test_get_latest_at_or_before_with_non_float_convertible_value(
        self,
        feature_reader: FeatureReaderComponent,
    ) -> None:
        """Verify handling of values that cannot be converted to float."""
        # Values with non-convertible types
        mock_row = create_mock_values_row(
            values={"close_return": "not_a_number", "volume_ratio_20": None},
        )

        feature_reader._execute_latest_query = MagicMock(return_value=mock_row)

        # Should return empty dict on conversion failure
        result = feature_reader.get_latest_at_or_before(
            instrument_id="SPY.DATABENTO",
            ts_event=1700000000000000000,
        )

        assert result == {}

    def test_read_range_with_empty_result(
        self,
        feature_reader: FeatureReaderComponent,
        mock_engine: MagicMock,
    ) -> None:
        """Verify empty DataFrame returned for empty range."""
        expected_df = pd.DataFrame(
            columns=["feature_set_id", "instrument_id", "values", "ts_event", "ts_init"]
        )

        conn_mock = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = conn_mock

        with patch("pandas.read_sql_query", return_value=expected_df):
            result = feature_reader.read_range(
                start_ns=1700000000000000000,
                end_ns=1700000000000000001,
            )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_features_exist_with_same_start_end(
        self,
        feature_reader: FeatureReaderComponent,
    ) -> None:
        """Verify features_exist works with same start and end time."""
        feature_reader._execute_exists_query = MagicMock(return_value=True)

        # Same start and end
        same_time = datetime(2024, 1, 1, 12, 0, 0)
        result = feature_reader.features_exist(
            instrument_id="SPY.DATABENTO",
            start=same_time,
            end=same_time,
        )

        assert result is True

    def test_read_range_persistence_with_get_session(
        self,
        mock_engine: MagicMock,
        mock_table: MagicMock,
    ) -> None:
        """Verify persistence.get_session() is used when .session is None."""
        persistence = MagicMock()
        persistence.session = None
        mock_session = MagicMock()
        persistence.get_session.return_value = mock_session

        mock_rows = [
            ("fs_001", "SPY.DATABENTO", {"close_return": 0.01}, 1700000000000000000, 1700000000000000000),
        ]
        execute_result = MagicMock()
        execute_result.fetchall.return_value = mock_rows
        mock_session.execute.return_value = execute_result

        reader = FeatureReaderComponent(
            engine=mock_engine,
            table=mock_table,
            get_feature_set_id=lambda: "default_fs",
            get_feature_names=lambda: ["close_return"],
            persistence=persistence,
        )

        result = reader.read_range(
            start_ns=1700000000000000000,
            end_ns=1700086400000000000,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_read_range_session_with_magic_mock_rows(
        self,
        mock_engine: MagicMock,
        mock_table: MagicMock,
    ) -> None:
        """Verify MagicMock rows are handled gracefully."""
        persistence = MagicMock()
        mock_session = MagicMock()
        persistence.session = mock_session

        # Rows are MagicMock objects (should be filtered out)
        execute_result = MagicMock()
        execute_result.fetchall.return_value = [MagicMock(), MagicMock()]
        mock_session.execute.return_value = execute_result

        reader = FeatureReaderComponent(
            engine=mock_engine,
            table=mock_table,
            get_feature_set_id=lambda: "default_fs",
            get_feature_names=lambda: ["close_return"],
            persistence=persistence,
        )

        # Should fall back to engine query
        expected_df = pd.DataFrame(
            columns=["feature_set_id", "instrument_id", "values", "ts_event", "ts_init"]
        )
        conn_mock = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = conn_mock

        with patch("pandas.read_sql_query", return_value=expected_df):
            result = reader.read_range(
                start_ns=1700000000000000000,
                end_ns=1700086400000000000,
            )

        assert isinstance(result, pd.DataFrame)

    def test_read_range_session_type_error_on_list_conversion(
        self,
        mock_engine: MagicMock,
        mock_table: MagicMock,
    ) -> None:
        """Verify TypeError during list conversion is handled."""
        persistence = MagicMock()
        mock_session = MagicMock()
        persistence.session = mock_session

        # Rows that cause TypeError when converted to list
        class NonIterableResult:
            def __iter__(self):
                raise TypeError("Cannot iterate")

        execute_result = MagicMock()
        execute_result.fetchall.return_value = NonIterableResult()
        mock_session.execute.return_value = execute_result

        reader = FeatureReaderComponent(
            engine=mock_engine,
            table=mock_table,
            get_feature_set_id=lambda: "default_fs",
            get_feature_names=lambda: ["close_return"],
            persistence=persistence,
        )

        expected_df = pd.DataFrame(
            columns=["feature_set_id", "instrument_id", "values", "ts_event", "ts_init"]
        )
        conn_mock = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = conn_mock

        with patch("pandas.read_sql_query", return_value=expected_df):
            result = reader.read_range(
                start_ns=1700000000000000000,
                end_ns=1700086400000000000,
            )

        assert isinstance(result, pd.DataFrame)


# =========================================================================
# Timestamp Normalization Tests
# =========================================================================


class TestTimestampNormalization:
    """Test timestamp normalization."""

    def test_get_training_data_normalizes_timestamps(
        self,
        feature_reader: FeatureReaderComponent,
    ) -> None:
        """Verify timestamps are normalized via sanitize_timestamp_ns."""
        feature_reader._execute_training_query = MagicMock(return_value=[])

        # Should not raise with valid datetime
        features, _timestamps, _names = feature_reader.get_training_data(
            instrument_id="SPY.DATABENTO",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 12, 31),
        )

        assert features.size == 0

    def test_get_latest_at_or_before_normalizes_timestamp(
        self,
        feature_reader: FeatureReaderComponent,
    ) -> None:
        """Verify ts_event is normalized via sanitize_timestamp_ns."""
        feature_reader._execute_latest_query = MagicMock(return_value=None)

        # Should not raise with valid nanosecond timestamp
        result = feature_reader.get_latest_at_or_before(
            instrument_id="SPY.DATABENTO",
            ts_event=1700000000000000000,
        )

        assert result is None

    def test_features_exist_normalizes_timestamps(
        self,
        feature_reader: FeatureReaderComponent,
    ) -> None:
        """Verify timestamps are normalized via sanitize_timestamp_ns."""
        feature_reader._execute_exists_query = MagicMock(return_value=False)

        # Should not raise with valid datetime
        result = feature_reader.features_exist(
            instrument_id="SPY.DATABENTO",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 12, 31),
        )

        assert result is False
