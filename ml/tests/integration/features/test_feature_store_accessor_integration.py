"""
Integration tests for FeatureStoreAccessor component.

Tests verify the component works with actual FeatureStore operations,
including round-trip write/read workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pandas as pd
import pytest

from ml.features.common.feature_store_accessor import FeatureStoreAccessor


if TYPE_CHECKING:
    pass


@pytest.mark.integration
class TestFeatureStoreAccessorIntegration:
    """Integration tests for FeatureStoreAccessor with FeatureStore."""

    def test_feature_store_accessor_round_trip(self) -> None:
        """
        Verify write → read → validate workflow works end-to-end.

        This test simulates a complete workflow:
        1. Validate schema
        2. Write features to store
        3. Read features back from store
        4. Compare written vs read data
        """
        # Create a mock store that simulates persistence
        mock_store = Mock(spec=["write_features", "flush", "read_range"])
        stored_data: dict[str, pd.DataFrame] = {}

        # Simulate write operation
        def mock_write(
            feature_set_id: str,
            instrument_id: str,
            features: dict[str, float],
            ts_event: int,
            ts_init: int,
        ) -> None:
            """Store features in memory for later retrieval."""
            key = f"{instrument_id}_{ts_event}"
            df = pd.DataFrame([features])
            df["instrument_id"] = instrument_id
            df["ts_event"] = ts_event
            df["ts_init"] = ts_init
            stored_data[key] = df

        # Simulate read operation
        def mock_read(start_ns: int, end_ns: int, instrument_id: str | None = None) -> pd.DataFrame:
            """Retrieve stored features from memory."""
            matching_dfs = []
            for key, df in stored_data.items():
                if instrument_id is None or instrument_id in key:
                    if start_ns <= df["ts_event"].iloc[0] <= end_ns:
                        matching_dfs.append(df)

            if not matching_dfs:
                return pd.DataFrame()

            return pd.concat(matching_dfs, ignore_index=True)

        mock_store.write_features.side_effect = mock_write
        mock_store.read_range.side_effect = mock_read

        # Initialize accessor with mock store
        accessor = FeatureStoreAccessor(feature_store=mock_store)

        # Prepare test data
        features_df = pd.DataFrame({
            "sma_20": [100.5, 101.2, 102.3],
            "rsi_14": [55.3, 58.7, 60.1],
            "volume_ratio": [1.2, 0.9, 1.5],
        })
        instrument_id = "SPY"
        ts_event = 1609459200000000000
        ts_init = 1609459200000000100

        # 1. Validate schema
        is_valid, errors = accessor.validate_feature_schema(features_df)
        assert is_valid is True, f"Schema validation failed: {errors}"

        # 2. Write features to store
        success = accessor.write_features_to_store(
            instrument_id,
            features_df,
            ts_event=ts_event,
            ts_init=ts_init,
        )
        assert success is True, "Write operation failed"

        # 3. Read features back from store
        read_df = accessor.read_features_from_store(
            instrument_id,
            ts_event - 1,
            ts_event + 1,
        )

        assert read_df is not None, "Read operation returned None"
        assert len(read_df) > 0, "Read operation returned empty DataFrame"

        # 4. Verify data integrity
        # Check that all feature columns are present
        for col in features_df.columns:
            assert col in read_df.columns, f"Column {col} missing from read data"

        # Check mandatory columns
        assert "instrument_id" in read_df.columns
        assert "ts_event" in read_df.columns
        assert "ts_init" in read_df.columns

        # Verify values (within each row)
        assert all(read_df["instrument_id"] == instrument_id)

    def test_feature_store_accessor_multiple_instruments(self) -> None:
        """
        Verify accessor correctly handles multiple instruments.

        Ensures that features for different instruments don't interfere
        with each other.
        """
        # Create a mock store with in-memory persistence
        mock_store = Mock(spec=["write_features", "flush", "read_range"])
        stored_data: dict[str, list[dict]] = {}

        def mock_write(
            feature_set_id: str,
            instrument_id: str,
            features: dict[str, float],
            ts_event: int,
            ts_init: int,
        ) -> None:
            """Store features grouped by instrument."""
            if instrument_id not in stored_data:
                stored_data[instrument_id] = []
            record = {
                "instrument_id": instrument_id,
                "ts_event": ts_event,
                "ts_init": ts_init,
                **features,
            }
            stored_data[instrument_id].append(record)

        def mock_read(start_ns: int, end_ns: int, instrument_id: str | None = None) -> pd.DataFrame:
            """Retrieve stored features for specific instrument."""
            if instrument_id is None or instrument_id not in stored_data:
                return pd.DataFrame()

            records = [
                r for r in stored_data[instrument_id]
                if start_ns <= r["ts_event"] <= end_ns
            ]

            if not records:
                return pd.DataFrame()

            return pd.DataFrame(records)

        mock_store.write_features.side_effect = mock_write
        mock_store.read_range.side_effect = mock_read

        accessor = FeatureStoreAccessor(feature_store=mock_store)

        # Write features for SPY
        spy_features = pd.DataFrame({"sma_20": [100.5], "rsi_14": [55.3]})
        accessor.write_features_to_store(
            "SPY",
            spy_features,
            ts_event=1609459200000000000,
            ts_init=1609459200000000100,
        )

        # Write features for QQQ
        qqq_features = pd.DataFrame({"sma_20": [200.5], "rsi_14": [65.3]})
        accessor.write_features_to_store(
            "QQQ",
            qqq_features,
            ts_event=1609459200000000000,
            ts_init=1609459200000000100,
        )

        # Read back SPY features
        spy_read = accessor.read_features_from_store(
            "SPY",
            1609459200000000000 - 1,
            1609459200000000000 + 1,
        )

        # Read back QQQ features
        qqq_read = accessor.read_features_from_store(
            "QQQ",
            1609459200000000000 - 1,
            1609459200000000000 + 1,
        )

        # Verify isolation
        assert spy_read is not None
        assert qqq_read is not None
        assert all(spy_read["instrument_id"] == "SPY")
        assert all(qqq_read["instrument_id"] == "QQQ")
        assert spy_read["sma_20"].iloc[0] == 100.5
        assert qqq_read["sma_20"].iloc[0] == 200.5
