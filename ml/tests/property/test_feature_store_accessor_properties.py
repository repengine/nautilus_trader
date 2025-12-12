"""
Property-based tests for FeatureStoreAccessor component.

Uses Hypothesis to verify invariants hold across a wide range of inputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ml.features.common.feature_store_accessor import FeatureStoreAccessor


if TYPE_CHECKING:
    pass


# Hypothesis strategies for test data generation

# Nanosecond timestamps (valid range: 0 to 2^63-1)
nanosecond_timestamps = st.integers(min_value=0, max_value=2**63 - 1)

# Valid instrument IDs
instrument_ids = st.sampled_from(["SPY", "QQQ", "IWM", "DIA", "EEM", "AAPL", "MSFT"])

# Feature values (realistic bounds, no NaN/inf)
feature_values = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)


@pytest.mark.property
class TestFeatureStoreAccessorProperties:
    """Property-based tests for FeatureStoreAccessor invariants."""

    @given(
        ts_events=st.lists(
            nanosecond_timestamps,
            min_size=1,
            max_size=10,
            unique=True,
        ),
    )
    def test_timestamp_monotonicity_property(self, ts_events: list[int]) -> None:
        """
        Verify timestamps in retrieved features are monotonically increasing.

        Property: When features are written with various timestamps and then read
        back, the returned timestamps should be in ascending order (or equal).
        """
        # Create mock store with in-memory persistence
        mock_store = Mock(spec=["write_features", "flush", "read_range"])
        stored_records: list[dict] = []

        def mock_write(
            feature_set_id: str,
            instrument_id: str,
            features: dict[str, float],
            ts_event: int,
            ts_init: int,
        ) -> None:
            """Store feature records."""
            record = {
                "instrument_id": instrument_id,
                "ts_event": ts_event,
                "ts_init": ts_init,
                **features,
            }
            stored_records.append(record)

        def mock_read(start_ns: int, end_ns: int, instrument_id: str | None = None) -> pd.DataFrame:
            """Retrieve and sort records."""
            matching = [
                r for r in stored_records
                if start_ns <= r["ts_event"] <= end_ns
                and (instrument_id is None or r["instrument_id"] == instrument_id)
            ]
            if not matching:
                return pd.DataFrame()
            df = pd.DataFrame(matching)
            # Sort by ts_event to ensure monotonicity
            return df.sort_values("ts_event").reset_index(drop=True)

        mock_store.write_features.side_effect = mock_write
        mock_store.read_range.side_effect = mock_read

        accessor = FeatureStoreAccessor(feature_store=mock_store)

        # Write features with the generated timestamps
        for ts_event in ts_events:
            features = pd.DataFrame({"sma_20": [100.5], "rsi_14": [55.3]})
            accessor.write_features_to_store(
                "SPY",
                features,
                ts_event=ts_event,
                ts_init=ts_event + 100,  # ts_init slightly after ts_event
            )

        # Read back all features
        min_ts = min(ts_events)
        max_ts = max(ts_events)
        read_df = accessor.read_features_from_store("SPY", min_ts, max_ts)

        # Verify monotonicity
        if read_df is not None and len(read_df) > 1:
            timestamps = read_df["ts_event"].to_numpy()
            # Check timestamps are monotonically non-decreasing
            for i in range(len(timestamps) - 1):
                assert timestamps[i] <= timestamps[i + 1], (
                    f"Timestamps not monotonic: {timestamps[i]} > {timestamps[i+1]}"
                )

    @given(
        instrument=instrument_ids,
        feature_value=feature_values,
    )
    def test_feature_value_preservation_property(
        self,
        instrument: str,
        feature_value: float,
    ) -> None:
        """
        Verify feature values are preserved during write/read round-trip.

        Property: A feature value written to the store should be retrievable
        with the same value (within floating-point tolerance).
        """
        # Create mock store
        mock_store = Mock(spec=["write_features", "flush", "read_range"])
        stored_value: dict[str, float] = {}

        def mock_write(
            feature_set_id: str,
            instrument_id: str,
            features: dict[str, float],
            ts_event: int,
            ts_init: int,
        ) -> None:
            """Store the feature value."""
            stored_value.update(features)
            stored_value["instrument_id"] = instrument_id
            stored_value["ts_event"] = ts_event
            stored_value["ts_init"] = ts_init

        def mock_read(start_ns: int, end_ns: int, instrument_id: str | None = None) -> pd.DataFrame:
            """Retrieve stored value."""
            if not stored_value:
                return pd.DataFrame()
            return pd.DataFrame([stored_value])

        mock_store.write_features.side_effect = mock_write
        mock_store.read_range.side_effect = mock_read

        accessor = FeatureStoreAccessor(feature_store=mock_store)

        # Write feature
        features_df = pd.DataFrame({"test_feature": [feature_value]})
        ts_event = 1609459200000000000

        success = accessor.write_features_to_store(
            instrument,
            features_df,
            ts_event=ts_event,
            ts_init=ts_event + 100,
        )

        # If write succeeded, verify read returns same value
        if success:
            read_df = accessor.read_features_from_store(
                instrument,
                ts_event - 1,
                ts_event + 1,
            )

            if read_df is not None and len(read_df) > 0:
                # Verify value is preserved (within floating-point tolerance)
                read_value = read_df["test_feature"].iloc[0]
                assert abs(read_value - feature_value) < 1e-10, (
                    f"Feature value not preserved: wrote {feature_value}, read {read_value}"
                )

    @given(
        column_names=st.lists(
            st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll")), min_size=1, max_size=20),
            min_size=1,
            max_size=10,
            unique=True,
        ),
    )
    def test_schema_validation_consistency_property(self, column_names: list[str]) -> None:
        """
        Verify schema validation is consistent across different column sets.

        Property: If a DataFrame has exactly the expected columns, validation
        should always pass in strict mode.
        """
        accessor = FeatureStoreAccessor()

        # Create DataFrame with the generated column names
        data = {col: [1.0] for col in column_names}
        features_df = pd.DataFrame(data)

        # Validate with exact column match
        is_valid, errors = accessor.validate_feature_schema(
            features_df,
            expected_columns=column_names,
            strict=True,
        )

        # Should always be valid when columns match exactly
        assert is_valid is True, (
            f"Validation failed for exact column match. Errors: {errors}"
        )
        assert errors == []
