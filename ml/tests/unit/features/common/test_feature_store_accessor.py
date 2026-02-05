"""
Unit tests for FeatureStoreAccessor component.

Tests cover all three methods with various scenarios including happy paths,
error conditions, and edge cases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pandas as pd
import pytest

from ml.features.common.feature_store_accessor import FeatureStoreAccessor


if TYPE_CHECKING:
    pass


@pytest.mark.unit
class TestReadFeaturesFromStore:
    """Tests for read_features_from_store method."""

    def test_read_features_from_store_success(
        self,
        mock_feature_store: Mock,
        valid_timestamps: dict[str, int],
    ) -> None:
        """Verify reading features from store works with valid inputs."""
        accessor = FeatureStoreAccessor(feature_store=mock_feature_store)

        result = accessor.read_features_from_store(
            "SPY",
            valid_timestamps["ts_start"],
            valid_timestamps["ts_end"],
        )

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert "instrument_id" in result.columns
        assert "ts_event" in result.columns
        assert "ts_init" in result.columns
        assert len(result) > 0

        # Verify store was called correctly
        mock_feature_store.read_range.assert_called_once_with(
            start_ns=valid_timestamps["ts_start"],
            end_ns=valid_timestamps["ts_end"],
            instrument_id="SPY",
        )

    def test_read_features_from_store_with_feature_filter(
        self,
        mock_feature_store: Mock,
        valid_timestamps: dict[str, int],
    ) -> None:
        """Verify reading specific features (not all) works."""
        accessor = FeatureStoreAccessor(feature_store=mock_feature_store)
        feature_names = ["price_sma_20", "rsi_14", "volume_ratio_20"]

        result = accessor.read_features_from_store(
            "SPY",
            valid_timestamps["ts_start"],
            valid_timestamps["ts_end"],
            feature_names=feature_names,
        )

        assert result is not None
        feature_cols = [c for c in result.columns if c not in ["instrument_id", "ts_event", "ts_init"]]
        assert set(feature_cols) == set(feature_names)

    def test_read_features_from_store_not_found(
        self,
        mock_feature_store: Mock,
        valid_timestamps: dict[str, int],
    ) -> None:
        """Verify graceful handling when no features found for requested time range."""
        # Configure mock to return empty DataFrame
        mock_feature_store.read_range.return_value = pd.DataFrame()

        accessor = FeatureStoreAccessor(feature_store=mock_feature_store)

        result = accessor.read_features_from_store(
            "NONEXISTENT",
            valid_timestamps["ts_start"],
            valid_timestamps["ts_end"],
        )

        assert result is None

    def test_read_features_from_store_no_store_available(
        self,
        valid_timestamps: dict[str, int],
    ) -> None:
        """Verify graceful degradation when FeatureStore is None."""
        accessor = FeatureStoreAccessor(feature_store=None)

        result = accessor.read_features_from_store(
            "SPY",
            valid_timestamps["ts_start"],
            valid_timestamps["ts_end"],
        )

        assert result is None

    def test_read_features_from_store_exception_handling(
        self,
        mock_feature_store: Mock,
        valid_timestamps: dict[str, int],
    ) -> None:
        """Verify exceptions during read are caught and logged."""
        # Configure mock to raise exception
        mock_feature_store.read_range.side_effect = Exception("Database connection failed")

        accessor = FeatureStoreAccessor(feature_store=mock_feature_store)

        result = accessor.read_features_from_store(
            "SPY",
            valid_timestamps["ts_start"],
            valid_timestamps["ts_end"],
        )

        assert result is None


@pytest.mark.unit
class TestWriteFeaturesToStore:
    """Tests for write_features_to_store method."""

    def test_write_features_to_store_success(
        self,
        mock_feature_store: Mock,
        sample_features_df: pd.DataFrame,
        valid_timestamps: dict[str, int],
    ) -> None:
        """Verify writing features to store works."""
        accessor = FeatureStoreAccessor(feature_store=mock_feature_store)

        success = accessor.write_features_to_store(
            "SPY",
            sample_features_df,
            ts_event=valid_timestamps["ts_event"],
            ts_init=valid_timestamps["ts_init"],
        )

        assert success is True

        # Verify write_features was called for each row
        assert mock_feature_store.write_features.call_count == len(sample_features_df)

    def test_write_features_to_store_invalid_timestamps(
        self,
        mock_feature_store: Mock,
        sample_features_df: pd.DataFrame,
    ) -> None:
        """Verify write fails when ts_init < ts_event."""
        accessor = FeatureStoreAccessor(feature_store=mock_feature_store)

        # ts_init earlier than ts_event (invalid)
        success = accessor.write_features_to_store(
            "SPY",
            sample_features_df,
            ts_event=1609459200000000100,
            ts_init=1609459200000000000,  # Before ts_event!
        )

        assert success is False

        # Verify write_features was NOT called
        mock_feature_store.write_features.assert_not_called()

    def test_write_features_to_store_no_store_available(
        self,
        sample_features_df: pd.DataFrame,
        valid_timestamps: dict[str, int],
    ) -> None:
        """Verify graceful degradation when FeatureStore is None."""
        accessor = FeatureStoreAccessor(feature_store=None)

        success = accessor.write_features_to_store(
            "SPY",
            sample_features_df,
            ts_event=valid_timestamps["ts_event"],
            ts_init=valid_timestamps["ts_init"],
        )

        assert success is False

    def test_write_features_to_store_exception_handling(
        self,
        mock_feature_store: Mock,
        sample_features_df: pd.DataFrame,
        valid_timestamps: dict[str, int],
    ) -> None:
        """Verify exceptions during write are caught and logged."""
        # Configure mock to raise exception
        mock_feature_store.write_features.side_effect = Exception("Write failed")

        accessor = FeatureStoreAccessor(feature_store=mock_feature_store)

        success = accessor.write_features_to_store(
            "SPY",
            sample_features_df,
            ts_event=valid_timestamps["ts_event"],
            ts_init=valid_timestamps["ts_init"],
        )

        assert success is False


@pytest.mark.unit
class TestValidateFeatureSchema:
    """Tests for validate_feature_schema method."""

    def test_validate_feature_schema_valid(self) -> None:
        """Verify schema validation passes for valid data."""
        accessor = FeatureStoreAccessor()

        features_df = pd.DataFrame({
            "price_sma_20": [100.5],
            "rsi_14": [55.3],
            "volume_ratio_20": [1.2],
        })
        expected_columns = ["price_sma_20", "rsi_14", "volume_ratio_20"]

        is_valid, errors = accessor.validate_feature_schema(
            features_df,
            expected_columns=expected_columns,
        )

        assert is_valid is True
        assert errors == []

    def test_validate_feature_schema_missing_columns(self) -> None:
        """Verify validation fails when required columns missing."""
        accessor = FeatureStoreAccessor()

        features_df = pd.DataFrame({
            "price_sma_20": [100.5],
            # Missing "rsi_14" and "volume_ratio_20"
        })
        expected_columns = ["price_sma_20", "rsi_14", "volume_ratio_20"]

        is_valid, errors = accessor.validate_feature_schema(
            features_df,
            expected_columns=expected_columns,
        )

        assert is_valid is False
        assert len(errors) > 0
        assert "rsi_14" in errors[0]
        assert "volume_ratio_20" in errors[0]

    def test_validate_feature_schema_extra_columns_strict(self) -> None:
        """Verify strict validation rejects extra columns."""
        accessor = FeatureStoreAccessor()

        features_df = pd.DataFrame({
            "price_sma_20": [100.5],
            "rsi_14": [55.3],
            "volume_ratio_20": [1.2],
            "unexpected_feature": [999],  # Extra column
        })
        expected_columns = ["price_sma_20", "rsi_14", "volume_ratio_20"]

        is_valid, errors = accessor.validate_feature_schema(
            features_df,
            expected_columns=expected_columns,
            strict=True,
        )

        assert is_valid is False
        assert len(errors) > 0
        assert "unexpected_feature" in errors[0]

    def test_validate_feature_schema_extra_columns_permissive(self) -> None:
        """Verify permissive validation allows extra columns."""
        accessor = FeatureStoreAccessor()

        features_df = pd.DataFrame({
            "price_sma_20": [100.5],
            "rsi_14": [55.3],
            "volume_ratio_20": [1.2],
            "unexpected_feature": [999],  # Extra column
        })
        expected_columns = ["price_sma_20", "rsi_14", "volume_ratio_20"]

        is_valid, errors = accessor.validate_feature_schema(
            features_df,
            expected_columns=expected_columns,
            strict=False,
        )

        assert is_valid is True
        assert errors == []

    def test_validate_feature_schema_empty_dataframe(self) -> None:
        """Verify validation fails for empty DataFrame."""
        accessor = FeatureStoreAccessor()

        features_df = pd.DataFrame({
            "price_sma_20": [],
            "rsi_14": [],
        })

        is_valid, errors = accessor.validate_feature_schema(
            features_df,
            expected_columns=["price_sma_20", "rsi_14"],
        )

        assert is_valid is False
        assert len(errors) > 0
        assert "empty" in errors[0].lower()

    def test_validate_feature_schema_no_expected_columns(self) -> None:
        """Verify validation with no expected columns just checks non-empty."""
        accessor = FeatureStoreAccessor()

        features_df = pd.DataFrame({
            "any_column": [1, 2, 3],
        })

        is_valid, errors = accessor.validate_feature_schema(
            features_df,
            expected_columns=None,
        )

        assert is_valid is True
        assert errors == []
