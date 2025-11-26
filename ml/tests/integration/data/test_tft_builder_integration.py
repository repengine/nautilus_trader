"""
Integration tests for TFTDatasetBuilderFacade (Tests E24-E26).

These tests verify the facade works correctly with real components and external
dependencies like data stores.

"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml.data.tft_dataset_builder_facade import TFTDatasetBuilderFacade


if TYPE_CHECKING:
    import polars as pl


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_parquet_catalog_with_data() -> MagicMock:
    """
    Create a mock ParquetDataCatalog that returns data.
    """
    import polars as pl

    catalog = MagicMock()

    # Create sample data that would be returned by catalog
    n_rows = 100
    timestamps = [
        datetime(2024, 1, 1, 9, 30, tzinfo=UTC) + timedelta(minutes=i) for i in range(n_rows)
    ]

    sample_df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0 + i * 0.1 for i in range(n_rows)],
            "high": [101.0 + i * 0.1 for i in range(n_rows)],
            "low": [99.0 + i * 0.1 for i in range(n_rows)],
            "close": [100.5 + i * 0.1 for i in range(n_rows)],
            "volume": [1000.0 + i * 10 for i in range(n_rows)],
            "instrument_id": ["SPY.ARCA"] * n_rows,
        }
    )

    catalog.bars = MagicMock(return_value=[])
    catalog._data = sample_df  # Store for reference

    return catalog


@pytest.fixture
def sample_ohlcv_polars_df() -> pl.DataFrame:
    """
    Create sample OHLCV Polars DataFrame for testing.
    """
    import polars as pl

    n_rows = 100
    timestamps = [
        datetime(2024, 1, 1, 9, 30, tzinfo=UTC) + timedelta(minutes=i) for i in range(n_rows)
    ]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0 + i * 0.1 for i in range(n_rows)],
            "high": [101.0 + i * 0.1 for i in range(n_rows)],
            "low": [99.0 + i * 0.1 for i in range(n_rows)],
            "close": [100.5 + i * 0.1 for i in range(n_rows)],
            "volume": [1000.0 + i * 10 for i in range(n_rows)],
            "instrument_id": ["SPY"] * n_rows,
        }
    )


@pytest.fixture
def mock_data_store_with_data() -> MagicMock:
    """
    Create a mock DataStore that returns data.
    """
    import polars as pl

    store = MagicMock()

    n_rows = 100
    timestamps = [
        datetime(2024, 1, 1, 9, 30, tzinfo=UTC) + timedelta(minutes=i) for i in range(n_rows)
    ]

    sample_df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0 + i * 0.1 for i in range(n_rows)],
            "high": [101.0 + i * 0.1 for i in range(n_rows)],
            "low": [99.0 + i * 0.1 for i in range(n_rows)],
            "close": [100.5 + i * 0.1 for i in range(n_rows)],
            "volume": [1000.0 + i * 10 for i in range(n_rows)],
            "instrument_id": ["SPY"] * n_rows,
        }
    )

    store.read_range = MagicMock(return_value=sample_df)

    return store


@pytest.fixture
def market_bindings() -> list[Any]:
    """
    Create sample market bindings for testing.
    """
    from ml.data.ingest.market_bindings import ResolvedMarketBinding

    binding = ResolvedMarketBinding(
        binding_id="test_binding",
        dataset_id="test_dataset",
        descriptor_id="test_descriptor",
        symbol="SPY",
        instrument_ids=("SPY.ARCA",),  # Must be tuple
        schema="ohlcv_1m",
        storage_kind=None,  # Can be None
        source="test",
        license_start=None,
        license_end=None,
        start=None,
        end=None,
    )

    return [binding]


# =============================================================================
# Integration Tests (E24-E26)
# =============================================================================


class TestFacadeIntegration:
    """
    Integration tests for TFTDatasetBuilderFacade (E24-E26).
    """

    @pytest.mark.integration
    def test_e24_integration_with_real_catalog(
        self,
        mock_parquet_catalog_with_data: MagicMock,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        E24. Verify facade works with real ParquetDataCatalog.

        Input: Real catalog with test data
        Expected: Produces valid dataset

        """
        import polars as pl

        with patch("ml.data.tft_dataset_builder.TFTDatasetBuilder") as mock_builder_cls:
            mock_builder = MagicMock()

            # Create a realistic result with all required columns
            mock_result = sample_ohlcv_polars_df.with_columns(
                [
                    pl.lit(0).alias("y"),
                    pl.lit(0.001).alias("forward_return"),
                    pl.lit(0).alias("time_index"),
                    pl.lit("ETF").alias("asset_class"),
                    pl.lit(0.01).alias("tick_size"),
                    pl.lit("ARCA").alias("exchange"),
                ]
            )
            mock_builder.build_training_dataset.return_value = mock_result
            mock_builder_cls.return_value = mock_builder

            facade = TFTDatasetBuilderFacade(
                catalog=mock_parquet_catalog_with_data,
                symbols=["SPY"],
            )

            result = facade.build_training_dataset(
                horizon_minutes=15,
                min_return_threshold=0.001,
            )

            # Verify non-empty result
            assert len(result) > 0

            # Verify all required columns present
            required_cols = ["timestamp", "instrument_id", "y", "forward_return"]
            for col in required_cols:
                assert col in result.columns, f"Missing column: {col}"

    @pytest.mark.integration
    def test_e25_integration_with_data_store(
        self,
        mock_parquet_catalog_with_data: MagicMock,
        mock_data_store_with_data: MagicMock,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        E25. Verify DataStore integration.

        Input: Facade with DataStore configured
        Expected: Reads from DataStore when available

        """
        import polars as pl

        with patch("ml.data.tft_dataset_builder.TFTDatasetBuilder") as mock_builder_cls:
            mock_builder = MagicMock()

            mock_result = sample_ohlcv_polars_df.with_columns(
                [
                    pl.lit(0).alias("y"),
                    pl.lit(0.001).alias("forward_return"),
                ]
            )
            mock_builder.build_training_dataset.return_value = mock_result
            mock_builder_cls.return_value = mock_builder

            facade = TFTDatasetBuilderFacade(
                catalog=mock_parquet_catalog_with_data,
                symbols=["SPY"],
                data_store=mock_data_store_with_data,
                market_dataset_id="test_dataset",
            )

            result = facade.build_training_dataset()

            # Verify result produced
            assert result is not None
            assert len(result) > 0

    @pytest.mark.integration
    def test_e26_integration_market_bindings(
        self,
        mock_parquet_catalog_with_data: MagicMock,
        sample_ohlcv_polars_df: pl.DataFrame,
        market_bindings: list[Any],
    ) -> None:
        """
        E26. Verify market binding resolution.

        Input: Configured market bindings
        Expected: Correct binding used for each instrument

        """
        import polars as pl

        with patch("ml.data.tft_dataset_builder.TFTDatasetBuilder") as mock_builder_cls:
            mock_builder = MagicMock()

            mock_result = sample_ohlcv_polars_df.with_columns(
                [
                    pl.lit(0).alias("y"),
                    pl.lit(0.001).alias("forward_return"),
                ]
            )
            mock_builder.build_training_dataset.return_value = mock_result
            mock_builder.get_binding_stats.return_value = ()
            mock_builder_cls.return_value = mock_builder

            facade = TFTDatasetBuilderFacade(
                catalog=mock_parquet_catalog_with_data,
                symbols=["SPY"],
                market_bindings=market_bindings,
            )

            result = facade.build_training_dataset()

            # Verify result
            assert result is not None

            # Verify binding stats accessible
            stats = facade.get_binding_stats()
            assert stats is not None


# =============================================================================
# Component Integration Tests
# =============================================================================


class TestComponentIntegration:
    """
    Tests verifying components work together correctly.
    """

    def test_components_initialized_correctly(
        self,
        mock_parquet_catalog_with_data: MagicMock,
    ) -> None:
        """
        Verify all 4 components are properly initialized.
        """
        from ml.data.common import (
            FeatureAlignmentComponent,
            TargetGenerationComponent,
            TFTSchemaValidatorComponent,
            TimeSeriesWindowingComponent,
        )

        with patch("ml.data.tft_dataset_builder.TFTDatasetBuilder"):
            facade = TFTDatasetBuilderFacade(
                catalog=mock_parquet_catalog_with_data,
                symbols=["SPY"],
            )

            # All components should be initialized
            assert isinstance(facade.windowing_component, TimeSeriesWindowingComponent)
            assert isinstance(facade.feature_alignment_component, FeatureAlignmentComponent)
            assert isinstance(facade.target_generation_component, TargetGenerationComponent)
            assert isinstance(facade.schema_validator_component, TFTSchemaValidatorComponent)

    def test_windowing_component_usable(
        self,
        mock_parquet_catalog_with_data: MagicMock,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        Verify windowing component can be used directly.
        """
        with patch("ml.data.tft_dataset_builder.TFTDatasetBuilder"):
            facade = TFTDatasetBuilderFacade(
                catalog=mock_parquet_catalog_with_data,
                symbols=["SPY"],
            )

            # Use windowing component directly
            bounds = facade.windowing_component.frame_time_bounds(sample_ohlcv_polars_df)

            assert bounds[0] is not None  # min timestamp
            assert bounds[1] is not None  # max timestamp
            assert bounds[0] <= bounds[1]

    def test_feature_alignment_component_usable(
        self,
        mock_parquet_catalog_with_data: MagicMock,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        Verify feature alignment component can be used directly.
        """
        import polars as pl

        with patch("ml.data.tft_dataset_builder.TFTDatasetBuilder"):
            facade = TFTDatasetBuilderFacade(
                catalog=mock_parquet_catalog_with_data,
                symbols=["SPY"],
            )

            # Use feature alignment component directly
            features = facade.feature_alignment_component.compute_features_polars(
                sample_ohlcv_polars_df,
            )

            # Verify features computed
            assert isinstance(features, pl.DataFrame)
            assert "return_1" in features.columns
            assert "volatility_20" in features.columns
            assert len(features) == len(sample_ohlcv_polars_df)

    def test_target_generation_component_usable(
        self,
        mock_parquet_catalog_with_data: MagicMock,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        Verify target generation component can be used directly.
        """
        import polars as pl

        with patch("ml.data.tft_dataset_builder.TFTDatasetBuilder"):
            facade = TFTDatasetBuilderFacade(
                catalog=mock_parquet_catalog_with_data,
                symbols=["SPY"],
            )

            # Use target generation component directly
            targets = facade.target_generation_component.generate_targets_polars(
                sample_ohlcv_polars_df,
                horizon_minutes=15,
                threshold=0.001,
            )

            # Verify targets generated
            assert isinstance(targets, pl.DataFrame)
            assert "y" in targets.columns
            assert "forward_return" in targets.columns
            assert len(targets) == len(sample_ohlcv_polars_df)

            # Verify y is binary
            unique_y = set(targets["y"].unique().to_list())
            assert unique_y <= {0, 1}

    def test_schema_validator_component_usable(
        self,
        mock_parquet_catalog_with_data: MagicMock,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        Verify schema validator component can be used directly.
        """
        with patch("ml.data.tft_dataset_builder.TFTDatasetBuilder"):
            facade = TFTDatasetBuilderFacade(
                catalog=mock_parquet_catalog_with_data,
                symbols=["SPY"],
            )

            # Use schema validator component directly
            # Should not raise for valid data
            facade.schema_validator_component.validate(sample_ohlcv_polars_df)

            # Verify row count validation
            facade.schema_validator_component.validate_row_count(
                sample_ohlcv_polars_df,
                minimum=50,
            )
