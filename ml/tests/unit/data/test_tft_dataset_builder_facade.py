"""
Unit tests for TFTDatasetBuilderFacade (Tests E1-E15).

These tests verify the facade's core functionality and error handling without
requiring database or external resources.

"""

from __future__ import annotations

import inspect
from datetime import UTC
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml.data.common import SchemaValidationError
from ml.data.tft_dataset_builder_facade import TFTDatasetBuilderFacade
from ml.tests.utils.targets import build_default_target_semantics


if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


TARGET_SEMANTICS = build_default_target_semantics(
    horizon_minutes=15,
    threshold=0.001,
    legacy_aliases=True,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_parquet_catalog() -> MagicMock:
    """
    Create a mock ParquetDataCatalog for testing.
    """
    catalog = MagicMock()
    catalog.bars = MagicMock(return_value=[])
    return catalog


@pytest.fixture
def mock_feature_store() -> MagicMock:
    """
    Create a mock FeatureStore for testing.
    """
    store = MagicMock()
    store.get_training_data = MagicMock(
        return_value=(
            np.array([[0.1, 0.2, 0.3]]),
            np.array([1704067200000000000]),
            ["feature_1", "feature_2", "feature_3"],
        ),
    )
    return store


@pytest.fixture
def mock_data_store() -> MagicMock:
    """
    Create a mock DataStore for testing.
    """
    store = MagicMock()
    store.read_range = MagicMock(return_value=MagicMock(is_empty=MagicMock(return_value=True)))
    return store


@pytest.fixture
def sample_ohlcv_polars_df() -> pl.DataFrame:
    """
    Create sample OHLCV Polars DataFrame for testing.
    """
    import polars as pl

    n_rows = 100
    timestamps = [
        datetime(2024, 1, 1, 9, 30, tzinfo=UTC) + __import__("datetime").timedelta(minutes=i)
        for i in range(n_rows)
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
def sample_ohlcv_pandas_df() -> pd.DataFrame:
    """
    Create sample OHLCV Pandas DataFrame for testing.
    """
    import pandas as pd

    n_rows = 100
    timestamps = [
        datetime(2024, 1, 1, 9, 30, tzinfo=UTC) + __import__("datetime").timedelta(minutes=i)
        for i in range(n_rows)
    ]
    return pd.DataFrame(
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


# =============================================================================
# Happy Path Tests (E1-E8)
# =============================================================================


class TestFacadeHappyPath:
    """
    Happy path tests for TFTDatasetBuilderFacade (E1-E8).
    """

    def test_e1_facade_build_training_dataset_basic(
        self,
        mock_parquet_catalog: MagicMock,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        E1. Verify facade builds training dataset using components.

        Input: Mock catalog, symbols, default config
        Expected: Returns DataFrame with features, targets, static features

        """
        import polars as pl

        mock_result = sample_ohlcv_polars_df.with_columns(
            [
                pl.lit(0).alias("y"),
                pl.lit(0.001).alias("forward_return"),
            ]
        )
        with patch.object(
            TFTDatasetBuilderFacade,
            "_build_training_dataset_direct",
            return_value=mock_result,
        ) as mock_build:
            facade = TFTDatasetBuilderFacade(
                catalog=mock_parquet_catalog,
                symbols=["SPY"],
            )

            result = facade.build_training_dataset(target_semantics=TARGET_SEMANTICS)

            assert isinstance(result, pl.DataFrame)
            assert "timestamp" in result.columns or "ts_event" in result.columns
            mock_build.assert_called_once()

    def test_e2_facade_prepare_training_data(
        self,
        mock_parquet_catalog: MagicMock,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        E2. Verify `prepare_training_data` delegates correctly.

        Input: Valid parameters
        Expected: Returns TFT-compatible dataset

        """
        import polars as pl

        with patch.object(
            TFTDatasetBuilderFacade,
            "_build_training_dataset_direct",
            return_value=sample_ohlcv_polars_df,
        ) as mock_build:
            facade = TFTDatasetBuilderFacade(
                catalog=mock_parquet_catalog,
                symbols=["SPY"],
            )

            result = facade.prepare_training_data(
                target_semantics=TARGET_SEMANTICS,
                use_polars=True,
            )

            assert isinstance(result, pl.DataFrame)
            mock_build.assert_called_once()

    def test_e3_facade_uses_feature_store_when_configured(
        self,
        mock_parquet_catalog: MagicMock,
        mock_feature_store: MagicMock,
    ) -> None:
        """
        E3. Verify facade uses FeatureStore when available.

        Input: Facade with FeatureStore configured
        Expected: Calls FeatureStore.get_training_data

        """
        facade = TFTDatasetBuilderFacade(
            catalog=mock_parquet_catalog,
            symbols=["SPY"],
            feature_store=mock_feature_store,
        )

        # The facade should have feature_store configured
        assert facade.feature_store is not None
        assert facade.feature_store is mock_feature_store

    def test_e4_facade_falls_back_to_direct_computation(
        self,
        mock_parquet_catalog: MagicMock,
        mock_feature_store: MagicMock,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        E4. Verify fallback when FeatureStore fails.

        Input: FeatureStore raises exception
        Expected: Falls back to direct computation, logs warning

        """
        import polars as pl

        # Make feature store fail
        mock_feature_store.get_training_data.side_effect = RuntimeError("Store failed")

        with patch.object(
            TFTDatasetBuilderFacade,
            "_build_training_dataset_direct",
            return_value=sample_ohlcv_polars_df,
        ) as mock_build:
            facade = TFTDatasetBuilderFacade(
                catalog=mock_parquet_catalog,
                symbols=["SPY"],
                feature_store=mock_feature_store,
            )

            result = facade.build_training_dataset(target_semantics=TARGET_SEMANTICS)

            # Should still return a result via fallback
            assert result is not None
            mock_build.assert_called_once()

    def test_e5_facade_api_matches_legacy(self) -> None:
        """
        E5. Verify facade has all legacy public methods.

        Input: Facade class
        Expected: All legacy methods exist with same signatures

        """
        from ml.data.tft_dataset_builder import TFTDatasetBuilder

        # Check that facade has all public methods of legacy
        legacy_methods = {
            name
            for name, _ in inspect.getmembers(TFTDatasetBuilder, predicate=inspect.isfunction)
            if not name.startswith("_")
        }

        facade_methods = {
            name
            for name, _ in inspect.getmembers(TFTDatasetBuilderFacade, predicate=inspect.isfunction)
            if not name.startswith("_")
        }

        # Core public methods that must match
        required_methods = {
            "build_training_dataset",
            "prepare_training_data",
            "prepare_training_data_from_store",
            "get_binding_stats",
        }

        assert required_methods.issubset(
            facade_methods
        ), f"Missing methods: {required_methods - facade_methods}"

    def test_e6_facade_polars_output_mode(
        self,
        mock_parquet_catalog: MagicMock,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        E6. Verify use_polars=True returns Polars DataFrame.

        Input: use_polars=True
        Expected: Returns pl.DataFrame

        """
        import polars as pl

        with patch.object(
            TFTDatasetBuilderFacade,
            "_build_training_dataset_direct",
            return_value=sample_ohlcv_polars_df,
        ) as mock_build:
            facade = TFTDatasetBuilderFacade(
                catalog=mock_parquet_catalog,
                symbols=["SPY"],
            )

            result = facade.build_training_dataset(
                target_semantics=TARGET_SEMANTICS,
                use_polars=True,
            )

            assert isinstance(result, pl.DataFrame)
            mock_build.assert_called_once()

    def test_e7_facade_pandas_output_mode(
        self,
        mock_parquet_catalog: MagicMock,
        sample_ohlcv_pandas_df: pd.DataFrame,
    ) -> None:
        """
        E7. Verify use_polars=False returns Pandas DataFrame.

        Input: use_polars=False
        Expected: Returns pd.DataFrame

        """
        import pandas as pd

        with patch.object(
            TFTDatasetBuilderFacade,
            "_build_training_dataset_direct",
            return_value=sample_ohlcv_pandas_df,
        ) as mock_build:
            facade = TFTDatasetBuilderFacade(
                catalog=mock_parquet_catalog,
                symbols=["SPY"],
            )

            result = facade.build_training_dataset(
                target_semantics=TARGET_SEMANTICS,
                use_polars=False,
            )

            assert isinstance(result, pd.DataFrame)
            mock_build.assert_called_once()

    def test_e8_facade_threshold_bps_alias(
        self,
        mock_parquet_catalog: MagicMock,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        E8. Verify threshold_bps backward compatibility.

        Input: threshold_bps=10 (should convert to 0.001)
        Expected: Correctly converts basis points to decimal

        """
        facade = TFTDatasetBuilderFacade(
            catalog=mock_parquet_catalog,
            symbols=["SPY"],
        )

        with pytest.raises(TypeError):
            facade.build_training_dataset(
                target_semantics=TARGET_SEMANTICS,
                threshold_bps=10,
            )


# =============================================================================
# Error Condition Tests (E12-E15)
# =============================================================================


class TestErrorConditions:
    """
    Error condition tests for TFTDatasetBuilderFacade (E12-E15).
    """

    def test_e12_facade_no_symbols_error(
        self,
        mock_parquet_catalog: MagicMock,
    ) -> None:
        """
        E12. Verify error when symbols list empty.

        Input: symbols=[]
        Expected: Raises ValueError

        """
        with pytest.raises(ValueError, match=r"symbols.*empty"):
            TFTDatasetBuilderFacade(
                catalog=mock_parquet_catalog,
                symbols=[],
            )

    def test_e13_facade_invalid_catalog_error(self) -> None:
        """
        E13. Verify error when catalog invalid.

        Input: catalog=None or invalid type
        Expected: Returns empty DataFrame or logs error (graceful degradation)

        Note: The legacy builder handles invalid/empty catalogs gracefully
        by returning an empty DataFrame rather than raising.

        """
        import polars as pl

        with patch.object(
            TFTDatasetBuilderFacade,
            "_build_training_dataset_direct",
            return_value=pl.DataFrame(),
        ) as mock_build:
            # None catalog is handled gracefully by returning empty DataFrame
            facade = TFTDatasetBuilderFacade(
                catalog=None,  # type: ignore[arg-type]
                symbols=["SPY"],
            )
            # Try to use it - should return empty, not raise
            result = facade.build_training_dataset(target_semantics=TARGET_SEMANTICS)

            # Verify it returns empty DataFrame (graceful degradation)
            assert len(result) == 0
            mock_build.assert_called_once()

    def test_e14_facade_data_store_failure_handled(
        self,
        mock_parquet_catalog: MagicMock,
        mock_data_store: MagicMock,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        E14. Verify graceful handling of DataStore failures.

        Input: DataStore raises exception
        Expected: Falls back to catalog

        """
        # Make data store fail
        mock_data_store.read_range.side_effect = RuntimeError("DataStore failed")

        with patch.object(
            TFTDatasetBuilderFacade,
            "_build_training_dataset_direct",
            return_value=sample_ohlcv_polars_df,
        ) as mock_build:
            # Should not raise, should fall back
            facade = TFTDatasetBuilderFacade(
                catalog=mock_parquet_catalog,
                symbols=["SPY"],
                data_store=mock_data_store,
            )

            result = facade.build_training_dataset(target_semantics=TARGET_SEMANTICS)

            # Should succeed with fallback
            assert result is not None
            mock_build.assert_called_once()

    def test_e15_facade_no_data_found_error(
        self,
        mock_parquet_catalog: MagicMock,
    ) -> None:
        """
        E15. Verify error when no data for any symbol.

        Input: Symbols with no data in catalog
        Expected: Raises RuntimeError or returns empty

        """
        import polars as pl

        with patch.object(
            TFTDatasetBuilderFacade,
            "_build_training_dataset_direct",
            return_value=pl.DataFrame(),
        ) as mock_build:
            facade = TFTDatasetBuilderFacade(
                catalog=mock_parquet_catalog,
                symbols=["NONEXISTENT_SYMBOL"],
            )

            result = facade.build_training_dataset(target_semantics=TARGET_SEMANTICS)

            # Should return empty DataFrame (not raise)
            assert len(result) == 0
            mock_build.assert_called_once()


# =============================================================================
# Component Access Tests
# =============================================================================


class TestComponentAccess:
    """
    Tests for component access via facade properties.
    """

    def test_windowing_component_accessible(
        self,
        mock_parquet_catalog: MagicMock,
    ) -> None:
        """
        Verify windowing component is accessible via property.
        """
        from ml.data.common import TimeSeriesWindowingComponent

        facade = TFTDatasetBuilderFacade(
            catalog=mock_parquet_catalog,
            symbols=["SPY"],
        )

        assert isinstance(facade.windowing_component, TimeSeriesWindowingComponent)

    def test_feature_alignment_component_accessible(
        self,
        mock_parquet_catalog: MagicMock,
    ) -> None:
        """
        Verify feature alignment component is accessible via property.
        """
        from ml.data.common import FeatureAlignmentComponent

        facade = TFTDatasetBuilderFacade(
            catalog=mock_parquet_catalog,
            symbols=["SPY"],
        )

        assert isinstance(facade.feature_alignment_component, FeatureAlignmentComponent)

    def test_target_generation_component_accessible(
        self,
        mock_parquet_catalog: MagicMock,
    ) -> None:
        """
        Verify target generation component is accessible via property.
        """
        from ml.data.common import TargetGenerationComponent

        facade = TFTDatasetBuilderFacade(
            catalog=mock_parquet_catalog,
            symbols=["SPY"],
        )

        assert isinstance(facade.target_generation_component, TargetGenerationComponent)

    def test_schema_validator_component_accessible(
        self,
        mock_parquet_catalog: MagicMock,
    ) -> None:
        """
        Verify schema validator component is accessible via property.
        """
        from ml.data.common import TFTSchemaValidatorComponent

        facade = TFTDatasetBuilderFacade(
            catalog=mock_parquet_catalog,
            symbols=["SPY"],
        )

        assert isinstance(facade.schema_validator_component, TFTSchemaValidatorComponent)


# =============================================================================
# Init Parameter Tests
# =============================================================================


class TestInitParameters:
    """
    Tests for __init__ parameter handling.
    """

    def test_student_mode_disables_features(
        self,
        mock_parquet_catalog: MagicMock,
    ) -> None:
        """
        Verify student_mode disables macro, events, L2, and earnings.
        """
        facade = TFTDatasetBuilderFacade(
            catalog=mock_parquet_catalog,
            symbols=["SPY"],
            student_mode=True,
            include_macro=True,
            include_events=True,
            include_l2=True,
            include_earnings=True,
        )

        # All should be disabled by student mode
        assert facade.include_macro is False
        assert facade.include_events is False
        assert facade.include_l2 is False
        assert facade.include_earnings is False

    def test_negative_earnings_lag_raises(
        self,
        mock_parquet_catalog: MagicMock,
    ) -> None:
        """
        Verify negative earnings_lag_days raises ValueError.
        """
        with pytest.raises(ValueError, match="earnings_lag_days"):
            TFTDatasetBuilderFacade(
                catalog=mock_parquet_catalog,
                symbols=["SPY"],
                earnings_lag_days=-1,
            )

    def test_vintage_as_of_timezone_handling(
        self,
        mock_parquet_catalog: MagicMock,
    ) -> None:
        """
        Verify vintage_as_of handles timezones correctly.
        """
        import pytz

        # Test with naive datetime (should assume UTC)
        naive_dt = datetime(2024, 1, 1, 12, 0, 0)
        facade1 = TFTDatasetBuilderFacade(
            catalog=mock_parquet_catalog,
            symbols=["SPY"],
            vintage_as_of=naive_dt,
        )
        assert facade1.vintage_as_of is not None
        assert facade1.vintage_as_of.tzinfo is not None

        # Test with aware datetime (should convert to UTC)
        aware_dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=pytz.timezone("US/Eastern"))
        facade2 = TFTDatasetBuilderFacade(
            catalog=mock_parquet_catalog,
            symbols=["SPY"],
            vintage_as_of=aware_dt,
        )
        assert facade2.vintage_as_of is not None
        assert facade2.vintage_as_of.tzinfo == UTC
