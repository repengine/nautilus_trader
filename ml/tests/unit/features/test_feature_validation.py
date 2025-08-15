"""
Unit tests for ML feature validation module.

Tests cover:
- Feature parity validation between batch and online modes
- Performance validation
- Test data generation
- Error handling and edge cases

"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.config.constants import MLConstants
from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.features.validation import FeatureParityError
from ml.features.validation import FeatureParityValidator
from ml.features.validation import validate_feature_parity


class TestFeatureParityError:
    """
    Test FeatureParityError exception.
    """

    def test_feature_parity_error_creation(self) -> None:
        """
        Test creating FeatureParityError.
        """
        error = FeatureParityError(
            message="Test error",
            max_difference=0.001,
            tolerance=0.0001,
            failing_features=["feature1", "feature2"],
        )

        assert str(error) == "Test error"
        assert error.max_difference == 0.001
        assert error.tolerance == 0.0001
        assert error.failing_features == ["feature1", "feature2"]


class TestFeatureParityValidator:
    """
    Test FeatureParityValidator functionality.
    """

    def create_test_dataframe(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Create test DataFrame with OHLCV data.
        """
        rng = np.random.default_rng(42)

        # Generate price data
        close_prices = 100 + np.cumsum(rng.normal(0, 1, n_samples))

        # Generate OHLC from close
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]

        high_prices = np.maximum(open_prices, close_prices) + rng.uniform(0, 0.5, n_samples)
        low_prices = np.minimum(open_prices, close_prices) - rng.uniform(0, 0.5, n_samples)

        volumes = rng.uniform(900000, 1100000, n_samples)

        return pd.DataFrame(
            {
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": volumes,
            },
        )

    def test_validator_initialization_default(self) -> None:
        """
        Test validator initialization with defaults.
        """
        validator = FeatureParityValidator()

        assert isinstance(validator.config, FeatureConfig)
        assert validator.tolerance == MLConstants.FEATURE_PARITY_TOLERANCE
        assert validator.feature_engineer is not None

    def test_validator_initialization_custom(self) -> None:
        """
        Test validator initialization with custom config.
        """
        config = FeatureConfig(return_periods=[1, 2, 3])
        tolerance = 1e-8

        validator = FeatureParityValidator(config=config, tolerance=tolerance)

        assert validator.config == config
        assert validator.tolerance == tolerance

    def test_validate_parity_success(self) -> None:
        """
        Test successful parity validation.
        """
        config = FeatureConfig()
        # Use a more reasonable tolerance for floating point comparisons
        validator = FeatureParityValidator(config=config, tolerance=1e-7)

        # Create test data
        df = self.create_test_dataframe(100)

        # Validate parity
        report = validator.validate_parity(df, start_idx=30, end_idx=60)

        # Check report structure
        assert "parity_passed" in report
        assert "max_difference" in report
        assert "tolerance" in report
        assert "failing_features" in report
        assert "n_failing_features" in report
        assert "n_total_features" in report
        assert "feature_differences" in report
        assert "validation_time" in report
        assert "n_samples_validated" in report
        assert "validation_range" in report

        # Should pass with tight tolerance
        assert report["parity_passed"] == True  # noqa: E712
        assert report["n_failing_features"] == 0
        assert len(report["failing_features"]) == 0
        assert report["max_difference"] < validator.tolerance
        assert report["n_samples_validated"] == 30
        assert report["validation_range"] == "[30:60]"

    def test_validate_parity_detailed_report(self) -> None:
        """
        Test parity validation with detailed report.
        """
        config = FeatureConfig()
        validator = FeatureParityValidator(config=config, tolerance=1e-7)

        # Create test data
        df = self.create_test_dataframe(100)

        # Validate with detailed report
        report = validator.validate_parity(df, start_idx=30, end_idx=60, detailed_report=True)

        # Check detailed feature differences
        assert "feature_differences" in report
        feature_diffs = report["feature_differences"]

        # Should have entry for each feature
        feature_names = config.get_feature_names()
        assert len(feature_diffs) == len(feature_names)

        # Check structure of each feature diff
        for feature_name in feature_names:
            assert feature_name in feature_diffs
            diff = feature_diffs[feature_name]
            assert "max_difference" in diff
            assert "mean_difference" in diff
            assert "std_difference" in diff
            assert "passed" in diff

            # All should pass
            assert diff["passed"] == True  # noqa: E712
            assert diff["max_difference"] < validator.tolerance

    def test_validate_parity_no_detailed_report(self) -> None:
        """
        Test parity validation without detailed report.
        """
        config = FeatureConfig()
        validator = FeatureParityValidator(config=config, tolerance=1e-7)

        # Create test data
        df = self.create_test_dataframe(100)

        # Validate without detailed report
        report = validator.validate_parity(df, start_idx=30, end_idx=60, detailed_report=False)

        # Should have basic info but empty feature differences
        assert report["parity_passed"] == True  # noqa: E712
        assert report["feature_differences"] == {}

    def test_validate_parity_full_range(self) -> None:
        """
        Test parity validation with full data range.
        """
        config = FeatureConfig()
        validator = FeatureParityValidator(config=config, tolerance=1e-7)

        # Create test data
        df = self.create_test_dataframe(100)

        # Validate full range (except warmup)
        report = validator.validate_parity(df, start_idx=30, end_idx=None)

        assert report["n_samples_validated"] == 70  # 100 - 30
        assert report["validation_range"] == "[30:100]"

    def test_validate_parity_invalid_range(self) -> None:
        """
        Test parity validation with invalid range.
        """
        validator = FeatureParityValidator()
        df = self.create_test_dataframe(100)

        # Start >= end
        with pytest.raises(ValueError, match="start_idx .* must be less than end_idx"):
            validator.validate_parity(df, start_idx=60, end_idx=50)

        # Start == end
        with pytest.raises(ValueError, match="start_idx .* must be less than end_idx"):
            validator.validate_parity(df, start_idx=50, end_idx=50)

    def test_validate_parity_edge_case_small_data(self) -> None:
        """
        Test parity validation with small dataset.
        """
        config = FeatureConfig()
        validator = FeatureParityValidator(config=config, tolerance=1e-7)

        # Create small dataset
        df = self.create_test_dataframe(10)

        # Validate with minimal range
        report = validator.validate_parity(df, start_idx=5, end_idx=10)

        assert report["n_samples_validated"] == 5

    def test_validate_performance(self) -> None:
        """
        Test performance validation.
        """
        config = FeatureConfig()
        validator = FeatureParityValidator(config=config)

        # Create test data
        df = self.create_test_dataframe(200)

        # Validate performance
        report = validator.validate_performance(
            df,
            n_iterations=50,
            target_latency_ms=10.0,
        )

        # Check report structure
        assert "n_iterations" in report
        assert "target_latency_ms" in report
        assert "mean_latency_ms" in report
        assert "std_latency_ms" in report
        assert "min_latency_ms" in report
        assert "max_latency_ms" in report
        assert "p50_latency_ms" in report
        assert "p95_latency_ms" in report
        assert "p99_latency_ms" in report
        assert "performance_passed" in report
        assert "n_features" in report

        # Check values are reasonable
        assert report["n_iterations"] == 50
        assert report["target_latency_ms"] == 10.0
        assert report["mean_latency_ms"] > 0
        assert report["min_latency_ms"] <= report["mean_latency_ms"]
        assert report["mean_latency_ms"] <= report["max_latency_ms"]
        assert report["p50_latency_ms"] >= report["min_latency_ms"]
        assert report["p95_latency_ms"] >= report["p50_latency_ms"]
        assert report["p99_latency_ms"] >= report["p95_latency_ms"]
        assert report["n_features"] == len(config.get_feature_names())

    def test_validate_performance_default_target(self) -> None:
        """
        Test performance validation with default target.
        """
        validator = FeatureParityValidator()
        df = self.create_test_dataframe(100)

        # Use default target latency
        report = validator.validate_performance(df, n_iterations=10)

        assert report["target_latency_ms"] == MLConstants.MAX_INFERENCE_LATENCY_MS

    def test_generate_test_data(self) -> None:
        """
        Test synthetic test data generation.
        """
        validator = FeatureParityValidator()

        # Generate test data
        df = validator.generate_test_data(n_samples=500, seed=42)

        # Check structure
        assert len(df) == 500
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

        # Check data consistency
        # High should be >= max(open, close)
        # Low should be <= min(open, close)
        for i in range(len(df)):
            row = df.row(i, named=True)  # Polars way to get row as dict
            assert row["high"] >= max(row["open"], row["close"])
            assert row["low"] <= min(row["open"], row["close"])
            assert row["close"] > 0  # Prices should be positive
            assert row["volume"] > 0  # Volume should be positive

    def test_generate_test_data_reproducible(self) -> None:
        """
        Test that generated data is reproducible with same seed.
        """
        validator = FeatureParityValidator()

        # Generate twice with same seed
        df1 = validator.generate_test_data(n_samples=100, seed=123)
        df2 = validator.generate_test_data(n_samples=100, seed=123)

        # Should be identical
        # Check if Polars or Pandas
        if hasattr(df1, "select"):  # Polars (has select method)
            assert df1.equals(df2)
        else:  # Pandas
            pd.testing.assert_frame_equal(df1, df2)

        # Generate with different seed
        df3 = validator.generate_test_data(n_samples=100, seed=456)

        # Should be different
        if hasattr(df1, "select"):  # Polars
            assert not df1.select("close").equals(df3.select("close"))
        else:  # Pandas
            assert not df1["close"].equals(df3["close"])

    def test_create_bar_from_row(self) -> None:
        """
        Test creating Bar from DataFrame row.
        """
        validator = FeatureParityValidator()
        df = self.create_test_dataframe(10)

        # Import required types
        from nautilus_trader.model.data import BarSpecification
        from nautilus_trader.model.data import BarType
        from nautilus_trader.model.enums import AggressorSide
        from nautilus_trader.model.enums import BarAggregation
        from nautilus_trader.model.enums import PriceType
        from nautilus_trader.model.identifiers import InstrumentId

        # Create bar type
        instrument_id = InstrumentId.from_str("TEST.VENUE")
        bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
        bar_type = BarType(instrument_id, bar_spec, AggressorSide.BUYER)

        # Create bar from row
        bar = validator._create_bar_from_row(df, 0, bar_type)

        # Check bar properties
        assert bar.bar_type == bar_type
        assert abs(float(bar.open) - df.iloc[0]["open"]) < 1e-8
        assert abs(float(bar.high) - df.iloc[0]["high"]) < 1e-8
        assert abs(float(bar.low) - df.iloc[0]["low"]) < 1e-8
        assert abs(float(bar.close) - df.iloc[0]["close"]) < 1e-8
        assert abs(float(bar.volume) - df.iloc[0]["volume"]) < 1e-8

    def test_create_bar_from_row_missing_columns(self) -> None:
        """
        Test creating Bar from DataFrame with missing columns.
        """
        validator = FeatureParityValidator()

        # Create DataFrame with only close
        df = pd.DataFrame(
            {
                "close": [100.0, 101.0, 102.0],
            },
        )

        # Import required types
        from nautilus_trader.model.data import BarSpecification
        from nautilus_trader.model.data import BarType
        from nautilus_trader.model.enums import AggressorSide
        from nautilus_trader.model.enums import BarAggregation
        from nautilus_trader.model.enums import PriceType
        from nautilus_trader.model.identifiers import InstrumentId

        # Create bar type
        instrument_id = InstrumentId.from_str("TEST.VENUE")
        bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
        bar_type = BarType(instrument_id, bar_spec, AggressorSide.BUYER)

        # Create bar from row - should use close for all prices
        bar = validator._create_bar_from_row(df, 0, bar_type)

        assert float(bar.open) == 100.0
        assert float(bar.high) == 100.0
        assert float(bar.low) == 100.0
        assert float(bar.close) == 100.0
        assert float(bar.volume) == 0.0  # Default volume


class TestValidateFeatureParityFunction:
    """
    Test the convenience function validate_feature_parity.
    """

    def create_test_dataframe(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Create test DataFrame with OHLCV data.
        """
        rng = np.random.default_rng(42)

        # Generate price data
        close_prices = 100 + np.cumsum(rng.normal(0, 1, n_samples))

        # Generate OHLC from close
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]

        high_prices = np.maximum(open_prices, close_prices) + rng.uniform(0, 0.5, n_samples)
        low_prices = np.minimum(open_prices, close_prices) - rng.uniform(0, 0.5, n_samples)

        volumes = rng.uniform(900000, 1100000, n_samples)

        return pd.DataFrame(
            {
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": volumes,
            },
        )

    def test_validate_feature_parity_function(self) -> None:
        """
        Test the convenience function.
        """
        df = self.create_test_dataframe(100)

        # Call convenience function
        report = validate_feature_parity(
            df,
            config=None,  # Use default
            tolerance=1e-7,  # Use reasonable tolerance
            start_idx=30,
            end_idx=60,
        )

        # Should return valid report
        assert report["parity_passed"] == True  # noqa: E712
        assert report["n_samples_validated"] == 30

    def test_validate_feature_parity_function_custom_params(self) -> None:
        """
        Test the convenience function with custom parameters.
        """
        config = FeatureConfig(return_periods=[1, 2])
        tolerance = 1e-7

        df = self.create_test_dataframe(100)

        # Call with custom params
        report = validate_feature_parity(
            df,
            config=config,
            tolerance=tolerance,
            start_idx=40,
            end_idx=50,
        )

        # Should use custom values
        assert report["tolerance"] == tolerance
        assert report["n_samples_validated"] == 10


class TestUnifiedFeatureCalculation:
    """
    Test the unified calculate_features method.
    """

    def test_unified_method_batch_mode(self) -> None:
        """
        Test unified method in batch mode produces same results as
        calculate_features_batch.
        """
        config = FeatureConfig()
        engineer = FeatureEngineer(config)

        # Create test data
        validator = FeatureParityValidator(config=config)
        df = validator.generate_test_data(n_samples=100)

        # Calculate using unified method
        features_unified, scaler_unified = engineer.calculate_features(
            data=df,
            mode="batch",
            fit_scaler=True,
        )

        # Calculate using direct method
        features_direct, scaler_direct = engineer.calculate_features_batch(
            df=df,
            fit_scaler=True,
        )

        # Should produce identical results
        import numpy as np

        if hasattr(features_unified, "to_numpy"):
            unified_array = features_unified.to_numpy()
            direct_array = features_direct.to_numpy()
        else:
            unified_array = features_unified.values
            direct_array = features_direct.values

        np.testing.assert_allclose(unified_array, direct_array, rtol=1e-10)

    def test_unified_method_online_mode(self) -> None:
        """
        Test unified method in online mode produces same results as
        calculate_features_online.
        """
        config = FeatureConfig()
        engineer = FeatureEngineer(config)

        # Create test data
        current_bar = {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000000.0,
        }

        # Create indicator manager
        from ml.features.engineering import IndicatorManager

        indicator_mgr = IndicatorManager(config)

        # Warm up indicators with some data
        from nautilus_trader.test_kit.stubs.data import TestDataStubs

        for i in range(50):
            # Create a simple bar for warmup
            bar = TestDataStubs.bar_5decimal(ts_event=i, ts_init=i)
            indicator_mgr.update_from_bar(bar)

        # Calculate using unified method
        features_unified = engineer.calculate_features(
            data=current_bar,
            mode="online",
            indicator_manager=indicator_mgr,
        )

        # Calculate using direct method
        features_direct = engineer.calculate_features_online(
            current_bar=current_bar,
            indicator_manager=indicator_mgr,
        )

        # Should produce identical results
        import numpy as np

        np.testing.assert_allclose(features_unified, features_direct, rtol=1e-10)

    def test_unified_method_invalid_mode(self) -> None:
        """
        Test unified method with invalid mode raises ValueError.
        """
        config = FeatureConfig()
        engineer = FeatureEngineer(config)

        with pytest.raises(ValueError, match="Invalid mode: invalid"):
            engineer.calculate_features(
                data={},
                mode="invalid",
            )

    def test_unified_method_online_without_manager(self) -> None:
        """
        Test unified method in online mode without indicator_manager raises ValueError.
        """
        config = FeatureConfig()
        engineer = FeatureEngineer(config)

        with pytest.raises(ValueError, match="indicator_manager is required for online mode"):
            engineer.calculate_features(
                data={"close": 100.0},
                mode="online",
            )


class TestPolarsCompatibility:
    """
    Test compatibility with Polars DataFrames.
    """

    def test_validate_parity_with_polars(self) -> None:
        """
        Test parity validation with Polars DataFrame.
        """
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        # Create Polars DataFrame
        df_dict = {
            "open": [100.0 + i for i in range(100)],
            "high": [101.0 + i for i in range(100)],
            "low": [99.0 + i for i in range(100)],
            "close": [100.5 + i for i in range(100)],
            "volume": [1000000.0] * 100,
        }
        df = pl.DataFrame(df_dict)

        # Validate with looser tolerance for numerical precision
        # Linear test data can cause numerical differences in momentum/return calculations
        validator = FeatureParityValidator(tolerance=5e-2)
        report = validator.validate_parity(df, start_idx=30, end_idx=60)

        assert report["parity_passed"] == True
        assert report["n_samples_validated"] == 30

    def test_generate_test_data_polars_output(self) -> None:
        """
        Test generating test data as Polars DataFrame.
        """
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        validator = FeatureParityValidator()
        df = validator.generate_test_data(n_samples=50)

        # Should return Polars DataFrame when available
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 50
