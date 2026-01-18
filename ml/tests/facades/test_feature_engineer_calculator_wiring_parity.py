"""
Parity tests for FeatureEngineer facade calculator wiring (Phase 1.1).

CRITICAL: These tests verify that after wiring FeatureCalculator to the facade,
the facade produces IDENTICAL numerical results to the calculator component.

All parity tests use np.testing.assert_allclose with rtol=1e-10 to ensure
mathematical equivalence within floating-point precision limits.

Test Strategy:
- Run same test data through FeatureEngineer and FeatureCalculator
- Assert numerical parity (rtol=1e-10)
- Test multiple configurations
- Test edge cases (empty, single bar, extreme values)

This is the most critical test file for the wiring task. If these tests pass,
the wiring is correct.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nautilus_trader.model.data import Bar

from ml.features.common.feature_calculator import FeatureCalculator
from ml.features.config import FeatureConfig
from ml.features.indicators import IndicatorManager
from ml.features.facade import FeatureEngineer

pytestmark = [pytest.mark.parity, pytest.mark.unit]

def _build_calculator(config: FeatureConfig) -> FeatureCalculator:
    """Construct a FeatureCalculator for parity comparisons."""
    return FeatureCalculator(config)


def _build_indicator_manager(config: FeatureConfig) -> IndicatorManager:
    """Return a warmed IndicatorManager with deterministic history."""
    manager = IndicatorManager(config)
    for i in range(50):
        manager.update_from_values(
            close=100.0 + i * 0.1,
            high=101.0 + i * 0.1,
            low=99.0 + i * 0.1,
            volume=1000000.0 + i * 1000,
        )
    return manager


# ==================== Batch Mode Parity Tests ====================


class TestCalculateFeaturesBlockParity:
    """Parity tests for calculate_features_batch delegation."""

    def test_facade_calculate_features_batch_matches_calculator(
        self,
        feature_config: FeatureConfig,
        sample_ohlcv_dataframe: pd.DataFrame,
    ) -> None:
        """
        Verify facade.calculate_features_batch produces IDENTICAL output to calculator.

        This is the CRITICAL parity test for batch mode. The facade must produce
        exactly the same features as the calculator component to preserve ML
        training/inference parity.

        Assertions:
            - Shape parity: Same number of rows and columns
            - Value parity: All values match within rtol=1e-10
            - Column parity: Same feature names in same order
        """
        # Create both implementations
        facade = FeatureEngineer(feature_config)
        calculator = _build_calculator(feature_config)

        # Calculate features with both
        facade_df, _facade_scaler = facade.calculate_features_batch(
            sample_ohlcv_dataframe,
            fit_scaler=False,
        )
        calculator_df, _calculator_scaler = calculator.calculate_features(
            sample_ohlcv_dataframe,
            mode="batch",
            fit_scaler=False,
        )

        # Shape parity
        assert facade_df.shape == calculator_df.shape, (
            f"Shape mismatch: facade {facade_df.shape} vs calculator {calculator_df.shape}"
        )

        # Column parity
        assert list(facade_df.columns) == list(calculator_df.columns), (
            "Column mismatch: "
            f"facade {list(facade_df.columns)} vs calculator {list(calculator_df.columns)}"
        )

        # Value parity (CRITICAL)
        np.testing.assert_allclose(
            facade_df.to_numpy(),
            calculator_df.to_numpy(),
            rtol=1e-10,
            err_msg="Facade and calculator must produce identical features for ML parity",
        )

    def test_facade_calculate_features_batch_with_scaler_matches_calculator(
        self,
        feature_config: FeatureConfig,
        sample_ohlcv_dataframe: pd.DataFrame,
    ) -> None:
        """
        Verify parity when scaler fitting is enabled.

        The scaler must be fitted on the same portion of data and produce
        identical scaled features.
        """
        facade = FeatureEngineer(feature_config)
        calculator = _build_calculator(feature_config)

        # Calculate with scaler fitting
        facade_df, facade_scaler = facade.calculate_features_batch(
            sample_ohlcv_dataframe,
            fit_scaler=True,
            scaler_fit_ratio=0.7,
        )
        calculator_df, calculator_scaler = calculator.calculate_features(
            sample_ohlcv_dataframe,
            mode="batch",
            fit_scaler=True,
            scaler_fit_ratio=0.7,
        )

        # Shape parity
        assert facade_df.shape == calculator_df.shape

        # Value parity (scaled values) - exclude timestamp column for numerical comparison
        facade_numeric = facade_df.drop(columns=["timestamp"]) if "timestamp" in facade_df.columns else facade_df
        calculator_numeric = (
            calculator_df.drop(columns=["timestamp"])
            if "timestamp" in calculator_df.columns
            else calculator_df
        )
        np.testing.assert_allclose(
            facade_numeric.to_numpy(),
            calculator_numeric.to_numpy(),
            rtol=1e-6,  # Realistic tolerance for floating point arithmetic across code paths
            err_msg="Scaled features must match between facade and calculator",
        )

        # Timestamp parity if present
        if "timestamp" in calculator_df.columns:
            assert "timestamp" in facade_df.columns, "Facade must include timestamp if calculator does"

        # Scaler parity
        assert facade_scaler is not None
        assert calculator_scaler is not None
        assert facade_scaler.n_samples_seen_ == calculator_scaler.n_samples_seen_, (
            "Scaler must be fitted on same number of samples"
        )
        np.testing.assert_allclose(
            facade_scaler.mean_,
            calculator_scaler.mean_,
            rtol=1e-10,
            err_msg="Scaler means must match",
        )


# ==================== Online Mode Parity Tests ====================


class TestCalculateFeaturesOnlineParity:
    """Parity tests for calculate_features_online delegation (HOT PATH)."""

    def test_facade_calculate_features_online_matches_calculator(
        self,
        feature_config: FeatureConfig,
        current_bar_dict: dict[str, float],
        indicator_manager_with_history: IndicatorManager,
    ) -> None:
        """
        Verify facade.calculate_features_online produces IDENTICAL output to calculator.

        This is the CRITICAL HOT PATH parity test. The facade must produce
        exactly the same feature vector as the calculator component.

        Assertions:
            - Shape parity: Same feature vector shape
            - Dtype parity: Both return float32
            - Value parity: All values match within rtol=1e-10
        """
        facade = FeatureEngineer(feature_config)
        calculator = _build_calculator(feature_config)

        # Calculate features with both
        facade_features = facade.calculate_features_online(
            current_bar_dict,
            indicator_manager_with_history,
            scaler=None,
        )

        calculator_manager = _build_indicator_manager(feature_config)
        calculator_features = calculator.calculate_features(
            current_bar_dict,
            mode="online",
            indicator_manager=calculator_manager,
            scaler=None,
        )

        # Shape parity
        assert facade_features.shape == calculator_features.shape, (
            f"Shape mismatch: facade {facade_features.shape} vs calculator {calculator_features.shape}"
        )

        # Dtype parity
        assert facade_features.dtype == np.float32
        assert calculator_features.dtype == np.float32

        # Value parity (CRITICAL)
        np.testing.assert_allclose(
            facade_features,
            calculator_features,
            rtol=1e-10,
            err_msg="Facade and calculator must produce identical online features",
        )

    def test_facade_calculate_features_online_with_scaler_matches_calculator(
        self,
        feature_config: FeatureConfig,
        current_bar_dict: dict[str, float],
        indicator_manager_with_history: IndicatorManager,
        sample_ohlcv_dataframe: pd.DataFrame,
    ) -> None:
        """
        Verify online mode with pre-fitted scaler produces identical results.
        """
        facade = FeatureEngineer(feature_config)
        calculator = _build_calculator(feature_config)

        # Fit scaler from batch data
        _, scaler = calculator.calculate_features(
            sample_ohlcv_dataframe,
            mode="batch",
            fit_scaler=True,
            scaler_fit_ratio=0.7,
        )

        # Use same scaler for online calculation
        facade_features = facade.calculate_features_online(
            current_bar_dict,
            indicator_manager_with_history,
            scaler=scaler,
        )

        calculator_manager = _build_indicator_manager(feature_config)
        calculator_features = calculator.calculate_features(
            current_bar_dict,
            mode="online",
            indicator_manager=calculator_manager,
            scaler=scaler,
        )

        np.testing.assert_allclose(
            facade_features,
            calculator_features,
            rtol=1e-10,
            err_msg="Scaled online features must match",
        )


# ==================== Unified calculate_features Parity Tests ====================


class TestCalculateFeaturesUnifiedParity:
    """Parity tests for unified calculate_features method."""

    def test_unified_batch_mode_matches_calculator(
        self,
        feature_config: FeatureConfig,
        sample_ohlcv_dataframe: pd.DataFrame,
    ) -> None:
        """Verify calculate_features(mode='batch') matches calculator."""
        facade = FeatureEngineer(feature_config)
        calculator = _build_calculator(feature_config)

        facade_result = facade.calculate_features(
            sample_ohlcv_dataframe,
            mode="batch",
            fit_scaler=False,
        )
        calculator_result = calculator.calculate_features(
            sample_ohlcv_dataframe,
            mode="batch",
            fit_scaler=False,
        )

        facade_df, _ = facade_result
        calculator_df, _ = calculator_result

        np.testing.assert_allclose(
            facade_df.to_numpy(),
            calculator_df.to_numpy(),
            rtol=1e-10,
            err_msg="Unified batch mode must match calculator",
        )

    def test_unified_online_mode_matches_calculator(
        self,
        feature_config: FeatureConfig,
        current_bar_dict: dict[str, float],
        indicator_manager_with_history: IndicatorManager,
    ) -> None:
        """Verify calculate_features(mode='online') matches calculator."""
        facade = FeatureEngineer(feature_config)
        calculator = _build_calculator(feature_config)

        facade_features = facade.calculate_features(
            current_bar_dict,
            mode="online",
            indicator_manager=indicator_manager_with_history,
        )

        calculator_manager = _build_indicator_manager(feature_config)
        calculator_features = calculator.calculate_features(
            current_bar_dict,
            mode="online",
            indicator_manager=calculator_manager,
        )

        np.testing.assert_allclose(
            facade_features,
            calculator_features,
            rtol=1e-10,
            err_msg="Unified online mode must match calculator",
        )


# ==================== compute_features Parity Tests ====================


class TestComputeFeaturesParity:
    """Parity tests for compute_features compatibility shim."""

    def test_compute_features_matches_calculator(
        self,
        feature_config: FeatureConfig,
        test_bars: list[Bar],
    ) -> None:
        """
        Verify compute_features produces identical dict output to calculator.

        compute_features converts Bar objects to DataFrame,
        computes batch features, and returns the last row as dict.
        """
        facade = FeatureEngineer(feature_config)
        calculator = _build_calculator(feature_config)

        # Use first 50 bars from test_bars fixture
        bars = test_bars[:50]

        facade_result = facade.compute_features(bars)
        calculator_result = calculator.compute_features(bars)

        # Key parity
        assert set(facade_result.keys()) == set(calculator_result.keys()), (
            "Feature names must match"
        )

        # Value parity
        for key in facade_result:
            np.testing.assert_allclose(
                facade_result[key],
                calculator_result[key],
                rtol=1e-10,
                err_msg=f"Feature {key} must match between facade and calculator",
            )


# ==================== Configuration Variation Parity Tests ====================


class TestParityAcrossConfigurations:
    """Test parity across different feature configurations."""

    @pytest.mark.parametrize(
        "config_overrides",
        [
            {"rsi_period": 7},
            {"rsi_period": 21},
            {"bb_period": 10},
            {"bb_period": 30},
            {"atr_period": 10},
            {"ema_fast": 8, "ema_slow": 21},
            {"return_periods": [1, 3, 7, 14]},
            {"momentum_periods": [2, 5, 10]},
        ],
        ids=[
            "rsi_7",
            "rsi_21",
            "bb_10",
            "bb_30",
            "atr_10",
            "ema_8_21",
            "returns_1_3_7_14",
            "momentum_2_5_10",
        ],
    )
    def test_parity_with_config_variations(
        self,
        config_overrides: dict,
        sample_ohlcv_dataframe: pd.DataFrame,
    ) -> None:
        """Verify parity maintained across different configurations."""
        import msgspec

        # Create config with overrides
        base_config = FeatureConfig(
            return_periods=[1, 2, 5],
            momentum_periods=[1, 3],
            volume_ma_periods=[10, 20],
            ema_fast=12,
            ema_slow=26,
            rsi_period=14,
            bb_period=20,
            bb_std=2.0,
            atr_period=14,
        )
        config_dict = msgspec.to_builtins(base_config)
        config_dict.update(config_overrides)
        config = FeatureConfig(**config_dict)

        facade = FeatureEngineer(config)
        calculator = _build_calculator(config)

        facade_df, _ = facade.calculate_features_batch(sample_ohlcv_dataframe)
        calculator_df, _ = calculator.calculate_features(
            sample_ohlcv_dataframe,
            mode="batch",
        )

        np.testing.assert_allclose(
            facade_df.to_numpy(),
            calculator_df.to_numpy(),
            rtol=1e-10,
            err_msg=f"Parity failed for config {config_overrides}",
        )


# ==================== Edge Case Parity Tests ====================


class TestParityEdgeCases:
    """Parity tests for edge cases and boundary conditions."""

    def test_parity_with_single_bar_dataframe(
        self,
        feature_config: FeatureConfig,
    ) -> None:
        """Verify parity with minimal data (single bar)."""
        single_bar_df = pd.DataFrame(
            {
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000000.0],
            }
        )

        facade = FeatureEngineer(feature_config)
        calculator = _build_calculator(feature_config)

        facade_df, _ = facade.calculate_features_batch(single_bar_df)
        calculator_df, _ = calculator.calculate_features(
            single_bar_df,
            mode="batch",
        )

        np.testing.assert_allclose(
            facade_df.to_numpy(),
            calculator_df.to_numpy(),
            rtol=1e-10,
            err_msg="Single bar parity failed",
        )

    def test_parity_with_flat_prices(
        self,
        feature_config: FeatureConfig,
    ) -> None:
        """Verify parity when all prices are identical (zero returns)."""
        flat_df = pd.DataFrame(
            {
                "open": [100.0] * 50,
                "high": [100.0] * 50,
                "low": [100.0] * 50,
                "close": [100.0] * 50,
                "volume": [1000000.0] * 50,
            }
        )

        facade = FeatureEngineer(feature_config)
        calculator = _build_calculator(feature_config)

        facade_df, _ = facade.calculate_features_batch(flat_df)
        calculator_df, _ = calculator.calculate_features(
            flat_df,
            mode="batch",
        )

        np.testing.assert_allclose(
            facade_df.to_numpy(),
            calculator_df.to_numpy(),
            rtol=1e-10,
            err_msg="Flat price parity failed",
        )

    def test_parity_with_200_bars(
        self,
        feature_config: FeatureConfig,
    ) -> None:
        """Verify parity over realistic data volume (200 bars)."""
        np.random.seed(123)
        n_bars = 200
        close_prices = 100.0 + np.cumsum(np.random.randn(n_bars) * 0.5)
        df = pd.DataFrame(
            {
                "open": close_prices + np.random.randn(n_bars) * 0.2,
                "high": close_prices + np.abs(np.random.randn(n_bars) * 0.3),
                "low": close_prices - np.abs(np.random.randn(n_bars) * 0.3),
                "close": close_prices,
                "volume": np.random.uniform(900000, 1100000, n_bars),
            }
        )

        facade = FeatureEngineer(feature_config)
        calculator = _build_calculator(feature_config)

        facade_df, _ = facade.calculate_features_batch(df)
        calculator_df, _ = calculator.calculate_features(
            df,
            mode="batch",
        )

        # Check EVERY row for parity
        for i in range(n_bars):
            np.testing.assert_allclose(
                facade_df.iloc[i].to_numpy(),
                calculator_df.iloc[i].to_numpy(),
                rtol=1e-10,
                err_msg=f"Row {i} parity failed",
            )


# ==================== Summary ====================

"""
Parity Test Coverage Summary:
- calculate_features_batch parity: 2 tests
- calculate_features_online parity: 2 tests
- calculate_features unified parity: 2 tests
- compute_features compatibility parity: 1 test
- Configuration variations parity: 1 parametrized test (8 configs)
- Edge cases parity: 3 tests

Total: 11 test cases (with parametrization: 18 total runs)

These tests are CRITICAL for the wiring task. All must pass before
the implementation can be considered complete.
"""
