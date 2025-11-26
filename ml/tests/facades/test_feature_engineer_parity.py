"""Parity tests for FeatureEngineer facade vs legacy implementation.

CRITICAL: These tests verify that legacy and facade implementations produce
IDENTICAL numerical results. This is essential for ML training/inference parity.

All parity tests use np.testing.assert_allclose with rtol=1e-10 to ensure
mathematical equivalence within floating-point precision limits.

Test Strategy:
- Run same test data through both implementations
- Assert numerical parity (rtol=1e-10)
- Test multiple configurations
- Test edge cases (empty, single bar, etc.)
- Test performance parity (within 10%)

"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from ml.config.base import MLFeatureConfig
    from nautilus_trader.model.data import Bar


pytestmark = pytest.mark.parity  # Mark all tests as parity tests


class TestFeatureEngineerParity:
    """Test mathematical parity between legacy and facade implementations."""

    @pytest.mark.skip(reason="Phase 1: Test design - to be implemented in Phase 2")
    def test_legacy_vs_facade_compute_features_identical_single_bar(
        self,
        feature_config: MLFeatureConfig,
        test_bar: Bar,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify legacy and facade produce IDENTICAL features for a single bar.

        This is the most basic parity test - if this fails, facade is broken.

        Args:
            feature_config: Standard feature configuration
            test_bar: Single test bar with OHLCV data
            monkeypatch: Pytest fixture for environment variable manipulation

        Expected Behavior:
            - Both implementations return numpy arrays
            - Arrays have identical shape
            - All feature values match within rtol=1e-10
            - dtypes match

        Assertions:
            - Shape parity
            - Dtype parity
            - Numerical parity (rtol=1e-10)

        """
        # Legacy mode
        monkeypatch.setenv("ML_USE_LEGACY_FEATURE_ENGINEER", "1")
        import importlib

        import ml.features

        importlib.reload(ml.features)
        from ml.features import FeatureEngineer as LegacyEngineer

        legacy_engineer = LegacyEngineer(feature_config)
        legacy_features = legacy_engineer.compute_features([test_bar])

        # Facade mode
        monkeypatch.setenv("ML_USE_LEGACY_FEATURE_ENGINEER", "0")
        importlib.reload(ml.features)
        from ml.features import FeatureEngineer as FacadeEngineer

        facade_engineer = FacadeEngineer(feature_config)
        facade_features = facade_engineer.compute_features([test_bar])

        # Assert parity
        assert legacy_features.shape == facade_features.shape, \
            f"Shape mismatch: legacy {legacy_features.shape} vs facade {facade_features.shape}"

        assert legacy_features.dtype == facade_features.dtype, \
            f"Dtype mismatch: legacy {legacy_features.dtype} vs facade {facade_features.dtype}"

        np.testing.assert_allclose(
            legacy_features,
            facade_features,
            rtol=1e-10,
            err_msg="Legacy and Facade must produce identical features for ML parity",
        )

    @pytest.mark.skip(reason="Phase 1: Test design - to be implemented in Phase 2")
    def test_legacy_vs_facade_compute_features_identical_100_bars(
        self,
        feature_config: MLFeatureConfig,
        test_bars_100: list[Bar],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify parity over realistic data volume (100 bars).

        Tests that parity holds over a realistic workload, not just single bars.

        Args:
            feature_config: Standard feature configuration
            test_bars_100: 100 test bars spanning multiple days
            monkeypatch: Pytest fixture for environment variable manipulation

        Expected Behavior:
            - Both return same feature matrix shape (100, n_features)
            - Element-wise parity within rtol=1e-10
            - All 100 bars processed identically

        Assertions:
            - Shape parity
            - Numerical parity for EVERY bar (row-wise comparison)

        """
        # Legacy mode
        monkeypatch.setenv("ML_USE_LEGACY_FEATURE_ENGINEER", "1")
        import importlib

        import ml.features

        importlib.reload(ml.features)
        from ml.features import FeatureEngineer as LegacyEngineer

        legacy_engineer = LegacyEngineer(feature_config)
        legacy_features = legacy_engineer.compute_features(test_bars_100)

        # Facade mode
        monkeypatch.setenv("ML_USE_LEGACY_FEATURE_ENGINEER", "0")
        importlib.reload(ml.features)
        from ml.features import FeatureEngineer as FacadeEngineer

        facade_engineer = FacadeEngineer(feature_config)
        facade_features = facade_engineer.compute_features(test_bars_100)

        # Assert shape parity
        assert legacy_features.shape == facade_features.shape

        # Test EVERY feature value for EVERY bar
        for i in range(len(test_bars_100)):
            np.testing.assert_allclose(
                legacy_features[i],
                facade_features[i],
                rtol=1e-10,
                err_msg=f"Bar {i} features differ between legacy and facade",
            )

    @pytest.mark.skip(reason="Phase 1: Test design - to be implemented in Phase 2")
    @pytest.mark.parametrize("lookback", [10, 20, 50, 100])
    def test_parity_with_different_lookback_periods(
        self,
        lookback: int,
        test_bars_200: list[Bar],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify parity across different lookback window configurations.

        Args:
            lookback: Lookback window size to test
            test_bars_200: 200 test bars (enough for max lookback)
            monkeypatch: Pytest fixture for environment variable manipulation

        Expected Behavior:
            - Parity maintained for all lookback periods
            - Results independent of configuration parameter

        """
        from ml.config.base import MLFeatureConfig

        config = MLFeatureConfig(lookback_window=lookback)

        # Legacy mode
        monkeypatch.setenv("ML_USE_LEGACY_FEATURE_ENGINEER", "1")
        import importlib

        import ml.features

        importlib.reload(ml.features)
        from ml.features import FeatureEngineer as LegacyEngineer

        legacy_engineer = LegacyEngineer(config)
        legacy_features = legacy_engineer.compute_features(test_bars_200)

        # Facade mode
        monkeypatch.setenv("ML_USE_LEGACY_FEATURE_ENGINEER", "0")
        importlib.reload(ml.features)
        from ml.features import FeatureEngineer as FacadeEngineer

        facade_engineer = FacadeEngineer(config)
        facade_features = facade_engineer.compute_features(test_bars_200)

        np.testing.assert_allclose(
            legacy_features,
            facade_features,
            rtol=1e-10,
            err_msg=f"Parity failed for lookback_window={lookback}",
        )

    @pytest.mark.skip(reason="Phase 1: Test design - to be implemented in Phase 2")
    @pytest.mark.parametrize("normalize", [True, False])
    def test_parity_with_normalization(
        self,
        normalize: bool,
        test_bars_100: list[Bar],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify parity with normalization enabled and disabled.

        Args:
            normalize: Whether to normalize features
            test_bars_100: 100 test bars
            monkeypatch: Pytest fixture for environment variable manipulation

        Expected Behavior:
            - Parity maintained for both normalization settings
            - Normalization produces different values but both implementations match

        """
        from ml.config.base import MLFeatureConfig

        config = MLFeatureConfig(normalize=normalize)

        # Legacy mode
        monkeypatch.setenv("ML_USE_LEGACY_FEATURE_ENGINEER", "1")
        import importlib

        import ml.features

        importlib.reload(ml.features)
        from ml.features import FeatureEngineer as LegacyEngineer

        legacy_engineer = LegacyEngineer(config)
        legacy_features = legacy_engineer.compute_features(test_bars_100)

        # Facade mode
        monkeypatch.setenv("ML_USE_LEGACY_FEATURE_ENGINEER", "0")
        importlib.reload(ml.features)
        from ml.features import FeatureEngineer as FacadeEngineer

        facade_engineer = FacadeEngineer(config)
        facade_features = facade_engineer.compute_features(test_bars_100)

        np.testing.assert_allclose(
            legacy_features,
            facade_features,
            rtol=1e-10,
            err_msg=f"Parity failed for normalize={normalize}",
        )

    @pytest.mark.skip(reason="Phase 1: Test design - to be implemented in Phase 2")
    def test_parity_with_different_indicators(
        self,
        test_bars_100: list[Bar],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify parity across different indicator configurations.

        Tests RSI period, Bollinger Band period, ATR period variations.

        Args:
            test_bars_100: 100 test bars
            monkeypatch: Pytest fixture for environment variable manipulation

        Expected Behavior:
            - Parity maintained for all indicator configurations

        """
        from ml.config.base import MLFeatureConfig

        configs = [
            MLFeatureConfig(rsi_period=7),
            MLFeatureConfig(rsi_period=14),
            MLFeatureConfig(rsi_period=21),
            MLFeatureConfig(bb_period=10),
            MLFeatureConfig(bb_period=20),
            MLFeatureConfig(atr_period=14),
        ]

        for config in configs:
            # Legacy mode
            monkeypatch.setenv("ML_USE_LEGACY_FEATURE_ENGINEER", "1")
            import importlib

            import ml.features

            importlib.reload(ml.features)
            from ml.features import FeatureEngineer as LegacyEngineer

            legacy_engineer = LegacyEngineer(config)
            legacy_features = legacy_engineer.compute_features(test_bars_100)

            # Facade mode
            monkeypatch.setenv("ML_USE_LEGACY_FEATURE_ENGINEER", "0")
            importlib.reload(ml.features)
            from ml.features import FeatureEngineer as FacadeEngineer

            facade_engineer = FacadeEngineer(config)
            facade_features = facade_engineer.compute_features(test_bars_100)

            np.testing.assert_allclose(
                legacy_features,
                facade_features,
                rtol=1e-10,
                err_msg=f"Parity failed for config {config}",
            )

    @pytest.mark.skip(reason="Phase 1: Test design - to be implemented in Phase 2")
    def test_parity_with_edge_cases_empty_data(
        self,
        feature_config: MLFeatureConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify both implementations handle empty data identically.

        Args:
            feature_config: Standard feature configuration
            monkeypatch: Pytest fixture for environment variable manipulation

        Expected Behavior:
            - Both raise ValueError OR return empty array
            - Error messages match (if exception raised)

        """
        # Legacy mode
        monkeypatch.setenv("ML_USE_LEGACY_FEATURE_ENGINEER", "1")
        import importlib

        import ml.features

        importlib.reload(ml.features)
        from ml.features import FeatureEngineer as LegacyEngineer

        legacy_engineer = LegacyEngineer(feature_config)

        # Facade mode
        monkeypatch.setenv("ML_USE_LEGACY_FEATURE_ENGINEER", "0")
        importlib.reload(ml.features)
        from ml.features import FeatureEngineer as FacadeEngineer

        facade_engineer = FacadeEngineer(feature_config)

        # Both should raise ValueError
        with pytest.raises(ValueError, match=r"No bars provided|empty"):
            legacy_engineer.compute_features([])

        with pytest.raises(ValueError, match=r"No bars provided|empty"):
            facade_engineer.compute_features([])

    @pytest.mark.skip(reason="Phase 1: Test design - to be implemented in Phase 2")
    def test_parity_with_edge_cases_single_bar(
        self,
        feature_config: MLFeatureConfig,
        test_bar: Bar,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify parity with minimal data (single bar).

        Args:
            feature_config: Standard feature configuration
            test_bar: Single test bar
            monkeypatch: Pytest fixture for environment variable manipulation

        Expected Behavior:
            - Both handle single bar gracefully
            - Results match

        """
        # Legacy mode
        monkeypatch.setenv("ML_USE_LEGACY_FEATURE_ENGINEER", "1")
        import importlib

        import ml.features

        importlib.reload(ml.features)
        from ml.features import FeatureEngineer as LegacyEngineer

        legacy_engineer = LegacyEngineer(feature_config)
        legacy_features = legacy_engineer.compute_features([test_bar])

        # Facade mode
        monkeypatch.setenv("ML_USE_LEGACY_FEATURE_ENGINEER", "0")
        importlib.reload(ml.features)
        from ml.features import FeatureEngineer as FacadeEngineer

        facade_engineer = FacadeEngineer(feature_config)
        facade_features = facade_engineer.compute_features([test_bar])

        np.testing.assert_allclose(
            legacy_features,
            facade_features,
            rtol=1e-10,
            err_msg="Parity failed for single bar edge case",
        )

    @pytest.mark.skip(reason="Phase 1: Test design - to be implemented in Phase 2")
    def test_parity_with_multiple_instruments(
        self,
        feature_config: MLFeatureConfig,
        multi_instrument_bars: dict[str, list[Bar]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify parity when computing features for multiple symbols.

        Args:
            feature_config: Standard feature configuration
            multi_instrument_bars: Bars for SPY, QQQ, IWM
            monkeypatch: Pytest fixture for environment variable manipulation

        Expected Behavior:
            - Features for each instrument identical between implementations

        """
        # Legacy mode
        monkeypatch.setenv("ML_USE_LEGACY_FEATURE_ENGINEER", "1")
        import importlib

        import ml.features

        importlib.reload(ml.features)
        from ml.features import FeatureEngineer as LegacyEngineer

        legacy_engineer = LegacyEngineer(feature_config)

        # Facade mode
        monkeypatch.setenv("ML_USE_LEGACY_FEATURE_ENGINEER", "0")
        importlib.reload(ml.features)
        from ml.features import FeatureEngineer as FacadeEngineer

        facade_engineer = FacadeEngineer(feature_config)

        for symbol, bars in multi_instrument_bars.items():
            legacy_features = legacy_engineer.compute_features(bars)
            facade_features = facade_engineer.compute_features(bars)

            np.testing.assert_allclose(
                legacy_features,
                facade_features,
                rtol=1e-10,
                err_msg=f"Parity failed for symbol {symbol}",
            )

    @pytest.mark.skip(reason="Phase 1: Test design - to be implemented in Phase 2")
    @pytest.mark.slow
    def test_parity_performance_within_10_percent(
        self,
        feature_config: MLFeatureConfig,
        test_bars_1000: list[Bar],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify facade performance overhead acceptable (<10%).

        Args:
            feature_config: Standard feature configuration
            test_bars_1000: 1000 bars (realistic production workload)
            monkeypatch: Pytest fixture for environment variable manipulation

        Expected Behavior:
            - Facade P99 <= legacy_p99 * 1.10 (within 10%)
            - Numerical results still match (parity)

        """
        # Legacy mode
        monkeypatch.setenv("ML_USE_LEGACY_FEATURE_ENGINEER", "1")
        import importlib

        import ml.features

        importlib.reload(ml.features)
        from ml.features import FeatureEngineer as LegacyEngineer

        legacy_engineer = LegacyEngineer(feature_config)

        # Legacy timing
        times_legacy = []
        for _ in range(100):
            start = time.perf_counter()
            legacy_features = legacy_engineer.compute_features(test_bars_1000)
            times_legacy.append(time.perf_counter() - start)
        legacy_p99 = np.percentile(times_legacy, 99)

        # Facade mode
        monkeypatch.setenv("ML_USE_LEGACY_FEATURE_ENGINEER", "0")
        importlib.reload(ml.features)
        from ml.features import FeatureEngineer as FacadeEngineer

        facade_engineer = FacadeEngineer(feature_config)

        # Facade timing
        times_facade = []
        for _ in range(100):
            start = time.perf_counter()
            facade_features = facade_engineer.compute_features(test_bars_1000)
            times_facade.append(time.perf_counter() - start)
        facade_p99 = np.percentile(times_facade, 99)

        # Performance parity (within 10%)
        assert facade_p99 <= legacy_p99 * 1.10, \
            f"Facade P99 {facade_p99*1000:.2f}ms exceeds 110% of legacy {legacy_p99*1000:.2f}ms"

        # Numerical parity (still must match!)
        np.testing.assert_allclose(
            legacy_features,
            facade_features,
            rtol=1e-10,
            err_msg="Parity failed even though performance acceptable",
        )
