#!/usr/bin/env python3
"""
Test for the critical feature parity fix.

This test validates that the fix properly resolves the mismatch between batch and online
feature computation when microstructure and trade flow features are enabled.

"""

import numpy as np
import polars as pl
import pytest

from ml.features.engineering import FeatureConfig, FeatureEngineer


class TestFeatureParityFix:
    """
    Test suite for the critical feature parity bug fix.
    """

    @pytest.fixture
    def mock_bars_data(self) -> list[dict[str, float]]:
        """
        Generate mock bar data for testing.
        """
        n_bars = 100
        base_price = 1.1000
        rng = np.random.default_rng(42)

        bars_data = []
        current_price = base_price
        for i in range(n_bars):
            price_var = rng.uniform(-0.001, 0.001)
            current_price += price_var
            close_price = current_price
            high_price = close_price + abs(rng.uniform(0, 0.002))
            low_price = close_price - abs(rng.uniform(0, 0.002))
            volume = 1000000 + rng.integers(-100000, 100000)

            bars_data.append(
                {
                    "close": close_price,
                    "high": high_price,
                    "low": low_price,
                    "volume": float(volume),
                    "ts_event": i * 60_000_000_000,  # 1 minute intervals in nanoseconds
                }
            )

        return bars_data

    @pytest.fixture
    def bars_df(self, mock_bars_data: list[dict[str, float]]) -> pl.DataFrame:
        """
        Convert bars data to Polars DataFrame.
        """
        return pl.DataFrame(mock_bars_data)

    def test_feature_count_parity_default_config(
        self,
        mock_bars_data: list[dict[str, float]],
        bars_df: pl.DataFrame,
    ):
        """
        Test that default config (L1 only) maintains feature count parity.
        """
        config = FeatureConfig()
        batch_engineer = FeatureEngineer(config)
        online_engineer = FeatureEngineer(config)

        # Compute batch features
        batch_features, _ = batch_engineer.calculate_features_batch(bars_df)
        batch_count = (
            batch_features.shape[1] if hasattr(batch_features, "shape") else len(batch_features)
        )

        # Compute online features
        online_features = []
        for bar_data in mock_bars_data[-5:]:  # Test last 5 bars
            features = online_engineer.calculate_features_online(
                close_price=bar_data["close"],
                high_price=bar_data["high"],
                low_price=bar_data["low"],
                volume=bar_data["volume"],
            )
            online_features.append(features)

        online_count = len(online_features[0]) if online_features else 0

        assert (
            batch_count == online_count
        ), f"Feature count mismatch: batch={batch_count}, online={online_count}"
        assert batch_count == 26, f"Expected 26 features for default config, got {batch_count}"

    def test_feature_count_parity_microstructure_config(
        self,
        mock_bars_data: list[dict[str, float]],
        bars_df: pl.DataFrame,
    ):
        """
        Test that microstructure config (L1+L2) maintains feature count parity.
        """
        config = FeatureConfig(include_microstructure=True)
        batch_engineer = FeatureEngineer(config)
        online_engineer = FeatureEngineer(config)

        # Compute batch features
        batch_features, _ = batch_engineer.calculate_features_batch(bars_df)
        batch_count = (
            batch_features.shape[1] if hasattr(batch_features, "shape") else len(batch_features)
        )

        # Compute online features
        online_features = []
        for bar_data in mock_bars_data[-5:]:  # Test last 5 bars
            features = online_engineer.calculate_features_online(
                close_price=bar_data["close"],
                high_price=bar_data["high"],
                low_price=bar_data["low"],
                volume=bar_data["volume"],
            )
            online_features.append(features)

        online_count = len(online_features[0]) if online_features else 0

        assert (
            batch_count == online_count
        ), f"Feature count mismatch: batch={batch_count}, online={online_count}"
        assert (
            batch_count == 33
        ), f"Expected 33 features for microstructure config, got {batch_count}"

    def test_feature_count_parity_full_config(
        self,
        mock_bars_data: list[dict[str, float]],
        bars_df: pl.DataFrame,
    ):
        """
        Test that full config (L1+L2+L3) maintains feature count parity.
        """
        config = FeatureConfig(include_microstructure=True, include_trade_flow=True)
        batch_engineer = FeatureEngineer(config)
        online_engineer = FeatureEngineer(config)

        # Compute batch features
        batch_features, _ = batch_engineer.calculate_features_batch(bars_df)
        batch_count = (
            batch_features.shape[1] if hasattr(batch_features, "shape") else len(batch_features)
        )

        # Compute online features
        online_features = []
        for bar_data in mock_bars_data[-5:]:  # Test last 5 bars
            features = online_engineer.calculate_features_online(
                close_price=bar_data["close"],
                high_price=bar_data["high"],
                low_price=bar_data["low"],
                volume=bar_data["volume"],
            )
            online_features.append(features)

        online_count = len(online_features[0]) if online_features else 0

        assert (
            batch_count == online_count
        ), f"Feature count mismatch: batch={batch_count}, online={online_count}"
        assert batch_count == 37, f"Expected 37 features for full config, got {batch_count}"

    def test_microstructure_features_computed_online(self, mock_bars_data: list[dict[str, float]]):
        """
        Test that microstructure features are actually computed in online mode.
        """
        config = FeatureConfig(include_microstructure=True)
        engineer = FeatureEngineer(config)

        # Warm up the engineer with some bars
        for bar_data in mock_bars_data[:50]:
            engineer.calculate_features_online(
                close_price=bar_data["close"],
                high_price=bar_data["high"],
                low_price=bar_data["low"],
                volume=bar_data["volume"],
            )

        # Get features for a test bar
        test_bar = mock_bars_data[50]
        features = engineer.calculate_features_online(
            close_price=test_bar["close"],
            high_price=test_bar["high"],
            low_price=test_bar["low"],
            volume=test_bar["volume"],
        )

        # Check that we have the expected number of features
        assert len(features) == 33, f"Expected 33 features, got {len(features)}"

        # Check that microstructure features are not all zeros
        # (they should have meaningful values after warmup)
        microstructure_start = 26  # First 26 are L1 features
        microstructure_features = features[microstructure_start:]

        # At least some microstructure features should be non-zero
        non_zero_count = np.count_nonzero(microstructure_features)
        assert non_zero_count > 0, "All microstructure features are zero"

    def test_trade_flow_features_computed_online(self, mock_bars_data: list[dict[str, float]]):
        """
        Test that trade flow features are actually computed in online mode.
        """
        config = FeatureConfig(include_microstructure=True, include_trade_flow=True)
        engineer = FeatureEngineer(config)

        # Warm up the engineer with some bars
        for bar_data in mock_bars_data[:50]:
            engineer.calculate_features_online(
                close_price=bar_data["close"],
                high_price=bar_data["high"],
                low_price=bar_data["low"],
                volume=bar_data["volume"],
            )

        # Get features for a test bar
        test_bar = mock_bars_data[50]
        features = engineer.calculate_features_online(
            close_price=test_bar["close"],
            high_price=test_bar["high"],
            low_price=test_bar["low"],
            volume=test_bar["volume"],
        )

        # Check that we have the expected number of features
        assert len(features) == 37, f"Expected 37 features, got {len(features)}"

        # Check that trade flow features are not all zeros
        trade_flow_start = 33  # First 33 are L1+microstructure features
        trade_flow_features = features[trade_flow_start:]

        # At least some trade flow features should be non-zero
        non_zero_count = np.count_nonzero(trade_flow_features)
        assert non_zero_count > 0, "All trade flow features are zero"

        # Specifically check that VWAP is computed (should equal close price as approximation)
        vwap_idx = 1  # Second trade flow feature is VWAP
        vwap_value = trade_flow_features[vwap_idx]
        expected_vwap = test_bar["close"]

        assert (
            abs(vwap_value - expected_vwap) < 1e-6
        ), f"VWAP mismatch: got {vwap_value}, expected {expected_vwap}"

    def test_feature_value_parity_microstructure(
        self,
        mock_bars_data: list[dict[str, float]],
        bars_df: pl.DataFrame,
    ):
        """
        Test that microstructure feature values match between batch and online modes.
        """
        config = FeatureConfig(include_microstructure=True)
        batch_engineer = FeatureEngineer(config)
        online_engineer = FeatureEngineer(config)

        # Compute batch features
        batch_features, _ = batch_engineer.calculate_features_batch(bars_df)
        batch_array = (
            batch_features.to_numpy()
            if hasattr(batch_features, "to_numpy")
            else np.array(batch_features)
        )

        # Compute online features
        online_features = []
        for bar_data in mock_bars_data:
            features = online_engineer.calculate_features_online(
                close_price=bar_data["close"],
                high_price=bar_data["high"],
                low_price=bar_data["low"],
                volume=bar_data["volume"],
            )
            online_features.append(features.copy())

        online_array = np.array(online_features)

        # Compare the last 10 rows (after indicators are warmed up)
        compare_start = len(mock_bars_data) - 10
        batch_subset = batch_array[compare_start:]
        online_subset = online_array[compare_start:]

        # Check for exact parity (our implementation should match exactly)
        abs_diff = np.abs(batch_subset - online_subset)
        max_diff = np.max(abs_diff)

        # Use a reasonable tolerance for floating point precision
        tolerance = 1e-10
        assert (
            max_diff < tolerance
        ), f"Feature value parity failed: max_diff={max_diff:.2e} > {tolerance:.0e}"

    def test_regression_scaler_dimension_mismatch(self, mock_bars_data: list[dict[str, float]]):
        """
        Regression test for the original issue: StandardScaler dimension mismatch.

        This test simulates the original error condition where a model trained with
        37 features (batch mode) would fail when receiving 26 features (online mode).
        """
        # Simulate training scenario: batch mode with microstructure+trade_flow enabled
        config_full = FeatureConfig(include_microstructure=True, include_trade_flow=True)
        training_engineer = FeatureEngineer(config_full)

        bars_df = pl.DataFrame(mock_bars_data)
        batch_features, scaler = training_engineer.calculate_features_batch(
            bars_df, fit_scaler=True
        )

        # Verify training produces 37 features
        feature_count = (
            batch_features.shape[1] if hasattr(batch_features, "shape") else len(batch_features)
        )
        assert feature_count == 37, f"Expected 37 training features, got {feature_count}"

        # Simulate inference scenario: online mode with same config
        inference_engineer = FeatureEngineer(config_full)

        # Warm up the inference engineer
        for bar_data in mock_bars_data[:50]:
            inference_engineer.calculate_features_online(
                close_price=bar_data["close"],
                high_price=bar_data["high"],
                low_price=bar_data["low"],
                volume=bar_data["volume"],
            )

        # Get inference features for a test bar
        test_bar = mock_bars_data[50]
        inference_features = inference_engineer.calculate_features_online(
            close_price=test_bar["close"],
            high_price=test_bar["high"],
            low_price=test_bar["low"],
            volume=test_bar["volume"],
        )

        # Verify inference produces 37 features (same as training)
        inference_count = len(inference_features)
        assert inference_count == 37, f"Expected 37 inference features, got {inference_count}"

        # Verify that the scaler would work (no dimension mismatch)
        if scaler is not None:
            # This would previously fail with "X has 26 features, but StandardScaler is expecting 37"
            try:
                scaled_features = scaler.transform(inference_features.reshape(1, -1))
                assert scaled_features.shape == (
                    1,
                    37,
                ), f"Scaler output shape mismatch: {scaled_features.shape}"
            except Exception as e:
                pytest.fail(f"Scaler transform failed (regression): {e}")
