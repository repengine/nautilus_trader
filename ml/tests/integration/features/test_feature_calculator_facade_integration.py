"""
Integration tests for FeatureCalculator facade wiring (Phase 1.1).

These tests verify end-to-end workflows after wiring the FeatureCalculator
component to the FeatureEngineer facade. They test complete scenarios
rather than isolated functionality.

Test Strategy:
- Test complete batch workflow (load data -> compute features -> scaler)
- Test complete online workflow (warm up -> inference)
- Test scaler sharing between batch and online modes
- Test multi-configuration scenarios
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from ml.features.common.feature_calculator import FeatureCalculator
from ml.features.engineering import FeatureConfig, IndicatorManager
from ml.features.facade import FeatureEngineer


if TYPE_CHECKING:
    pass


pytestmark = [pytest.mark.integration, pytest.mark.serial]


# ==================== End-to-End Batch Workflow Tests ====================


class TestEndToEndBatchWorkflow:
    """Integration tests for complete batch (training) workflow."""

    @pytest.mark.skip(reason="Test design - implementation pending")
    def test_complete_batch_training_workflow(
        self,
        feature_config: FeatureConfig,
        training_dataframe: pd.DataFrame,
    ) -> None:
        """
        Test complete batch training workflow:
        1. Initialize facade
        2. Compute features on training data
        3. Fit scaler
        4. Verify output quality

        This simulates the typical ML training pipeline.
        """
        # Initialize
        facade = FeatureEngineer(feature_config)

        # Compute features with scaler fitting
        features_df, scaler = facade.calculate_features_batch(
            training_dataframe,
            fit_scaler=True,
            scaler_fit_ratio=0.7,
        )

        # Verify output
        assert features_df is not None
        assert len(features_df) == len(training_dataframe)
        assert scaler is not None

        # Verify scaler fitted on correct portion
        expected_fit_samples = int(len(training_dataframe) * 0.7)
        assert scaler.n_samples_seen_ == expected_fit_samples

        # Verify feature quality (no NaN after warmup)
        warmup_period = 50  # Indicators need warmup
        assert not features_df.iloc[warmup_period:].isna().any().any()

        # Verify features are scaled (mean near 0, std near 1)
        # Note: Only check after warmup period
        scaled_features = features_df.iloc[warmup_period:].to_numpy()
        # Check for reasonable scaling (not exact due to fit ratio)
        assert np.abs(scaled_features.mean()) < 1.0

    @pytest.mark.skip(reason="Test design - implementation pending")
    def test_batch_workflow_produces_reproducible_results(
        self,
        feature_config: FeatureConfig,
        training_dataframe: pd.DataFrame,
    ) -> None:
        """
        Verify batch workflow produces deterministic results.

        Running the same data through twice should produce identical features.
        """
        facade1 = FeatureEngineer(feature_config)
        facade2 = FeatureEngineer(feature_config)

        result1, _ = facade1.calculate_features_batch(training_dataframe)
        result2, _ = facade2.calculate_features_batch(training_dataframe)

        np.testing.assert_allclose(
            result1.to_numpy(),
            result2.to_numpy(),
            rtol=1e-10,
            err_msg="Batch results must be reproducible",
        )


# ==================== End-to-End Online Workflow Tests ====================


class TestEndToEndOnlineWorkflow:
    """Integration tests for complete online (inference) workflow."""

    @pytest.mark.skip(reason="Test design - implementation pending")
    def test_complete_online_inference_workflow(
        self,
        feature_config: FeatureConfig,
        inference_bars: list[dict[str, float]],
    ) -> None:
        """
        Test complete online inference workflow:
        1. Initialize facade and indicator manager
        2. Warm up with historical bars
        3. Compute features for each new bar
        4. Verify output quality

        This simulates the typical live trading inference pipeline.
        """
        facade = FeatureEngineer(feature_config)
        indicator_manager = IndicatorManager(feature_config)

        # Warm up indicator manager
        warmup_bars = inference_bars[:50]
        for bar in warmup_bars:
            indicator_manager.update_from_values(
                close=bar["close"],
                high=bar["high"],
                low=bar["low"],
                volume=bar["volume"],
            )

        # Compute features for remaining bars
        features_list = []
        for bar in inference_bars[50:]:
            # Update indicator manager with new bar
            indicator_manager.update_from_values(
                close=bar["close"],
                high=bar["high"],
                low=bar["low"],
                volume=bar["volume"],
            )

            # Compute features
            features = facade.calculate_features_online(
                bar,
                indicator_manager,
            )
            features_list.append(features)

        # Verify output
        assert len(features_list) == 50
        for features in features_list:
            assert isinstance(features, np.ndarray)
            assert features.dtype == np.float32
            assert not np.isnan(features).any()
            assert not np.isinf(features).any()

    @pytest.mark.skip(reason="Test design - implementation pending")
    def test_online_workflow_state_isolation(
        self,
        feature_config: FeatureConfig,
        inference_bars: list[dict[str, float]],
    ) -> None:
        """
        Verify online workflow state isolation between facade instances.

        Two facade instances with separate indicator managers should
        produce independent results.
        """
        # Create two independent pipelines
        facade1 = FeatureEngineer(feature_config)
        facade2 = FeatureEngineer(feature_config)
        manager1 = IndicatorManager(feature_config)
        manager2 = IndicatorManager(feature_config)

        # Warm up both with same data
        for bar in inference_bars[:50]:
            manager1.update_from_values(
                close=bar["close"],
                high=bar["high"],
                low=bar["low"],
                volume=bar["volume"],
            )
            manager2.update_from_values(
                close=bar["close"],
                high=bar["high"],
                low=bar["low"],
                volume=bar["volume"],
            )

        # Compute features from same bar
        test_bar = inference_bars[50]
        manager1.update_from_values(
            close=test_bar["close"],
            high=test_bar["high"],
            low=test_bar["low"],
            volume=test_bar["volume"],
        )
        manager2.update_from_values(
            close=test_bar["close"],
            high=test_bar["high"],
            low=test_bar["low"],
            volume=test_bar["volume"],
        )

        features1 = facade1.calculate_features_online(test_bar, manager1)
        features2 = facade2.calculate_features_online(test_bar, manager2)

        # Should produce identical results
        np.testing.assert_allclose(
            features1,
            features2,
            rtol=1e-10,
            err_msg="Independent pipelines with same data should match",
        )


# ==================== Scaler Sharing Workflow Tests ====================


class TestScalerSharingWorkflow:
    """Integration tests for scaler sharing between batch and online modes."""

    @pytest.mark.skip(reason="Test design - implementation pending")
    def test_scaler_fitted_in_batch_applied_in_online(
        self,
        feature_config: FeatureConfig,
        training_dataframe: pd.DataFrame,
        inference_bars: list[dict[str, float]],
    ) -> None:
        """
        Test the complete training->inference workflow with scaler sharing:
        1. Fit scaler during batch training
        2. Use same scaler for online inference
        3. Verify scaled features are in expected range

        This is the typical production workflow.
        """
        facade = FeatureEngineer(feature_config)

        # Phase 1: Training (batch mode)
        _, scaler = facade.calculate_features_batch(
            training_dataframe,
            fit_scaler=True,
            scaler_fit_ratio=0.7,
        )
        assert scaler is not None

        # Phase 2: Inference (online mode)
        indicator_manager = IndicatorManager(feature_config)

        # Warm up
        for bar in inference_bars[:50]:
            indicator_manager.update_from_values(
                close=bar["close"],
                high=bar["high"],
                low=bar["low"],
                volume=bar["volume"],
            )

        # Inference with scaler
        test_bar = inference_bars[50]
        indicator_manager.update_from_values(
            close=test_bar["close"],
            high=test_bar["high"],
            low=test_bar["low"],
            volume=test_bar["volume"],
        )

        features_scaled = facade.calculate_features_online(
            test_bar,
            indicator_manager,
            scaler=scaler,
        )

        # Verify scaled features are in reasonable range
        assert isinstance(features_scaled, np.ndarray)
        assert features_scaled.dtype == np.float32
        # Scaled features should be roughly standardized
        # Allow for some variation (market conditions differ)
        assert np.abs(features_scaled.mean()) < 5.0

    @pytest.mark.skip(reason="Test design - implementation pending")
    def test_unscaled_features_match_scaled_before_transform(
        self,
        feature_config: FeatureConfig,
        training_dataframe: pd.DataFrame,
        inference_bars: list[dict[str, float]],
    ) -> None:
        """
        Verify that scaler correctly transforms features by comparing
        unscaled and scaled versions.
        """
        facade = FeatureEngineer(feature_config)

        # Get scaler from training
        _, scaler = facade.calculate_features_batch(
            training_dataframe,
            fit_scaler=True,
        )

        # Prepare for online mode
        indicator_manager = IndicatorManager(feature_config)
        for bar in inference_bars[:50]:
            indicator_manager.update_from_values(
                close=bar["close"],
                high=bar["high"],
                low=bar["low"],
                volume=bar["volume"],
            )

        test_bar = inference_bars[50]
        indicator_manager.update_from_values(
            close=test_bar["close"],
            high=test_bar["high"],
            low=test_bar["low"],
            volume=test_bar["volume"],
        )

        # Get features without scaling
        features_unscaled = facade.calculate_features_online(
            test_bar,
            indicator_manager,
            scaler=None,
        )

        # Manually apply scaler
        features_manually_scaled = scaler.transform(
            features_unscaled.reshape(1, -1)
        ).astype(np.float32).ravel()

        # Get features with scaling
        # Need fresh manager to get same state
        indicator_manager2 = IndicatorManager(feature_config)
        for bar in inference_bars[:51]:
            indicator_manager2.update_from_values(
                close=bar["close"],
                high=bar["high"],
                low=bar["low"],
                volume=bar["volume"],
            )

        features_auto_scaled = facade.calculate_features_online(
            test_bar,
            indicator_manager2,
            scaler=scaler,
        )

        # Verify scaling is applied correctly
        np.testing.assert_allclose(
            features_auto_scaled,
            features_manually_scaled,
            rtol=1e-10,
            err_msg="Auto-scaled and manually scaled features should match",
        )


# ==================== Multi-Configuration Workflow Tests ====================


class TestMultiConfigWorkflow:
    """Integration tests for workflows with different configurations."""

    @pytest.mark.skip(reason="Test design - implementation pending")
    @pytest.mark.parametrize(
        "config_name,config_overrides",
        [
            ("minimal", {"return_periods": [1], "momentum_periods": [1]}),
            ("standard", {}),  # Use default config
            ("extended", {"return_periods": [1, 2, 5, 10, 20], "momentum_periods": [1, 3, 5, 10]}),
        ],
    )
    def test_workflow_with_different_configs(
        self,
        config_name: str,
        config_overrides: dict,
        training_dataframe: pd.DataFrame,
    ) -> None:
        """
        Verify workflow works correctly with different feature configurations.
        """
        import msgspec

        base_config = FeatureConfig(
            return_periods=[1, 2, 5],
            momentum_periods=[1, 3],
            volume_ma_periods=[10, 20],
            ema_fast=12,
            ema_slow=26,
        )

        config_dict = msgspec.to_builtins(base_config)
        config_dict.update(config_overrides)
        config = FeatureConfig(**config_dict)

        facade = FeatureEngineer(config)

        # Run workflow
        features_df, scaler = facade.calculate_features_batch(
            training_dataframe,
            fit_scaler=True,
        )

        # Verify
        assert features_df is not None
        assert len(features_df) == len(training_dataframe)
        assert scaler is not None

        # Feature count should match config
        expected_feature_count = len(config.get_feature_names())
        assert len(features_df.columns) == expected_feature_count, (
            f"Config {config_name}: expected {expected_feature_count} features, "
            f"got {len(features_df.columns)}"
        )


# ==================== Summary ====================

"""
Integration Test Coverage Summary:
- End-to-end batch workflow: 2 tests
- End-to-end online workflow: 2 tests
- Scaler sharing workflow: 2 tests
- Multi-configuration workflow: 1 parametrized test (3 configs)

Total: 7 integration tests (9 with parametrization)

These tests verify the complete workflows work correctly after wiring
the FeatureCalculator component to the facade. They simulate real-world
usage patterns for ML training and inference.
"""
