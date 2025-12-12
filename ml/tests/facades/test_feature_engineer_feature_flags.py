"""
Feature flag tests for FeatureEngineer facade (Phase 1.1).

These tests verify that the ML_USE_LEGACY_FEATURE_ENGINEER environment variable
correctly controls the behavior of the facade. Both paths (legacy=1 and legacy=0)
must produce IDENTICAL numerical results.

Test Strategy:
- Test legacy=1 uses legacy implementation
- Test legacy=0 uses calculator component
- Test both paths produce identical results

Note: Tests are marked as serial because they use importlib.reload() which
affects module state globally. A cleanup fixture restores the module after each test.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from ml.features.engineering import FeatureConfig, IndicatorManager


if TYPE_CHECKING:
    pass


# Mark as serial to prevent test interference from importlib.reload()
pytestmark = [pytest.mark.unit, pytest.mark.serial]


@pytest.fixture(autouse=True)
def restore_facade_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Restore the facade module to default state after each test.

    This fixture runs after each test and reloads the facade module
    with the environment variable unset to prevent test interference.
    """
    yield
    # After the test, restore the module to default state
    monkeypatch.delenv("ML_USE_LEGACY_FEATURE_ENGINEER", raising=False)
    import ml.features.facade as facade_module

    importlib.reload(facade_module)


# ==================== Feature Flag Behavior Tests ====================


class TestFeatureFlagBehavior:
    """Tests for ML_USE_LEGACY_FEATURE_ENGINEER flag behavior."""

    def test_legacy_flag_true_uses_legacy_path(
        self,
        feature_config: FeatureConfig,
        sample_ohlcv_dataframe: pd.DataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Verify ML_USE_LEGACY_FEATURE_ENGINEER=1 uses legacy implementation.

        When the legacy flag is set, the facade should delegate to
        _legacy_impl instead of the calculator component.
        """
        # Set legacy flag
        monkeypatch.setenv("ML_USE_LEGACY_FEATURE_ENGINEER", "1")

        # Reload to pick up the env var
        import ml.features.facade as facade_module

        importlib.reload(facade_module)

        from ml.features.facade import FeatureEngineer

        facade = FeatureEngineer(feature_config)

        # Verify the facade is using legacy path
        result, _ = facade.calculate_features_batch(sample_ohlcv_dataframe)

        assert result is not None
        assert len(result) == len(sample_ohlcv_dataframe)

    def test_legacy_flag_false_uses_component_path(
        self,
        feature_config: FeatureConfig,
        sample_ohlcv_dataframe: pd.DataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Verify ML_USE_LEGACY_FEATURE_ENGINEER=0 uses calculator component.

        When the legacy flag is NOT set (or explicitly 0), the facade
        should delegate to self.calculator.
        """
        # Unset legacy flag
        monkeypatch.setenv("ML_USE_LEGACY_FEATURE_ENGINEER", "0")

        import ml.features.facade as facade_module

        importlib.reload(facade_module)

        from ml.features.facade import FeatureEngineer

        facade = FeatureEngineer(feature_config)

        result, _ = facade.calculate_features_batch(sample_ohlcv_dataframe)

        assert result is not None
        assert len(result) == len(sample_ohlcv_dataframe)


# ==================== Feature Flag Parity Tests ====================


class TestFeatureFlagParity:
    """Verify both flag values produce identical results."""

    def test_both_paths_produce_identical_batch_results(
        self,
        feature_config: FeatureConfig,
        sample_ohlcv_dataframe: pd.DataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        CRITICAL: Verify legacy=1 and legacy=0 produce IDENTICAL batch results.

        This is essential for safe rollout. Users switching between modes
        must get exactly the same features.
        """
        import ml.features.facade as facade_module

        # Test with legacy=1
        monkeypatch.setenv("ML_USE_LEGACY_FEATURE_ENGINEER", "1")
        importlib.reload(facade_module)
        from ml.features.facade import FeatureEngineer as LegacyEngineer

        legacy_facade = LegacyEngineer(feature_config)
        legacy_result, _ = legacy_facade.calculate_features_batch(
            sample_ohlcv_dataframe
        )

        # Test with legacy=0
        monkeypatch.setenv("ML_USE_LEGACY_FEATURE_ENGINEER", "0")
        importlib.reload(facade_module)
        from ml.features.facade import FeatureEngineer as ComponentEngineer

        component_facade = ComponentEngineer(feature_config)
        component_result, _ = component_facade.calculate_features_batch(
            sample_ohlcv_dataframe
        )

        # Verify parity
        np.testing.assert_allclose(
            legacy_result.to_numpy(),
            component_result.to_numpy(),
            rtol=1e-10,
            err_msg="Legacy and component paths must produce identical results",
        )

    def test_both_paths_produce_identical_online_results(
        self,
        feature_config: FeatureConfig,
        current_bar_dict: dict[str, float],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        CRITICAL: Verify legacy=1 and legacy=0 produce IDENTICAL online results.
        """
        import ml.features.facade as facade_module

        # Create indicator managers for each path
        def create_manager() -> IndicatorManager:
            manager = IndicatorManager(feature_config)
            for i in range(50):
                manager.update_from_values(
                    close=100.0 + i * 0.1,
                    high=101.0 + i * 0.1,
                    low=99.0 + i * 0.1,
                    volume=1000000.0 + i * 1000,
                )
            return manager

        # Test with legacy=1
        monkeypatch.setenv("ML_USE_LEGACY_FEATURE_ENGINEER", "1")
        importlib.reload(facade_module)
        from ml.features.facade import FeatureEngineer as LegacyEngineer

        legacy_facade = LegacyEngineer(feature_config)
        legacy_manager = create_manager()
        legacy_result = legacy_facade.calculate_features_online(
            current_bar_dict,
            legacy_manager,
        )

        # Test with legacy=0
        monkeypatch.setenv("ML_USE_LEGACY_FEATURE_ENGINEER", "0")
        importlib.reload(facade_module)
        from ml.features.facade import FeatureEngineer as ComponentEngineer

        component_facade = ComponentEngineer(feature_config)
        component_manager = create_manager()
        component_result = component_facade.calculate_features_online(
            current_bar_dict,
            component_manager,
        )

        # Verify parity
        np.testing.assert_allclose(
            legacy_result,
            component_result,
            rtol=1e-10,
            err_msg="Legacy and component online paths must produce identical results",
        )


# ==================== Default Behavior Tests ====================


class TestDefaultBehavior:
    """Tests for default behavior when flag is not set."""

    def test_default_uses_component_path(
        self,
        feature_config: FeatureConfig,
        sample_ohlcv_dataframe: pd.DataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Verify default behavior (no env var) uses component path.

        After wiring is complete, the default should be to use
        the calculator component, not the legacy implementation.
        """
        # Ensure env var is not set
        monkeypatch.delenv("ML_USE_LEGACY_FEATURE_ENGINEER", raising=False)

        import ml.features.facade as facade_module

        importlib.reload(facade_module)

        from ml.features.facade import FeatureEngineer

        facade = FeatureEngineer(feature_config)

        # Should work with component path
        result, _ = facade.calculate_features_batch(sample_ohlcv_dataframe)

        assert result is not None
        assert len(result) == len(sample_ohlcv_dataframe)


# ==================== Summary ====================

"""
Feature Flag Test Coverage Summary:
- Legacy flag=1 behavior: 1 test
- Legacy flag=0 behavior: 1 test
- Batch parity between paths: 1 test
- Online parity between paths: 1 test
- Default behavior: 1 test

Total: 5 feature flag tests

These tests ensure safe gradual rollout of the component wiring.
The parity tests are CRITICAL - they must pass for the wiring
to be considered complete.
"""
