"""
Component delegation tests for FeatureEngineer facade (Phase 1.1 - Category 14).

These tests verify that after wiring, the facade's public methods delegate to
self.calculator instead of self._legacy_impl. This is essential for proving
the component is actually being used.

Test Strategy:
- Mock the calculator component
- Call facade methods
- Verify calculator methods are called (not legacy)
- Verify _legacy_impl methods are NOT called
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from ml.features.common.feature_calculator import FeatureCalculator
from ml.features.engineering import FeatureConfig, IndicatorManager
from ml.features.facade import FeatureEngineer


if TYPE_CHECKING:
    pass


pytestmark = pytest.mark.unit


# ==================== calculate_features_batch Delegation Tests ====================


class TestCalculateFeaturesBlockDelegation:
    """Verify calculate_features_batch delegates to self.calculator."""

    def test_calculate_features_batch_calls_calculator(
        self,
        feature_config: FeatureConfig,
        sample_ohlcv_dataframe: pd.DataFrame,
        mock_calculator_result_batch: tuple[pd.DataFrame, None],
    ) -> None:
        """
        Verify facade.calculate_features_batch() calls self.calculator._calculate_features_batch().

        After wiring, the facade should delegate to the calculator component,
        NOT to the legacy implementation.
        """
        facade = FeatureEngineer(feature_config)

        # Mock the calculator's _calculate_features_batch method
        with patch.object(
            facade.calculator,
            "_calculate_features_batch",
            return_value=mock_calculator_result_batch,
        ) as mock_calc:
            result = facade.calculate_features_batch(sample_ohlcv_dataframe)

            # Verify calculator was called
            mock_calc.assert_called_once()

            # Verify the call arguments
            call_args = mock_calc.call_args
            assert call_args is not None
            # First positional arg should be the dataframe
            pd.testing.assert_frame_equal(
                call_args.kwargs.get("data", call_args.args[0] if call_args.args else None),
                sample_ohlcv_dataframe,
            )

    def test_calculate_features_batch_does_not_call_legacy(
        self,
        feature_config: FeatureConfig,
        sample_ohlcv_dataframe: pd.DataFrame,
    ) -> None:
        """
        Verify facade.calculate_features_batch() does NOT call _legacy_impl.

        This is the inverse test - proving the legacy path is NOT used.
        """
        facade = FeatureEngineer(feature_config)

        facade._legacy_impl = MagicMock()
        facade.calculate_features_batch(sample_ohlcv_dataframe)

        facade._legacy_impl.calculate_features_batch.assert_not_called()


# ==================== calculate_features_online Delegation Tests ====================


class TestCalculateFeaturesOnlineDelegation:
    """Verify calculate_features_online delegates to self.calculator (HOT PATH)."""

    def test_calculate_features_online_calls_calculator(
        self,
        feature_config: FeatureConfig,
        current_bar_dict: dict[str, float],
        indicator_manager_with_history: IndicatorManager,
        mock_calculator_result_online: np.ndarray,
    ) -> None:
        """
        Verify facade.calculate_features_online() calls self.calculator._calculate_features_online().

        This is the HOT PATH test - critical for performance.
        """
        facade = FeatureEngineer(feature_config)

        with patch.object(
            facade.calculator,
            "_calculate_features_online",
            return_value=mock_calculator_result_online,
        ) as mock_calc:
            result = facade.calculate_features_online(
                current_bar_dict,
                indicator_manager_with_history,
            )

            # Verify calculator was called
            mock_calc.assert_called_once()

            # Verify call arguments
            call_args = mock_calc.call_args
            assert call_args is not None

    def test_calculate_features_online_does_not_call_legacy(
        self,
        feature_config: FeatureConfig,
        current_bar_dict: dict[str, float],
        indicator_manager_with_history: IndicatorManager,
    ) -> None:
        """
        Verify facade.calculate_features_online() does NOT call _legacy_impl.
        """
        facade = FeatureEngineer(feature_config)

        facade._legacy_impl = MagicMock()
        facade.calculate_features_online(
            current_bar_dict,
            indicator_manager_with_history,
        )

        facade._legacy_impl.calculate_features_online.assert_not_called()


# ==================== compute_features Delegation Tests ====================


class TestComputeFeaturesDelegation:
    """Verify compute_features delegates to self.calculator."""

    def test_compute_features_prefers_legacy_shim(
        self,
        feature_config: FeatureConfig,
        test_data_factory,
    ) -> None:
        """
        Verify compute_features uses the legacy compatibility shim when available.
        """
        facade = FeatureEngineer(feature_config)
        bars = test_data_factory.bars(n=10)

        mock_result = {"feature_1": 1.0, "feature_2": 2.0}
        facade._legacy_impl = MagicMock()
        facade._legacy_impl.compute_features.return_value = mock_result
        facade.calculator.compute_features = MagicMock()  # defensive: should not be called

        result = facade.compute_features(bars)

        facade._legacy_impl.compute_features.assert_called_once_with(bars)
        facade.calculator.compute_features.assert_not_called()
        assert result == mock_result

    def test_compute_features_initializes_legacy_when_missing(
        self,
        feature_config: FeatureConfig,
        test_data_factory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Verify compute_features lazily instantiates the legacy shim when absent.
        """
        bars = test_data_factory.bars(n=5)

        sentinel = {"feature": 1.0}

        class _LegacyShim:
            def __init__(self, *args: object, **kwargs: object) -> None: ...

            def compute_features(self, inner_bars: list[object]) -> dict[str, float]:
                assert inner_bars == bars
                return sentinel

        monkeypatch.setattr("ml.features.facade.LegacyFeatureEngineer", _LegacyShim)

        facade = FeatureEngineer(feature_config)
        facade._legacy_impl = None

        result = facade.compute_features(bars)

        assert result == sentinel


# ==================== Unified calculate_features Delegation Tests ====================


class TestCalculateFeaturesUnifiedDelegation:
    """Verify calculate_features unified method delegates to calculator."""

    def test_calculate_features_batch_mode_delegates_to_calculator(
        self,
        feature_config: FeatureConfig,
        sample_ohlcv_dataframe: pd.DataFrame,
        mock_calculator_result_batch: tuple[pd.DataFrame, None],
    ) -> None:
        """
        Verify calculate_features(mode='batch') calls calculator._calculate_features_batch().
        """
        facade = FeatureEngineer(feature_config)

        with patch.object(
            facade.calculator,
            "_calculate_features_batch",
            return_value=mock_calculator_result_batch,
        ) as mock_calc:
            result = facade.calculate_features(
                sample_ohlcv_dataframe,
                mode="batch",
            )

            mock_calc.assert_called_once()

    def test_calculate_features_online_mode_delegates_to_calculator(
        self,
        feature_config: FeatureConfig,
        current_bar_dict: dict[str, float],
        indicator_manager_with_history: IndicatorManager,
        mock_calculator_result_online: np.ndarray,
    ) -> None:
        """
        Verify calculate_features(mode='online') calls calculator._calculate_features_online().
        """
        facade = FeatureEngineer(feature_config)

        with patch.object(
            facade.calculator,
            "_calculate_features_online",
            return_value=mock_calculator_result_online,
        ) as mock_calc:
            result = facade.calculate_features(
                current_bar_dict,
                mode="online",
                indicator_manager=indicator_manager_with_history,
            )

            mock_calc.assert_called_once()

    def test_calculate_features_unified_does_not_call_legacy(
        self,
        feature_config: FeatureConfig,
        sample_ohlcv_dataframe: pd.DataFrame,
    ) -> None:
        """
        Verify calculate_features() does NOT call _legacy_impl.calculate_features().
        """
        facade = FeatureEngineer(feature_config)

        facade._legacy_impl = MagicMock()
        facade.calculate_features(sample_ohlcv_dataframe, mode="batch")

        facade._legacy_impl.calculate_features.assert_not_called()


# ==================== Feature Buffer Access Tests ====================


class TestFeatureBufferAccess:
    """Verify feature_buffer property accesses calculator's buffer."""

    def test_feature_buffer_property_returns_calculator_buffer(
        self,
        feature_config: FeatureConfig,
    ) -> None:
        """
        Verify facade.feature_buffer returns self.calculator.feature_buffer.

        After wiring, the facade's feature_buffer property should return
        the calculator's buffer (for zero-allocation hot path).
        """
        facade = FeatureEngineer(feature_config)

        # The feature_buffer should be the same object as calculator's buffer
        # (Currently it returns _legacy_impl.feature_buffer - this should change)
        assert facade.feature_buffer is facade.calculator.feature_buffer, (
            "feature_buffer should return calculator's buffer, not legacy"
        )

    def test_feature_buffer_is_numpy_array(
        self,
        feature_config: FeatureConfig,
    ) -> None:
        """Verify feature_buffer is a numpy array with correct dtype."""
        facade = FeatureEngineer(feature_config)

        buffer = facade.feature_buffer

        assert isinstance(buffer, np.ndarray)
        assert buffer.dtype == np.float32


# ==================== Summary ====================

"""
Delegation Test Coverage Summary:
- calculate_features_batch delegation: 2 tests
- calculate_features_online delegation: 2 tests
- compute_features delegation: 2 tests
- calculate_features unified delegation: 3 tests
- feature_buffer access: 2 tests

Total: 11 delegation tests

These tests prove that after wiring, the facade uses self.calculator
instead of self._legacy_impl. They use mocking to verify call paths.
"""
