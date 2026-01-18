"""
Component delegation tests for FeatureEngineer facade (Phase 1.1 - Category 14).

These tests verify that after wiring, the facade's public methods delegate to
self.calculator. This is essential for proving the component is actually
being used.

Test Strategy:
- Mock the calculator component
- Call facade methods
- Verify calculator methods are called
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ml.features.config import FeatureConfig
from ml.features.indicators import IndicatorManager
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

        After wiring, the facade should delegate to the calculator component.
        """
        facade = FeatureEngineer(feature_config)

        # Mock the calculator's _calculate_features_batch method
        with patch.object(
            facade.calculator,
            "_calculate_features_batch",
            return_value=mock_calculator_result_batch,
        ) as mock_calc:
            _ = facade.calculate_features_batch(sample_ohlcv_dataframe)

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
            _ = facade.calculate_features_online(
                current_bar_dict,
                indicator_manager_with_history,
            )

            # Verify calculator was called
            mock_calc.assert_called_once()

            # Verify call arguments
            call_args = mock_calc.call_args
            assert call_args is not None


# ==================== compute_features Delegation Tests ====================


class TestComputeFeaturesDelegation:
    """Verify compute_features delegates to self.calculator."""

    def test_compute_features_delegates_to_calculator(
        self,
        feature_config: FeatureConfig,
        test_data_factory,
    ) -> None:
        """
        Verify compute_features delegates to calculator.compute_features().
        """
        facade = FeatureEngineer(feature_config)
        bars = test_data_factory.bars(n=10)

        mock_result = {"feature_1": 1.0, "feature_2": 2.0}
        with patch.object(facade.calculator, "compute_features", return_value=mock_result) as mock_calc:
            result = facade.compute_features(bars)

        mock_calc.assert_called_once_with(bars)
        assert result == mock_result


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
            _ = facade.calculate_features(
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
            _ = facade.calculate_features(
                current_bar_dict,
                mode="online",
                indicator_manager=indicator_manager_with_history,
            )

            mock_calc.assert_called_once()


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
        assert facade.feature_buffer is facade.calculator.feature_buffer, (
            "feature_buffer should return calculator's buffer"
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
- calculate_features_batch delegation: 1 test
- calculate_features_online delegation: 1 test
- compute_features delegation: 1 test
- calculate_features unified delegation: 2 tests
- feature_buffer access: 2 tests

Total: 7 delegation tests

These tests prove that after wiring, the facade uses self.calculator
with the calculator as the single implementation. They use mocking to
verify call paths.
"""
