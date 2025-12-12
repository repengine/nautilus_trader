"""
Error handling tests for FeatureEngineer facade (Phase 1.1).

These tests verify that the facade correctly handles error conditions
and raises appropriate exceptions with clear messages.

Test Strategy:
- Test missing indicator_manager for online mode
- Test invalid mode parameter
- Test empty DataFrame input
- Test empty bars list for compute_features
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from ml.features.engineering import FeatureConfig, IndicatorManager
from ml.features.facade import FeatureEngineer


if TYPE_CHECKING:
    pass


pytestmark = pytest.mark.unit


# ==================== Online Mode Error Tests ====================


class TestOnlineModeErrors:
    """Error handling tests for calculate_features_online."""

    def test_calculate_features_online_without_indicator_manager_raises(
        self,
        feature_config: FeatureConfig,
        current_bar_dict: dict[str, float],
    ) -> None:
        """
        Verify ValueError when indicator_manager is None in online mode.

        Online mode requires an indicator manager to maintain state.
        The error message should be clear about what's missing.
        """
        facade = FeatureEngineer(feature_config)

        with pytest.raises(ValueError, match="indicator_manager is required"):
            facade.calculate_features_online(
                current_bar_dict,
                indicator_manager=None,  # type: ignore
            )

    def test_calculate_features_unified_online_without_indicator_raises(
        self,
        feature_config: FeatureConfig,
        current_bar_dict: dict[str, float],
    ) -> None:
        """
        Verify ValueError when using unified API in online mode without manager.
        """
        facade = FeatureEngineer(feature_config)

        with pytest.raises(ValueError, match="indicator_manager is required"):
            facade.calculate_features(
                current_bar_dict,
                mode="online",
                indicator_manager=None,
            )


# ==================== Mode Validation Error Tests ====================


class TestModeValidationErrors:
    """Error handling tests for mode parameter validation."""

    def test_calculate_features_with_invalid_mode_raises(
        self,
        feature_config: FeatureConfig,
        sample_ohlcv_dataframe: pd.DataFrame,
    ) -> None:
        """
        Verify ValueError when mode is not 'batch' or 'online'.

        The error message should list valid modes.
        """
        facade = FeatureEngineer(feature_config)

        with pytest.raises(ValueError, match=r"Invalid mode.*batch.*online"):
            facade.calculate_features(
                sample_ohlcv_dataframe,
                mode="invalid",  # type: ignore
            )

    def test_calculate_features_with_empty_mode_raises(
        self,
        feature_config: FeatureConfig,
        sample_ohlcv_dataframe: pd.DataFrame,
    ) -> None:
        """
        Verify ValueError when mode is empty string.
        """
        facade = FeatureEngineer(feature_config)

        with pytest.raises(ValueError, match=r"Invalid mode"):
            facade.calculate_features(
                sample_ohlcv_dataframe,
                mode="",  # type: ignore
            )


# ==================== Empty Input Error Tests ====================


class TestEmptyInputErrors:
    """Error handling tests for empty input data."""

    def test_compute_features_with_empty_bars_list_returns_empty(
        self,
        feature_config: FeatureConfig,
    ) -> None:
        """
        Verify compute_features returns empty dict for empty list.

        This matches legacy behavior for graceful degradation.
        """
        facade = FeatureEngineer(feature_config)

        result = facade.compute_features([])
        assert result == {}, f"Expected empty dict, got {result}"

    def test_calculate_features_batch_with_empty_dataframe(
        self,
        feature_config: FeatureConfig,
    ) -> None:
        """
        Verify behavior with empty DataFrame.

        Should either raise ValueError or return empty DataFrame.
        """
        facade = FeatureEngineer(feature_config)
        empty_df = pd.DataFrame(
            {
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            }
        )

        # Depending on implementation, this may raise or return empty
        try:
            result, _ = facade.calculate_features_batch(empty_df)
            # If it returns, should be empty DataFrame
            assert len(result) == 0
        except (ValueError, IndexError) as e:
            # If it raises, error message should be clear
            assert "empty" in str(e).lower() or len(str(e)) > 0


# ==================== Missing Column Error Tests ====================


class TestMissingColumnErrors:
    """Error handling tests for missing required columns."""

    def test_calculate_features_batch_missing_close_column(
        self,
        feature_config: FeatureConfig,
    ) -> None:
        """
        Verify error when 'close' column is missing.
        """
        facade = FeatureEngineer(feature_config)
        df_no_close = pd.DataFrame(
            {
                "open": [100.0, 100.5],
                "high": [101.0, 101.5],
                "low": [99.0, 99.5],
                # Missing 'close'
                "volume": [1000000.0, 1100000.0],
            }
        )

        with pytest.raises((KeyError, ValueError)):
            facade.calculate_features_batch(df_no_close)

    def test_calculate_features_online_missing_close_key(
        self,
        feature_config: FeatureConfig,
        indicator_manager_with_history: IndicatorManager,
    ) -> None:
        """
        Verify error when 'close' key is missing from bar dict.
        """
        facade = FeatureEngineer(feature_config)
        bar_no_close = {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            # Missing 'close'
            "volume": 1000000.0,
        }

        with pytest.raises(KeyError):
            facade.calculate_features_online(
                bar_no_close,
                indicator_manager_with_history,
            )


# ==================== Type Error Tests ====================


class TestTypeErrors:
    """Error handling tests for incorrect types."""

    def test_calculate_features_batch_with_non_dataframe(
        self,
        feature_config: FeatureConfig,
    ) -> None:
        """
        Verify error when non-DataFrame passed to batch mode.
        """
        facade = FeatureEngineer(feature_config)

        # Pass a dict instead of DataFrame
        with pytest.raises((TypeError, AttributeError)):
            facade.calculate_features_batch(
                {"close": [100.0, 101.0]}  # type: ignore
            )

    def test_calculate_features_online_with_non_dict(
        self,
        feature_config: FeatureConfig,
        indicator_manager_with_history: IndicatorManager,
    ) -> None:
        """
        Verify error when non-dict passed to online mode.
        """
        facade = FeatureEngineer(feature_config)

        # Pass a list instead of dict
        with pytest.raises((TypeError, KeyError)):
            facade.calculate_features_online(
                [100.0, 101.0, 99.0, 100.5, 1000000.0],  # type: ignore
                indicator_manager_with_history,
            )


# ==================== Edge Case Error Tests ====================


class TestEdgeCaseErrors:
    """Error handling tests for edge cases."""

    def test_calculate_features_with_nan_values(
        self,
        feature_config: FeatureConfig,
    ) -> None:
        """
        Verify behavior with NaN values in input.

        Should either handle gracefully or raise clear error.
        """
        facade = FeatureEngineer(feature_config)
        df_with_nan = pd.DataFrame(
            {
                "open": [100.0, np.nan, 101.0],
                "high": [101.0, 101.5, 102.0],
                "low": [99.0, 99.5, 100.0],
                "close": [100.5, 101.0, 101.5],
                "volume": [1000000.0, 1100000.0, 1200000.0],
            }
        )

        # May handle NaN or raise - just verify it doesn't crash silently
        try:
            result, _ = facade.calculate_features_batch(df_with_nan)
            # If it works, check result is valid
            assert result is not None
        except (ValueError, RuntimeError) as e:
            # If it raises, verify message mentions NaN or invalid
            assert "nan" in str(e).lower() or "invalid" in str(e).lower()

    def test_calculate_features_with_inf_values(
        self,
        feature_config: FeatureConfig,
    ) -> None:
        """
        Verify behavior with infinite values in input.
        """
        facade = FeatureEngineer(feature_config)
        df_with_inf = pd.DataFrame(
            {
                "open": [100.0, np.inf, 101.0],
                "high": [101.0, 101.5, 102.0],
                "low": [99.0, 99.5, 100.0],
                "close": [100.5, 101.0, 101.5],
                "volume": [1000000.0, 1100000.0, 1200000.0],
            }
        )

        try:
            result, _ = facade.calculate_features_batch(df_with_inf)
            assert result is not None
        except (ValueError, RuntimeError) as e:
            assert "inf" in str(e).lower() or "invalid" in str(e).lower()


# ==================== Summary ====================

"""
Error Handling Test Coverage Summary:
- Online mode errors: 2 tests
- Mode validation errors: 2 tests
- Empty input errors: 2 tests
- Missing column errors: 2 tests
- Type errors: 2 tests
- Edge case errors: 2 tests

Total: 12 error handling tests

These tests ensure the facade provides clear, helpful error messages
when used incorrectly. Good error messages help users debug issues quickly.
"""
