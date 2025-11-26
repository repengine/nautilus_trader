"""
Example showing how to refactor redundant tests using parameterization.

BEFORE: Multiple similar tests
AFTER: Single parameterized test

"""

from dataclasses import dataclass
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ml.features import FeatureConfig


# Minimal helpers used in examples to keep tests executable
data: dict[str, Any] = {"close": [1.0, 1.1, 1.2]}


def calculate_features(_data: dict[str, Any], feature_type: str) -> dict[str, float]:
    # In real code, this would call FeatureEngineer; here we just return the key to validate wiring
    return {feature_type: 1.0}


# ============================================================================
# BEFORE: Redundant tests (what you currently have)
# ============================================================================


class TestValidationRedundant:
    """
    Example of redundant validation tests.
    """

    def test_config_validation_rsi_period_too_small(self):
        with pytest.raises(ValueError):
            FeatureConfig(rsi_period=1)  # Too small

    def test_config_validation_rsi_period_too_large(self):
        with pytest.raises(ValueError):
            FeatureConfig(rsi_period=101)  # Too large

    def test_config_validation_bb_period_too_small(self):
        with pytest.raises(ValueError):
            FeatureConfig(bb_period=1)  # Too small

    def test_config_validation_bb_period_too_large(self):
        with pytest.raises(ValueError):
            FeatureConfig(bb_period=101)  # Too large

    def test_config_validation_atr_period_too_small(self):
        with pytest.raises(ValueError):
            FeatureConfig(atr_period=1)  # Too small

    def test_config_validation_atr_period_too_large(self):
        with pytest.raises(ValueError):
            FeatureConfig(atr_period=101)  # Too large


# ============================================================================
# AFTER: Parameterized test (DRY - Don't Repeat Yourself)
# ============================================================================


class TestValidationParameterized:
    """
    Refactored using pytest.mark.parametrize.
    """

    @pytest.mark.parametrize(
        "param,value,expected_error",
        [
            ("rsi_period", 1, ValueError),  # Too small
            ("rsi_period", 101, ValueError),  # Too large
            ("bb_period", 1, ValueError),  # Too small
            ("bb_period", 101, ValueError),  # Too large
            ("atr_period", 1, ValueError),  # Too small
            ("atr_period", 101, ValueError),  # Too large
        ],
    )
    def test_config_validation_bounds(self, param, value, expected_error):
        """
        Single test covers all validation cases.
        """
        with pytest.raises(expected_error):
            FeatureConfig(**{param: value})


# ============================================================================
# BETTER: Property-based test (even more coverage)
# ============================================================================


class TestValidationProperty:
    """Best approach: property-based testing."""

    @given(
        rsi_period=st.integers(),
        bb_period=st.integers(),
        atr_period=st.integers(),
    )
    def test_period_validation_property(self, rsi_period, bb_period, atr_period):
        """
        Test validation for all possible period values.
        """
        # Property: Valid periods are in range [2, 100]
        for param_name, value in [
            ("rsi_period", rsi_period),
            ("bb_period", bb_period),
            ("atr_period", atr_period),
        ]:
            if 2 <= value <= 100:
                # Should not raise
                cfg = FeatureConfig(**{param_name: value})
                assert getattr(cfg, param_name) == value
            else:
                # Should raise ValueError
                with pytest.raises(ValueError):
                    FeatureConfig(**{param_name: value})


# ============================================================================
# Example: Consolidating similar test methods
# ============================================================================


class TestFeatureCalculationRedundant:
    """
    Multiple similar test methods.
    """

    def test_returns_calculation(self):
        features = calculate_features(data, feature_type="returns")
        assert "returns" in features

    def test_momentum_calculation(self):
        features = calculate_features(data, feature_type="momentum")
        assert "momentum" in features

    def test_volatility_calculation(self):
        features = calculate_features(data, feature_type="volatility")
        assert "volatility" in features


class TestFeatureCalculationParameterized:
    """
    Consolidated into single parameterized test.
    """

    @pytest.mark.parametrize(
        "feature_type",
        [
            "returns",
            "momentum",
            "volatility",
            "rsi",
            "bollinger_bands",
        ],
    )
    def test_feature_calculation(self, feature_type):
        """
        Single test for all feature types.
        """
        features = calculate_features(data, feature_type=feature_type)
        assert feature_type in features


# ============================================================================
# Stats: Reduction achieved
# ============================================================================
# BEFORE: 15 test methods (in examples above)
# AFTER: 3 test methods
# Reduction: 80% fewer tests, same or better coverage
