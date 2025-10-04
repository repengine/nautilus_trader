"""
Unit tests for earnings feature transforms.

Tests TransformSpec classes for pipeline integration following Universal ML
Architecture Patterns.

Coverage:
- compute_feature_names() returns correct names
- Ticker parameterization works correctly
- TransformSpec instances are frozen (immutable)
- All 4 TransformSpec classes
"""

import pytest

from ml.features.earnings import (
    EarningsCalendarTransformSpec,
    EarningsGrowthTransformSpec,
    EarningsMomentumTransformSpec,
    EarningsSurpriseTransformSpec,
)


class TestEarningsSurpriseTransformSpec:
    """Test suite for EarningsSurpriseTransformSpec."""

    def test_compute_feature_names_returns_correct_names(self) -> None:
        """Test that compute_feature_names returns expected feature names."""
        # Arrange
        spec = EarningsSurpriseTransformSpec(ticker="AAPL")

        # Act
        feature_names = spec.compute_feature_names()

        # Assert
        assert len(feature_names) == 3
        assert feature_names == [
            "eps_surprise_q0_AAPL",
            "eps_surprise_pct_q0_AAPL",
            "revenue_surprise_pct_q0_AAPL",
        ]

    def test_ticker_parameterization(self) -> None:
        """Test that ticker parameter correctly parameterizes feature names."""
        # Arrange
        tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]

        for ticker in tickers:
            # Act
            spec = EarningsSurpriseTransformSpec(ticker=ticker)
            feature_names = spec.compute_feature_names()

            # Assert - all feature names should include ticker suffix
            assert all(name.endswith(f"_{ticker}") for name in feature_names)

    def test_empty_ticker_creates_valid_names(self) -> None:
        """Test that empty ticker string creates valid feature names."""
        # Arrange
        spec = EarningsSurpriseTransformSpec(ticker="")

        # Act
        feature_names = spec.compute_feature_names()

        # Assert - feature names should still be valid (just end with _)
        assert len(feature_names) == 3
        assert all(name.endswith("_") for name in feature_names)

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        # Arrange & Act
        spec = EarningsSurpriseTransformSpec()

        # Assert
        assert spec.name == "earnings_surprise"
        assert spec.ticker == ""
        assert spec.lookback_quarters == 1

    def test_immutability(self) -> None:
        """Test that TransformSpec is frozen (immutable)."""
        # Arrange
        spec = EarningsSurpriseTransformSpec(ticker="AAPL")

        # Act & Assert - attempting to modify should raise FrozenInstanceError
        with pytest.raises(Exception):  # dataclass FrozenInstanceError
            spec.ticker = "MSFT"  # type: ignore[misc]


class TestEarningsGrowthTransformSpec:
    """Test suite for EarningsGrowthTransformSpec."""

    def test_compute_feature_names_returns_correct_names(self) -> None:
        """Test that compute_feature_names returns expected feature names."""
        # Arrange
        spec = EarningsGrowthTransformSpec(ticker="MSFT")

        # Act
        feature_names = spec.compute_feature_names()

        # Assert
        assert len(feature_names) == 2
        assert feature_names == [
            "eps_growth_yoy_MSFT",
            "eps_growth_qoq_MSFT",
        ]

    def test_ticker_parameterization(self) -> None:
        """Test that ticker parameter correctly parameterizes feature names."""
        # Arrange
        tickers = ["JPM", "GS", "BAC"]

        for ticker in tickers:
            # Act
            spec = EarningsGrowthTransformSpec(ticker=ticker)
            feature_names = spec.compute_feature_names()

            # Assert
            assert all(name.endswith(f"_{ticker}") for name in feature_names)

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        # Arrange & Act
        spec = EarningsGrowthTransformSpec()

        # Assert
        assert spec.name == "earnings_growth"
        assert spec.ticker == ""
        assert spec.lookback_quarters == 5

    def test_custom_lookback_quarters(self) -> None:
        """Test that custom lookback_quarters can be set."""
        # Arrange & Act
        spec = EarningsGrowthTransformSpec(ticker="AAPL", lookback_quarters=8)

        # Assert
        assert spec.lookback_quarters == 8
        # Feature names should not change based on lookback
        assert len(spec.compute_feature_names()) == 2

    def test_immutability(self) -> None:
        """Test that TransformSpec is frozen (immutable)."""
        # Arrange
        spec = EarningsGrowthTransformSpec(ticker="MSFT")

        # Act & Assert
        with pytest.raises(Exception):
            spec.lookback_quarters = 10  # type: ignore[misc]


class TestEarningsMomentumTransformSpec:
    """Test suite for EarningsMomentumTransformSpec."""

    def test_compute_feature_names_returns_correct_names(self) -> None:
        """Test that compute_feature_names returns expected feature names."""
        # Arrange
        spec = EarningsMomentumTransformSpec(ticker="GOOGL")

        # Act
        feature_names = spec.compute_feature_names()

        # Assert
        assert len(feature_names) == 2
        assert feature_names == [
            "earnings_beat_streak_GOOGL",
            "eps_volatility_4q_GOOGL",
        ]

    def test_ticker_parameterization(self) -> None:
        """Test that ticker parameter correctly parameterizes feature names."""
        # Arrange
        tickers = ["AMZN", "FB", "NFLX"]

        for ticker in tickers:
            # Act
            spec = EarningsMomentumTransformSpec(ticker=ticker)
            feature_names = spec.compute_feature_names()

            # Assert
            assert all(name.endswith(f"_{ticker}") for name in feature_names)

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        # Arrange & Act
        spec = EarningsMomentumTransformSpec()

        # Assert
        assert spec.name == "earnings_momentum"
        assert spec.ticker == ""
        assert spec.lookback_quarters == 4

    def test_immutability(self) -> None:
        """Test that TransformSpec is frozen (immutable)."""
        # Arrange
        spec = EarningsMomentumTransformSpec(ticker="GOOGL")

        # Act & Assert
        with pytest.raises(Exception):
            spec.name = "modified_name"  # type: ignore[misc]


class TestEarningsCalendarTransformSpec:
    """Test suite for EarningsCalendarTransformSpec."""

    def test_compute_feature_names_returns_correct_names(self) -> None:
        """Test that compute_feature_names returns expected feature names."""
        # Arrange
        spec = EarningsCalendarTransformSpec(ticker="TSLA")

        # Act
        feature_names = spec.compute_feature_names()

        # Assert
        assert len(feature_names) == 1
        assert feature_names == ["days_to_next_earnings_TSLA"]

    def test_ticker_parameterization(self) -> None:
        """Test that ticker parameter correctly parameterizes feature names."""
        # Arrange
        tickers = ["NVDA", "AMD", "INTC"]

        for ticker in tickers:
            # Act
            spec = EarningsCalendarTransformSpec(ticker=ticker)
            feature_names = spec.compute_feature_names()

            # Assert
            assert feature_names[0] == f"days_to_next_earnings_{ticker}"

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        # Arrange & Act
        spec = EarningsCalendarTransformSpec()

        # Assert
        assert spec.name == "earnings_calendar"
        assert spec.ticker == ""

    def test_immutability(self) -> None:
        """Test that TransformSpec is frozen (immutable)."""
        # Arrange
        spec = EarningsCalendarTransformSpec(ticker="TSLA")

        # Act & Assert
        with pytest.raises(Exception):
            spec.ticker = "AAPL"  # type: ignore[misc]


class TestEarningsTransformSpecIntegration:
    """Integration tests for multiple TransformSpec classes."""

    def test_all_specs_produce_unique_feature_names_for_same_ticker(self) -> None:
        """Test that all 4 specs produce unique feature names for same ticker."""
        # Arrange
        ticker = "AAPL"
        surprise_spec = EarningsSurpriseTransformSpec(ticker=ticker)
        growth_spec = EarningsGrowthTransformSpec(ticker=ticker)
        momentum_spec = EarningsMomentumTransformSpec(ticker=ticker)
        calendar_spec = EarningsCalendarTransformSpec(ticker=ticker)

        # Act
        all_feature_names = (
            surprise_spec.compute_feature_names()
            + growth_spec.compute_feature_names()
            + momentum_spec.compute_feature_names()
            + calendar_spec.compute_feature_names()
        )

        # Assert - all feature names should be unique
        assert len(all_feature_names) == len(set(all_feature_names))
        assert len(all_feature_names) == 8  # 3 + 2 + 2 + 1

    def test_total_feature_count(self) -> None:
        """Test that total feature count matches expected."""
        # Arrange
        specs = [
            EarningsSurpriseTransformSpec(ticker="AAPL"),
            EarningsGrowthTransformSpec(ticker="AAPL"),
            EarningsMomentumTransformSpec(ticker="AAPL"),
            EarningsCalendarTransformSpec(ticker="AAPL"),
        ]

        # Act
        total_features = sum(len(spec.compute_feature_names()) for spec in specs)

        # Assert - matches the 8 core earnings features
        assert total_features == 8

    def test_multi_instrument_support(self) -> None:
        """Test multi-instrument pipeline support."""
        # Arrange
        tickers = ["AAPL", "MSFT", "GOOGL"]
        specs = []

        for ticker in tickers:
            specs.extend([
                EarningsSurpriseTransformSpec(ticker=ticker),
                EarningsGrowthTransformSpec(ticker=ticker),
                EarningsMomentumTransformSpec(ticker=ticker),
                EarningsCalendarTransformSpec(ticker=ticker),
            ])

        # Act
        all_feature_names = []
        for spec in specs:
            all_feature_names.extend(spec.compute_feature_names())

        # Assert
        # Should have 8 features per instrument × 3 instruments = 24 features
        assert len(all_feature_names) == 24
        # All names should be unique
        assert len(all_feature_names) == len(set(all_feature_names))
        # Each ticker should appear in exactly 8 feature names
        for ticker in tickers:
            ticker_features = [name for name in all_feature_names if ticker in name]
            assert len(ticker_features) == 8


# ============================================================================
# Performance and Compliance Tests
# ============================================================================


class TestTransformSpecCompliance:
    """Test compliance with Universal ML Architecture Patterns."""

    def test_all_specs_are_dataclasses(self) -> None:
        """Test that all TransformSpec classes are dataclasses."""
        # Arrange
        from dataclasses import is_dataclass

        specs = [
            EarningsSurpriseTransformSpec,
            EarningsGrowthTransformSpec,
            EarningsMomentumTransformSpec,
            EarningsCalendarTransformSpec,
        ]

        # Act & Assert
        for spec_class in specs:
            assert is_dataclass(spec_class)

    def test_all_specs_are_frozen(self) -> None:
        """Test that all TransformSpec instances are frozen (Pattern 2)."""
        # Arrange
        specs = [
            EarningsSurpriseTransformSpec(ticker="TEST"),
            EarningsGrowthTransformSpec(ticker="TEST"),
            EarningsMomentumTransformSpec(ticker="TEST"),
            EarningsCalendarTransformSpec(ticker="TEST"),
        ]

        # Act & Assert - all should raise when trying to modify
        for spec in specs:
            with pytest.raises(Exception):
                spec.name = "modified"  # type: ignore[misc]

    def test_all_specs_implement_compute_feature_names(self) -> None:
        """Test that all specs implement compute_feature_names method."""
        # Arrange
        specs = [
            EarningsSurpriseTransformSpec(ticker="TEST"),
            EarningsGrowthTransformSpec(ticker="TEST"),
            EarningsMomentumTransformSpec(ticker="TEST"),
            EarningsCalendarTransformSpec(ticker="TEST"),
        ]

        # Act & Assert
        for spec in specs:
            assert hasattr(spec, "compute_feature_names")
            assert callable(spec.compute_feature_names)
            feature_names = spec.compute_feature_names()
            assert isinstance(feature_names, list)
            assert all(isinstance(name, str) for name in feature_names)
            assert len(feature_names) > 0
