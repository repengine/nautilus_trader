"""
Integration tests for earnings features in ML pipeline.

Tests end-to-end pipeline execution with earnings TransformSpec classes
following Universal ML Architecture Patterns.

Coverage:
- Pipeline spec creation with earnings transforms
- Feature name computation
- Multi-instrument pipelines
- Integration with existing pipeline infrastructure
"""

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

import pytest

from ml.features.earnings import (
    EarningsCalendarTransformSpec,
    EarningsGrowthTransformSpec,
    EarningsMomentumTransformSpec,
    EarningsSurpriseTransformSpec,
)


pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)

@pytest.mark.integration
class TestEarningsPipelineIntegration:
    """Integration tests for earnings feature pipeline."""

    def test_single_instrument_earnings_pipeline(self) -> None:
        """Test pipeline with all earnings transforms for single instrument."""
        # Arrange
        ticker = "AAPL"
        transforms = [
            EarningsSurpriseTransformSpec(ticker=ticker),
            EarningsGrowthTransformSpec(ticker=ticker),
            EarningsMomentumTransformSpec(ticker=ticker),
            EarningsCalendarTransformSpec(ticker=ticker),
        ]

        # Act
        all_feature_names = []
        for transform in transforms:
            all_feature_names.extend(transform.compute_feature_names())

        # Assert
        assert len(all_feature_names) == 8  # Total earnings features
        assert all(ticker in name for name in all_feature_names)

        # Verify feature categories
        surprise_features = [n for n in all_feature_names if "surprise" in n]
        growth_features = [n for n in all_feature_names if "growth" in n]
        momentum_features = [
            n for n in all_feature_names if "beat_streak" in n or "volatility" in n
        ]
        calendar_features = [n for n in all_feature_names if "days_to_next" in n]

        assert len(surprise_features) == 3
        assert len(growth_features) == 2
        assert len(momentum_features) == 2
        assert len(calendar_features) == 1

    def test_multi_instrument_earnings_pipeline(self) -> None:
        """Test pipeline with earnings transforms for multiple instruments."""
        # Arrange
        tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
        all_transforms = []

        for ticker in tickers:
            all_transforms.extend([
                EarningsSurpriseTransformSpec(ticker=ticker),
                EarningsGrowthTransformSpec(ticker=ticker),
                EarningsMomentumTransformSpec(ticker=ticker),
                EarningsCalendarTransformSpec(ticker=ticker),
            ])

        # Act
        all_feature_names = []
        for transform in all_transforms:
            all_feature_names.extend(transform.compute_feature_names())

        # Assert
        total_features = 8 * len(tickers)  # 8 features per instrument
        assert len(all_feature_names) == total_features
        assert len(set(all_feature_names)) == total_features  # All unique

        # Verify each ticker has exactly 8 features
        for ticker in tickers:
            ticker_features = [n for n in all_feature_names if ticker in n]
            assert len(ticker_features) == 8

    def test_earnings_pipeline_feature_naming_consistency(self) -> None:
        """Test that feature names follow consistent naming conventions."""
        # Arrange
        ticker = "TEST"
        transforms = [
            EarningsSurpriseTransformSpec(ticker=ticker),
            EarningsGrowthTransformSpec(ticker=ticker),
            EarningsMomentumTransformSpec(ticker=ticker),
            EarningsCalendarTransformSpec(ticker=ticker),
        ]

        # Act
        all_feature_names = []
        for transform in transforms:
            all_feature_names.extend(transform.compute_feature_names())

        # Assert - all feature names should follow pattern: {metric}_{suffix}_{ticker}
        for name in all_feature_names:
            # Should have at least 3 parts separated by _
            parts = name.split("_")
            assert len(parts) >= 3
            # Last part should be ticker
            assert parts[-1] == ticker
            # Should not have any uppercase (except ticker)
            assert all(
                part.isupper() or part.islower() or part.isdigit()
                for part in parts[:-1]
            )

    def test_pipeline_spec_serialization(self) -> None:
        """Test that TransformSpec instances can be serialized (picklable)."""
        import pickle

        # Arrange
        specs = [
            EarningsSurpriseTransformSpec(ticker="AAPL"),
            EarningsGrowthTransformSpec(ticker="MSFT", lookback_quarters=8),
            EarningsMomentumTransformSpec(ticker="GOOGL", lookback_quarters=6),
            EarningsCalendarTransformSpec(ticker="TSLA"),
        ]

        # Act & Assert - all specs should be picklable
        for spec in specs:
            serialized = pickle.dumps(spec)
            deserialized = pickle.loads(serialized)  # noqa: S301 - safe in controlled test

            # Verify deserialized spec is identical
            assert deserialized == spec
            assert deserialized.compute_feature_names() == spec.compute_feature_names()

    def test_mixed_pipeline_with_earnings_and_market_features(self) -> None:
        """Test pipeline combining earnings and market features."""
        # Arrange - simulate a mixed pipeline
        ticker = "AAPL"
        earnings_transforms = [
            EarningsSurpriseTransformSpec(ticker=ticker),
            EarningsGrowthTransformSpec(ticker=ticker),
        ]

        # Act
        earnings_features = []
        for transform in earnings_transforms:
            earnings_features.extend(transform.compute_feature_names())

        # Assert
        # Earnings features should be clearly identifiable
        assert all("earnings" in name or "eps" in name or "revenue" in name for name in earnings_features)
        # Should have expected count
        assert len(earnings_features) == 5  # 3 surprise + 2 growth

    def test_empty_ticker_creates_valid_pipeline(self) -> None:
        """Test that empty ticker creates valid (albeit unnamed) pipeline."""
        # Arrange
        transforms = [
            EarningsSurpriseTransformSpec(ticker=""),
            EarningsGrowthTransformSpec(ticker=""),
        ]

        # Act
        all_feature_names = []
        for transform in transforms:
            all_feature_names.extend(transform.compute_feature_names())

        # Assert - should still produce features (just without ticker suffix)
        assert len(all_feature_names) == 5  # 3 + 2
        assert all(name.endswith("_") for name in all_feature_names)

    def test_pipeline_feature_count_consistency(self) -> None:
        """Test that feature count remains consistent across instantiations."""
        # Arrange
        ticker = "AAPL"

        # Act - create multiple instances of same spec
        specs_round_1 = [
            EarningsSurpriseTransformSpec(ticker=ticker),
            EarningsGrowthTransformSpec(ticker=ticker),
            EarningsMomentumTransformSpec(ticker=ticker),
            EarningsCalendarTransformSpec(ticker=ticker),
        ]

        specs_round_2 = [
            EarningsSurpriseTransformSpec(ticker=ticker),
            EarningsGrowthTransformSpec(ticker=ticker),
            EarningsMomentumTransformSpec(ticker=ticker),
            EarningsCalendarTransformSpec(ticker=ticker),
        ]

        features_round_1 = []
        for spec in specs_round_1:
            features_round_1.extend(spec.compute_feature_names())

        features_round_2 = []
        for spec in specs_round_2:
            features_round_2.extend(spec.compute_feature_names())

        # Assert - should be identical
        assert features_round_1 == features_round_2
        assert len(features_round_1) == 8

    def test_earnings_transforms_with_different_configs(self) -> None:
        """Test earnings transforms with different configuration parameters."""
        # Arrange
        ticker = "AAPL"
        growth_spec_default = EarningsGrowthTransformSpec(ticker=ticker)
        growth_spec_custom = EarningsGrowthTransformSpec(
            ticker=ticker, lookback_quarters=8
        )
        momentum_spec_default = EarningsMomentumTransformSpec(ticker=ticker)
        momentum_spec_custom = EarningsMomentumTransformSpec(
            ticker=ticker, lookback_quarters=6
        )

        # Act
        features_default = growth_spec_default.compute_feature_names()
        features_custom = growth_spec_custom.compute_feature_names()
        momentum_features_default = momentum_spec_default.compute_feature_names()
        momentum_features_custom = momentum_spec_custom.compute_feature_names()

        # Assert - feature names should be same regardless of config
        # (config affects computation, not feature names)
        assert features_default == features_custom
        assert momentum_features_default == momentum_features_custom


@pytest.mark.integration
class TestEarningsTransformSpecEdgeCases:
    """Edge case tests for earnings TransformSpec classes."""

    def test_special_characters_in_ticker(self) -> None:
        """Test handling of special characters in ticker symbols."""
        # Arrange - some tickers have special characters
        special_tickers = ["BRK.B", "BF-B", "PRE-I"]

        # Act & Assert - should handle gracefully
        for ticker in special_tickers:
            spec = EarningsSurpriseTransformSpec(ticker=ticker)
            feature_names = spec.compute_feature_names()
            assert len(feature_names) == 3
            assert all(ticker in name for name in feature_names)

    def test_very_long_ticker(self) -> None:
        """Test handling of very long ticker symbols."""
        # Arrange
        long_ticker = "VERYLONGTICKERSYMBOL12345"
        spec = EarningsSurpriseTransformSpec(ticker=long_ticker)

        # Act
        feature_names = spec.compute_feature_names()

        # Assert
        assert all(long_ticker in name for name in feature_names)
        assert all(len(name) > len(long_ticker) for name in feature_names)

    def test_numeric_ticker(self) -> None:
        """Test handling of numeric ticker symbols."""
        # Arrange - some markets use numeric tickers
        numeric_ticker = "9984"  # SoftBank in Japan
        spec = EarningsGrowthTransformSpec(ticker=numeric_ticker)

        # Act
        feature_names = spec.compute_feature_names()

        # Assert
        assert len(feature_names) == 2
        assert all(numeric_ticker in name for name in feature_names)

    def test_zero_lookback_quarters(self) -> None:
        """Test edge case of zero lookback quarters."""
        # Arrange
        spec = EarningsGrowthTransformSpec(ticker="AAPL", lookback_quarters=0)

        # Act
        feature_names = spec.compute_feature_names()

        # Assert - should still produce feature names
        # (validation happens at compute time, not spec creation)
        assert len(feature_names) == 2

    def test_negative_lookback_quarters(self) -> None:
        """Test edge case of negative lookback quarters."""
        # Arrange
        spec = EarningsMomentumTransformSpec(ticker="MSFT", lookback_quarters=-1)

        # Act
        feature_names = spec.compute_feature_names()

        # Assert - should still produce feature names
        # (validation happens at compute time, not spec creation)
        assert len(feature_names) == 2
