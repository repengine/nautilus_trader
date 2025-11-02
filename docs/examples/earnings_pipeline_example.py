#!/usr/bin/env python3
"""
Example: Using Earnings Feature Transforms in ML Pipeline.

Demonstrates how to integrate earnings features into the Nautilus Trader ML
pipeline using the new TransformSpec classes.

This example shows:
1. Creating earnings transform specifications
2. Building a multi-instrument pipeline
3. Computing feature names for dataset building
4. Combining earnings with market features

Usage:
    python ml/tests/examples/earnings_pipeline_example.py
"""

from __future__ import annotations

from ml.features.earnings import (
    EarningsCalendarTransformSpec,
    EarningsGrowthTransformSpec,
    EarningsMomentumTransformSpec,
    EarningsSurpriseTransformSpec,
)


def example_1_single_instrument_earnings_pipeline() -> None:
    """Example 1: Create earnings pipeline for a single stock."""
    print("\n=== Example 1: Single Instrument Earnings Pipeline ===\n")

    # Define the ticker
    ticker = "AAPL"

    # Create earnings transform specifications
    surprise_spec = EarningsSurpriseTransformSpec(ticker=ticker)
    growth_spec = EarningsGrowthTransformSpec(ticker=ticker)
    momentum_spec = EarningsMomentumTransformSpec(ticker=ticker)
    calendar_spec = EarningsCalendarTransformSpec(ticker=ticker)

    # Compute feature names
    print(f"Earnings features for {ticker}:\n")
    print(f"Surprise features ({len(surprise_spec.compute_feature_names())} total):")
    for name in surprise_spec.compute_feature_names():
        print(f"  - {name}")

    print(f"\nGrowth features ({len(growth_spec.compute_feature_names())} total):")
    for name in growth_spec.compute_feature_names():
        print(f"  - {name}")

    print(f"\nMomentum features ({len(momentum_spec.compute_feature_names())} total):")
    for name in momentum_spec.compute_feature_names():
        print(f"  - {name}")

    print(f"\nCalendar features ({len(calendar_spec.compute_feature_names())} total):")
    for name in calendar_spec.compute_feature_names():
        print(f"  - {name}")

    # Total feature count
    all_specs = [surprise_spec, growth_spec, momentum_spec, calendar_spec]
    total_features = sum(len(spec.compute_feature_names()) for spec in all_specs)
    print(f"\nTotal earnings features: {total_features}")


def example_2_multi_instrument_pipeline() -> None:
    """Example 2: Create earnings pipeline for multiple stocks."""
    print("\n=== Example 2: Multi-Instrument Earnings Pipeline ===\n")

    # Define tickers for portfolio
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]

    # Create earnings specs for all instruments
    all_specs = []
    for ticker in tickers:
        all_specs.extend(
            [
                EarningsSurpriseTransformSpec(ticker=ticker),
                EarningsGrowthTransformSpec(ticker=ticker),
                EarningsMomentumTransformSpec(ticker=ticker),
                EarningsCalendarTransformSpec(ticker=ticker),
            ]
        )

    # Compute all feature names
    all_feature_names = []
    for spec in all_specs:
        all_feature_names.extend(spec.compute_feature_names())

    print(f"Portfolio tickers: {', '.join(tickers)}")
    print(f"Total earnings features: {len(all_feature_names)}")
    print(f"Features per instrument: {len(all_feature_names) // len(tickers)}")

    # Show first 5 features
    print("\nFirst 5 features:")
    for name in all_feature_names[:5]:
        print(f"  - {name}")

    print(f"\n... and {len(all_feature_names) - 5} more features")


def example_3_custom_lookback_configuration() -> None:
    """Example 3: Customize lookback periods for different strategies."""
    print("\n=== Example 3: Custom Lookback Configuration ===\n")

    ticker = "MSFT"

    # Short-term strategy: 4 quarters
    short_term_growth = EarningsGrowthTransformSpec(
        ticker=ticker,
        lookback_quarters=4,
    )

    # Long-term strategy: 8 quarters
    long_term_growth = EarningsGrowthTransformSpec(
        ticker=ticker,
        lookback_quarters=8,
    )

    # Very long-term: 12 quarters (3 years)
    very_long_term_growth = EarningsGrowthTransformSpec(
        ticker=ticker,
        lookback_quarters=12,
    )

    print(f"Strategy configurations for {ticker}:\n")
    print(f"Short-term (4Q): lookback={short_term_growth.lookback_quarters} quarters")
    print(f"  Features: {', '.join(short_term_growth.compute_feature_names())}")

    print(f"\nLong-term (8Q): lookback={long_term_growth.lookback_quarters} quarters")
    print(f"  Features: {', '.join(long_term_growth.compute_feature_names())}")

    print(f"\nVery long-term (12Q): lookback={very_long_term_growth.lookback_quarters} quarters")
    print(f"  Features: {', '.join(very_long_term_growth.compute_feature_names())}")

    print("\nNote: Feature names remain the same; lookback affects computation only.")


def example_4_combined_features_pipeline() -> None:
    """Example 4: Combine earnings features with market features."""
    print("\n=== Example 4: Combined Earnings + Market Features Pipeline ===\n")

    ticker = "AAPL"

    # Earnings features
    earnings_specs = [
        EarningsSurpriseTransformSpec(ticker=ticker),
        EarningsGrowthTransformSpec(ticker=ticker),
        EarningsMomentumTransformSpec(ticker=ticker),
        EarningsCalendarTransformSpec(ticker=ticker),
    ]

    # Compute earnings feature names
    earnings_features = []
    for spec in earnings_specs:
        earnings_features.extend(spec.compute_feature_names())

    # Simulated market features (would come from FeatureEngineer)
    market_features = [
        f"return_1_{ticker}",
        f"return_5_{ticker}",
        f"return_10_{ticker}",
        f"volatility_20_{ticker}",
        f"rsi_{ticker}",
        f"bb_width_{ticker}",
    ]

    # Simulated macro features
    macro_features = [
        "PAYEMS__value_real_time",  # Payroll employment
        "UNRATE__value_real_time",  # Unemployment rate
        "CPIAUCSL__value_real_time",  # CPI
    ]

    print(f"Feature categories for {ticker}:\n")
    print(f"Earnings features: {len(earnings_features)}")
    print(f"Market features: {len(market_features)}")
    print(f"Macro features: {len(macro_features)}")
    print(
        f"\nTotal features: {len(earnings_features) + len(market_features) + len(macro_features)}"
    )

    print("\nEarnings features:")
    for name in earnings_features:
        print(f"  - {name}")


def example_5_portfolio_factor_analysis() -> None:
    """Example 5: Setup for portfolio factor analysis with earnings."""
    print("\n=== Example 5: Portfolio Factor Analysis Setup ===\n")

    # Different sectors
    tech_stocks = ["AAPL", "MSFT", "GOOGL", "NVDA"]
    financial_stocks = ["JPM", "BAC", "GS", "C"]
    healthcare_stocks = ["JNJ", "UNH", "PFE", "ABBV"]

    all_stocks = tech_stocks + financial_stocks + healthcare_stocks

    # Create earnings surprise specs for all stocks
    surprise_specs = [EarningsSurpriseTransformSpec(ticker=ticker) for ticker in all_stocks]

    # Collect all earnings surprise features
    all_surprise_features = []
    for spec in surprise_specs:
        all_surprise_features.extend(spec.compute_feature_names())

    print(f"Portfolio composition:")
    print(f"  Tech: {len(tech_stocks)} stocks")
    print(f"  Financials: {len(financial_stocks)} stocks")
    print(f"  Healthcare: {len(healthcare_stocks)} stocks")
    print(f"  Total: {len(all_stocks)} stocks")

    print(f"\nEarnings surprise features: {len(all_surprise_features)}")
    print(f"Features per stock: {len(all_surprise_features) // len(all_stocks)}")

    # Show sample features from each sector
    print("\nSample features by sector:")
    print(f"  Tech (AAPL): {', '.join(surprise_specs[0].compute_feature_names())}")
    print(
        f"  Financial (JPM): {', '.join(surprise_specs[len(tech_stocks)].compute_feature_names())}"
    )
    print(
        f"  Healthcare (JNJ): {', '.join(surprise_specs[len(tech_stocks) + len(financial_stocks)].compute_feature_names())}"
    )


def example_6_serialization_and_configuration() -> None:
    """Example 6: Serialization and configuration management."""
    print("\n=== Example 6: Serialization and Configuration ===\n")

    import pickle

    # Create a spec
    original_spec = EarningsGrowthTransformSpec(ticker="TSLA", lookback_quarters=6)

    # Serialize
    serialized = pickle.dumps(original_spec)
    print(f"Original spec: {original_spec}")
    print(f"Serialized size: {len(serialized)} bytes")

    # Deserialize
    deserialized_spec = pickle.loads(serialized)
    print(f"Deserialized spec: {deserialized_spec}")

    # Verify equality
    assert original_spec == deserialized_spec
    assert original_spec.compute_feature_names() == deserialized_spec.compute_feature_names()
    print("\n✓ Serialization verified: specs are identical")

    # Configuration dictionary (for saving to JSON/YAML)
    config = {
        "ticker": deserialized_spec.ticker,
        "lookback_quarters": deserialized_spec.lookback_quarters,
        "name": deserialized_spec.name,
    }
    print(f"\nConfiguration dict: {config}")


def main() -> None:
    """
    Run all examples.
    """
    print("=" * 70)
    print("Earnings Feature Transforms - Example Usage")
    print("=" * 70)

    example_1_single_instrument_earnings_pipeline()
    example_2_multi_instrument_pipeline()
    example_3_custom_lookback_configuration()
    example_4_combined_features_pipeline()
    example_5_portfolio_factor_analysis()
    example_6_serialization_and_configuration()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
