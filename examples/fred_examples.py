#!/usr/bin/env python3
"""
FRED Economic Data Examples for Nautilus Trader ML Pipeline.

Combines functionality from:
- demo_fred_ml_features.py
- explore_fred_features.py

Shows practical examples of:
1. Loading and exploring FRED economic indicators
2. Creating ML features from economic data
3. Generating trading signals and regime indicators
4. Market regime analysis and trading applications

"""

import argparse
import warnings

import numpy as np
import pandas as pd
import polars as pl


warnings.filterwarnings("ignore")


def load_fred_data():
    """
    Load FRED data from available sources.
    """
    print("📊 Loading FRED Economic Data")
    print("=" * 40)

    # Try ML format first, then fall back to updated format
    ml_file = "data/fred/fred_indicators_ml_format.parquet"
    updated_file = "data/fred/fred_indicators_updated.parquet"
    original_file = "data/fred/fred_indicators.parquet"

    for file_path in [ml_file, updated_file, original_file]:
        try:
            if file_path == ml_file:
                df = pl.read_parquet(file_path)
                print(
                    f"✅ Loaded ML format: {df.shape[0]} rows, {df['series_id'].n_unique()} indicators",
                )
                return df, "ml_format"
            else:
                df = pd.read_parquet(file_path)
                print(f"✅ Loaded {file_path}: {len(df)} rows")
                return df, "wide_format"
        except Exception as e:
            print(f"⚠️  Could not load {file_path}: {e}")
            continue

    print("❌ No FRED data files found")
    return None, None


def explore_indicators(df, format_type):
    """
    Explore available economic indicators.
    """
    print("\n🔍 Economic Indicators Overview")
    print("=" * 40)

    if format_type == "ml_format":
        indicators = df["series_id"].unique().to_list()
        date_range = (df["timestamp"].min(), df["timestamp"].max())

        print(f"📅 Date Range: {date_range[0]} to {date_range[1]}")
        print(f"📊 Available Indicators ({len(indicators)}):")

        # Group by category
        categories = {
            "Interest Rates": [
                "DGS1",
                "DGS2",
                "DGS10",
                "DGS30",
                "SOFR",
                "MORTGAGE30US",
                "FEDFUNDS",
            ],
            "Volatility": ["VIXCLS"],
            "Credit Risk": ["BAMLH0A0HYM2", "BAMLC0A0CM"],
            "Currency": ["DTWEXBGS", "DEXUSEU"],
            "Economic": ["GDP", "GDPC1", "CPIAUCSL", "CPILFESL", "UNRATE", "PAYEMS", "UMCSENT"],
        }

        for category, series_list in categories.items():
            available = [s for s in series_list if s in indicators]
            if available:
                print(f"\n  {category}:")
                for series in available:
                    # Get latest value
                    latest = df.filter(pl.col("series_id") == series).sort("timestamp").tail(1)
                    if not latest.is_empty():
                        value = latest["value"][0]
                        date = latest["timestamp"][0]
                        print(f"    • {series}: {value:.2f} (as of {date.strftime('%Y-%m-%d')})")

    else:  # wide_format
        if "date" in df.columns:
            date_range = (df["date"].min(), df["date"].max())
            print(f"📅 Date Range: {date_range[0]} to {date_range[1]}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        indicators = [col for col in numeric_cols if col not in ["timestamp_ns"]]

        print(f"📊 Available Indicators ({len(indicators)}):")

        # Show latest values
        if len(df) > 0:
            latest_row = df.iloc[-1]
            for indicator in indicators:
                if pd.notna(latest_row[indicator]):
                    print(f"    • {indicator}: {latest_row[indicator]:.2f}")


def create_regime_features(df, format_type):
    """
    Create market regime features from FRED data.
    """
    print("\n🎯 Market Regime Feature Engineering")
    print("=" * 40)

    if format_type == "ml_format":
        # Convert to wide format for feature engineering
        wide_df = df.pivot_table(
            index="timestamp",
            columns="series_id",
            values="value",
        ).sort("timestamp")
        wide_pd = wide_df.to_pandas()
        wide_pd = wide_pd.set_index("timestamp")
    else:
        wide_pd = df.set_index("date") if "date" in df.columns else df

    features = pd.DataFrame(index=wide_pd.index)

    # 1. Volatility Regime (VIX-based)
    if "VIXCLS" in wide_pd.columns:
        vix = wide_pd["VIXCLS"].dropna()
        features["vol_regime_low"] = (vix < 15).astype(int)
        features["vol_regime_normal"] = ((vix >= 15) & (vix < 25)).astype(int)
        features["vol_regime_high"] = (vix >= 25).astype(int)
        features["vix_zscore"] = (vix - vix.rolling(20).mean()) / vix.rolling(20).std()
        print(f"✅ VIX Regime: Current = {vix.iloc[-1]:.1f}")

        # Historical regime distribution
        low_pct = features["vol_regime_low"].mean() * 100
        normal_pct = features["vol_regime_normal"].mean() * 100
        high_pct = features["vol_regime_high"].mean() * 100
        print(
            f"   Distribution: Low {low_pct:.1f}%, Normal {normal_pct:.1f}%, High {high_pct:.1f}%",
        )

    # 2. Interest Rate Environment
    if "DGS10" in wide_pd.columns:
        rates_10y = wide_pd["DGS10"].dropna()
        features["rate_regime_low"] = (rates_10y < 2.0).astype(int)
        features["rate_regime_normal"] = ((rates_10y >= 2.0) & (rates_10y < 4.0)).astype(int)
        features["rate_regime_high"] = (rates_10y >= 4.0).astype(int)
        print(f"✅ Rate Regime: Current 10Y = {rates_10y.iloc[-1]:.2f}%")

    # 3. Yield Curve Shape
    if all(col in wide_pd.columns for col in ["DGS2", "DGS10"]):
        features["yield_slope"] = wide_pd["DGS10"] - wide_pd["DGS2"]
        features["yield_inversion"] = (features["yield_slope"] < 0).astype(int)

        if "DGS30" in wide_pd.columns:
            features["yield_curvature"] = 2 * wide_pd["DGS10"] - wide_pd["DGS2"] - wide_pd["DGS30"]

        current_slope = (
            features["yield_slope"].dropna().iloc[-1]
            if not features["yield_slope"].dropna().empty
            else np.nan
        )
        inversion_days = features["yield_inversion"].sum()
        print(
            f"✅ Yield Curve: Current slope = {current_slope:.2f}%, Inversions = {inversion_days} days",
        )

    # 4. Credit Risk Environment
    if "BAMLH0A0HYM2" in wide_pd.columns:
        hy_spread = wide_pd["BAMLH0A0HYM2"].dropna()
        hy_ma = hy_spread.rolling(60).mean()
        features["credit_stress"] = (hy_spread > hy_ma + 1.0).astype(int)
        features["hy_spread_zscore"] = (hy_spread - hy_ma) / hy_spread.rolling(60).std()

        current_spread = hy_spread.iloc[-1] if not hy_spread.empty else np.nan
        stress_days = features["credit_stress"].sum()
        print(
            f"✅ Credit Risk: Current HY spread = {current_spread:.2f}%, Stress days = {stress_days}",
        )

    print(f"\n📈 Created {len(features.columns)} regime features")
    return features


def generate_trading_signals(features):
    """
    Generate trading signals from regime features.
    """
    print("\n⚡ Trading Signal Generation")
    print("=" * 35)

    signals = pd.DataFrame(index=features.index)

    # Signal 1: VIX Spike Mean Reversion
    if "vix_zscore" in features.columns:
        signals["vix_spike_buy"] = (features["vix_zscore"] > 2.0).astype(int)
        vix_signals = signals["vix_spike_buy"].sum()
        print(f"1. VIX Spike Signals: {vix_signals} buy opportunities")

    # Signal 2: Yield Curve Inversion Warning
    if "yield_inversion" in features.columns:
        signals["curve_inversion"] = features["yield_inversion"]
        inversion_days = signals["curve_inversion"].sum()
        print(f"2. Curve Inversion: {inversion_days} days inverted")

    # Signal 3: Credit Stress Alert
    if "credit_stress" in features.columns:
        signals["credit_stress_signal"] = features["credit_stress"]
        stress_days = signals["credit_stress_signal"].sum()
        print(f"3. Credit Stress: {stress_days} days of elevated stress")

    # Signal 4: Multi-Factor Risk-Off
    risk_factors = []
    for col in ["vol_regime_high", "credit_stress", "yield_inversion"]:
        if col in features.columns:
            risk_factors.append(features[col])

    if risk_factors:
        signals["risk_off"] = (sum(risk_factors) >= 2).astype(int)
        risk_off_days = signals["risk_off"].sum()
        print(f"4. Multi-Factor Risk-Off: {risk_off_days} days")

    # Current signal status
    print("\n📊 Current Signal Status:")
    for col in signals.columns:
        if not signals[col].empty and not signals[col].dropna().empty:
            current = signals[col].dropna().iloc[-1]
            status = "🔴 ACTIVE" if current else "🟢 INACTIVE"
            print(f"   {col}: {status}")

    return signals


def market_regime_analysis(features):
    """
    Analyze historical market regimes.
    """
    print("\n📈 Historical Market Regime Analysis")
    print("=" * 40)

    # Volatility regimes
    if all(
        col in features.columns
        for col in ["vol_regime_low", "vol_regime_normal", "vol_regime_high"]
    ):
        vol_regimes = features[["vol_regime_low", "vol_regime_normal", "vol_regime_high"]]
        regime_pcts = vol_regimes.mean() * 100

        print("Volatility Regime Distribution:")
        print(f"  📉 Low Vol (<15):    {regime_pcts['vol_regime_low']:.1f}%")
        print(f"  📊 Normal Vol (15-25): {regime_pcts['vol_regime_normal']:.1f}%")
        print(f"  📈 High Vol (>25):   {regime_pcts['vol_regime_high']:.1f}%")

    # Interest rate regimes
    if all(
        col in features.columns
        for col in ["rate_regime_low", "rate_regime_normal", "rate_regime_high"]
    ):
        rate_regimes = features[["rate_regime_low", "rate_regime_normal", "rate_regime_high"]]
        regime_pcts = rate_regimes.mean() * 100

        print("\nInterest Rate Regime Distribution:")
        print(f"  📉 Low Rates (<2%):   {regime_pcts['rate_regime_low']:.1f}%")
        print(f"  📊 Normal Rates (2-4%): {regime_pcts['rate_regime_normal']:.1f}%")
        print(f"  📈 High Rates (>4%):  {regime_pcts['rate_regime_high']:.1f}%")

    # Risk episodes
    risk_indicators = ["yield_inversion", "credit_stress", "vol_regime_high"]
    risk_data = features[[col for col in risk_indicators if col in features.columns]]

    if not risk_data.empty:
        risk_days = risk_data.sum()
        print("\nRisk Episode Frequency:")
        for indicator, days in risk_days.items():
            pct = (days / len(features)) * 100
            print(f"  {indicator}: {days} days ({pct:.1f}%)")


def integration_examples():
    """
    Show integration examples with ML pipeline.
    """
    print("\n💡 ML Pipeline Integration Examples")
    print("=" * 40)

    print("1. TFT Model Integration:")
    print("   ```python")
    print("   # Add regime features as static covariates")
    print("   static_features = ['vol_regime_low', 'rate_regime_high', 'yield_inversion']")
    print("   ")
    print("   # Use economic indicators as known future inputs")
    print("   known_future = ['DGS10', 'DGS2', 'VIXCLS']  # Available with delay")
    print("   ```")

    print("\n2. Trading Strategy Integration:")
    print("   ```python")
    print("   # Risk-based position sizing")
    print("   if features['risk_off'].iloc[-1]:")
    print("       position_size *= 0.5  # Reduce exposure")
    print("   ")
    print("   # VIX-based entry filters")
    print("   if features['vol_regime_high'].iloc[-1]:")
    print("       skip_momentum_trades = True")
    print("   ```")

    print("\n3. Risk Management:")
    print("   ```python")
    print("   # Dynamic leverage based on regimes")
    print("   if features['yield_inversion'].iloc[-1]:")
    print("       max_leverage = 1.0  # Conservative during inversions")
    print("   elif features['vol_regime_low'].iloc[-1]:")
    print("       max_leverage = 3.0  # Aggressive in low vol")
    print("   ```")


def main():
    """
    Run FRED examples with CLI interface.
    """
    parser = argparse.ArgumentParser(description="FRED Economic Data Examples")
    parser.add_argument(
        "--example",
        choices=["explore", "features", "signals", "regimes", "integration", "all"],
        default="all",
        help="Type of example to run",
    )

    args = parser.parse_args()

    print("🏦 FRED Economic Data Examples")
    print("=" * 50)

    # Load data
    df, format_type = load_fred_data()
    if df is None:
        print("❌ No FRED data available. Run simple_fred_updater.py first.")
        return

    if args.example in ["explore", "all"]:
        explore_indicators(df, format_type)

    if args.example in ["features", "signals", "regimes", "all"]:
        features = create_regime_features(df, format_type)

        if args.example in ["signals", "all"]:
            generate_trading_signals(features)

        if args.example in ["regimes", "all"]:
            market_regime_analysis(features)

    if args.example in ["integration", "all"]:
        integration_examples()

    print("\n" + "=" * 50)
    print("🎉 Examples complete!")

    print("\nUsage examples:")
    print("  python examples/fred_examples.py --example explore")
    print("  python examples/fred_examples.py --example features")
    print("  python examples/fred_examples.py --example signals")


if __name__ == "__main__":
    main()
