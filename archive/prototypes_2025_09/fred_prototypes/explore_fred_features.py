#!/usr/bin/env python3
"""
Explore FRED economic data for ML feature engineering.
"""
import warnings

import pandas as pd


warnings.filterwarnings("ignore")


def load_fred_data():
    """
    Load and clean FRED data.
    """
    df = pd.read_parquet("data/fred/fred_indicators.parquet")

    # Convert timestamp to datetime index
    if "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("date")
        df = df.drop(["timestamp", "timestamp_ns"], axis=1, errors="ignore")

    # Sort by date
    df = df.sort_index()

    return df


def analyze_regime_indicators(df):
    """
    Analyze market regime indicators.
    """
    print("🎯 Market Regime Analysis")
    print("=" * 40)

    # VIX regimes
    if "VIXCLS" in df.columns:
        vix = df["VIXCLS"].dropna()

        # Define regimes
        low_vol = vix < 15
        normal_vol = (vix >= 15) & (vix < 25)
        high_vol = vix >= 25

        print("VIX Regime Distribution:")
        print(f"  Low Vol (<15): {low_vol.sum()} days ({low_vol.mean()*100:.1f}%)")
        print(f"  Normal Vol (15-25): {normal_vol.sum()} days ({normal_vol.mean()*100:.1f}%)")
        print(f"  High Vol (>25): {high_vol.sum()} days ({high_vol.mean()*100:.1f}%)")

        print(f"\nCurrent VIX: {vix.iloc[-1]:.1f}")

    # Interest rate environment
    if "DGS10" in df.columns:
        rates_10y = df["DGS10"].dropna()

        # Define rate environments
        low_rates = rates_10y < 2.0
        normal_rates = (rates_10y >= 2.0) & (rates_10y < 4.0)
        high_rates = rates_10y >= 4.0

        print("\n10Y Treasury Rate Environment:")
        print(f"  Low Rates (<2%): {low_rates.sum()} days ({low_rates.mean()*100:.1f}%)")
        print(f"  Normal Rates (2-4%): {normal_rates.sum()} days ({normal_rates.mean()*100:.1f}%)")
        print(f"  High Rates (>4%): {high_rates.sum()} days ({high_rates.mean()*100:.1f}%)")

        print(f"\nCurrent 10Y Rate: {rates_10y.iloc[-1]:.2f}%")


def create_ml_features(df):
    """
    Create ML features from FRED data.
    """
    print("\n🔧 ML Feature Engineering")
    print("=" * 40)

    features = pd.DataFrame(index=df.index)

    # 1. Yield Curve Features
    if all(col in df.columns for col in ["DGS2", "DGS10", "DGS30"]):
        # Yield curve slope (10Y - 2Y)
        features["yield_curve_slope"] = df["DGS10"] - df["DGS2"]

        # Yield curve curvature (2*10Y - 2Y - 30Y)
        features["yield_curve_curvature"] = 2 * df["DGS10"] - df["DGS2"] - df["DGS30"]

        # Level (average of 2Y, 10Y, 30Y)
        features["yield_curve_level"] = (df["DGS2"] + df["DGS10"] + df["DGS30"]) / 3

        print("✅ Yield curve features: slope, curvature, level")

    # 2. Volatility Features
    if "VIXCLS" in df.columns:
        vix = df["VIXCLS"]
        features["vix"] = vix
        features["vix_ma_20"] = vix.rolling(20).mean()
        features["vix_zscore"] = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()

        print("✅ VIX features: level, MA, z-score")

    # 3. Credit Risk Features
    if "BAMLH0A0HYM2" in df.columns:
        hy_spread = df["BAMLH0A0HYM2"]
        features["high_yield_spread"] = hy_spread
        features["hy_spread_ma_20"] = hy_spread.rolling(20).mean()

        print("✅ Credit risk features: HY spread, moving average")

    # 4. Dollar Strength
    if "DTWEXBGS" in df.columns:
        dxy = df["DTWEXBGS"]
        features["dollar_index"] = dxy
        features["dollar_momentum_20"] = dxy.pct_change(20)

        print("✅ Currency features: DXY level, momentum")

    # 5. Economic Growth Proxies (monthly data, forward fill)
    economic_indicators = ["UNRATE", "CPIAUCSL", "PAYEMS"]
    for indicator in economic_indicators:
        if indicator in df.columns:
            # Forward fill monthly data to daily frequency
            features[f"{indicator.lower()}_ffill"] = df[indicator].fillna(method="ffill")

            # Year-over-year change
            features[f"{indicator.lower()}_yoy"] = df[indicator].pct_change(252)  # ~1 year

    print(
        f"✅ Economic indicators: {len([c for c in economic_indicators if c in df.columns])} series",
    )

    # 6. Regime Classifications
    if "VIXCLS" in df.columns:
        features["vol_regime"] = pd.cut(
            df["VIXCLS"],
            bins=[0, 15, 25, 100],
            labels=["low", "normal", "high"],
        )

    if "DGS10" in df.columns:
        features["rate_regime"] = pd.cut(
            df["DGS10"],
            bins=[0, 2, 4, 100],
            labels=["low", "normal", "high"],
        )

    print("✅ Regime features: volatility, interest rate regimes")

    # Feature summary
    print("\nFeature Summary:")
    print(f"  Total features created: {len(features.columns)}")
    print(
        f"  Date range: {features.index.min().strftime('%Y-%m-%d')} to {features.index.max().strftime('%Y-%m-%d')}",
    )
    print(
        f"  Non-null coverage: {(features.count().sum() / (len(features) * len(features.columns)) * 100):.1f}%",
    )

    return features


def demonstrate_trading_signals(df):
    """
    Show how FRED data can generate trading signals.
    """
    print("\n⚡ Trading Signal Examples")
    print("=" * 40)

    # Example 1: VIX Mean Reversion
    if "VIXCLS" in df.columns:
        vix = df["VIXCLS"].dropna()
        vix_ma = vix.rolling(20).mean()
        vix_std = vix.rolling(20).std()

        # VIX spike signal (buy when VIX > MA + 2*STD)
        vix_spike = vix > (vix_ma + 2 * vix_std)

        print("1. VIX Spike Signal:")
        print(f"   Signals in last year: {vix_spike.tail(252).sum()}")
        print(f"   Latest VIX: {vix.iloc[-1]:.1f} (MA: {vix_ma.iloc[-1]:.1f})")

    # Example 2: Yield Curve Inversion
    if all(col in df.columns for col in ["DGS2", "DGS10"]):
        curve_slope = df["DGS10"] - df["DGS2"]
        curve_inverted = curve_slope < 0

        print("\n2. Yield Curve Inversion:")
        print(f"   Days inverted in last year: {curve_inverted.tail(252).sum()}")
        print(f"   Current slope (10Y-2Y): {curve_slope.dropna().iloc[-1]:.2f}%")

    # Example 3: Credit Stress Signal
    if "BAMLH0A0HYM2" in df.columns:
        hy_spread = df["BAMLH0A0HYM2"].dropna()
        hy_ma = hy_spread.rolling(60).mean()

        # Credit stress when HY spread > MA + threshold
        credit_stress = hy_spread > (hy_ma + 1.0)  # 100bp above MA

        print("\n3. Credit Stress Signal:")
        print(f"   Stress days in last year: {credit_stress.tail(252).sum()}")
        print(f"   Current HY spread: {hy_spread.iloc[-1]:.2f}%")


def main():
    """
    Main analysis function.
    """
    print("🏦 FRED Economic Data Analysis for ML")
    print("=" * 50)

    # Load data
    df = load_fred_data()
    print(f"Loaded FRED data: {len(df)} observations, {len(df.columns)} series")
    print(
        f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
    )

    # Regime analysis
    analyze_regime_indicators(df)

    # Feature engineering
    features = create_ml_features(df)

    # Trading signals
    demonstrate_trading_signals(df)

    print("\n" + "=" * 50)
    print("💡 Next Steps:")
    print("1. Update FRED data: python ml/scripts/populate_fred_data.py --update-only")
    print("2. Integrate features into TFT training data")
    print("3. Use regime indicators for portfolio allocation")
    print("4. Combine with market data for enhanced ML models")

    return features


if __name__ == "__main__":
    features = main()
