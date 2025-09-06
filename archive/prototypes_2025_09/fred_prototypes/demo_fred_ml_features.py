#!/usr/bin/env python3
"""
Demonstration of FRED economic data integration with ML pipeline.

Shows how to use the updated FRED data for feature engineering and trading signals.

"""
import warnings

import pandas as pd
import polars as pl


warnings.filterwarnings("ignore")


def load_fred_ml_data():
    """
    Load FRED data in ML pipeline format.
    """
    try:
        df = pl.read_parquet("data/fred/fred_indicators_ml_format.parquet")
        print(
            f"📊 Loaded FRED ML data: {df.shape[0]} rows, {df['series_id'].n_unique()} indicators"
        )
        return df
    except Exception as e:
        print(f"❌ Error loading FRED data: {e}")
        return None


def create_regime_features(df):
    """
    Create market regime features from FRED data.
    """
    print("\n🎯 Creating Market Regime Features")
    print("=" * 40)

    # Pivot to wide format for easier feature engineering
    wide_df = df.pivot(
        index="timestamp",
        columns="series_id",
        values="value",
    ).sort("timestamp")

    # Convert to pandas for easier manipulation
    wide_pd = wide_df.to_pandas()
    wide_pd.set_index("timestamp", inplace=True)

    features = pd.DataFrame(index=wide_pd.index)

    # 1. Volatility Regime (VIX-based)
    if "VIXCLS" in wide_pd.columns:
        vix = wide_pd["VIXCLS"]
        features["vol_regime_low"] = (vix < 15).astype(int)  # Low vol
        features["vol_regime_normal"] = ((vix >= 15) & (vix < 25)).astype(int)  # Normal
        features["vol_regime_high"] = (vix >= 25).astype(int)  # High vol
        features["vix_zscore"] = (vix - vix.rolling(20).mean()) / vix.rolling(20).std()
        print(f"✅ VIX regime features: Current VIX = {vix.iloc[-1]:.1f}")

    # 2. Interest Rate Environment
    if "DGS10" in wide_pd.columns:
        rates_10y = wide_pd["DGS10"]
        features["rate_regime_low"] = (rates_10y < 2.0).astype(int)  # Low rates
        features["rate_regime_normal"] = ((rates_10y >= 2.0) & (rates_10y < 4.0)).astype(int)
        features["rate_regime_high"] = (rates_10y >= 4.0).astype(int)  # High rates
        print(f"✅ Rate regime features: Current 10Y = {rates_10y.iloc[-1]:.2f}%")

    # 3. Yield Curve Shape
    if all(col in wide_pd.columns for col in ["DGS2", "DGS10", "DGS30"]):
        features["yield_slope"] = wide_pd["DGS10"] - wide_pd["DGS2"]  # 10Y-2Y slope
        features["yield_curvature"] = 2 * wide_pd["DGS10"] - wide_pd["DGS2"] - wide_pd["DGS30"]
        features["yield_level"] = (wide_pd["DGS2"] + wide_pd["DGS10"] + wide_pd["DGS30"]) / 3
        features["yield_inversion"] = (features["yield_slope"] < 0).astype(int)
        print(f"✅ Yield curve features: Current slope = {features['yield_slope'].iloc[-1]:.2f}%")

    # 4. Credit Risk Environment
    if "BAMLH0A0HYM2" in wide_pd.columns:
        hy_spread = wide_pd["BAMLH0A0HYM2"]
        features["credit_stress"] = (hy_spread > hy_spread.rolling(60).mean() + 1.0).astype(int)
        features["hy_spread_zscore"] = (
            hy_spread - hy_spread.rolling(60).mean()
        ) / hy_spread.rolling(60).std()
        print(f"✅ Credit features: Current HY spread = {hy_spread.iloc[-1]:.2f}%")

    # 5. Dollar Strength
    if "DTWEXBGS" in wide_pd.columns:
        dxy = wide_pd["DTWEXBGS"]
        features["dollar_strong"] = (dxy > dxy.rolling(20).mean()).astype(int)
        features["dollar_momentum"] = dxy.pct_change(5)  # 5-day momentum
        print(f"✅ Dollar features: Current DXY = {dxy.iloc[-1]:.1f}")

    print(f"\n📈 Created {len(features.columns)} regime features")
    return features


def generate_trading_signals(df, features):
    """
    Generate sample trading signals using FRED data.
    """
    print("\n⚡ Generating Trading Signals")
    print("=" * 35)

    signals = pd.DataFrame(index=features.index)

    # Signal 1: VIX Spike Mean Reversion
    if "vix_zscore" in features.columns:
        # Buy when VIX spikes above 2 standard deviations
        signals["vix_spike_buy"] = (features["vix_zscore"] > 2.0).astype(int)
        vix_signals = signals["vix_spike_buy"].sum()
        print(f"1. VIX Spike Signals: {vix_signals} buy opportunities")

    # Signal 2: Yield Curve Inversion Warning
    if "yield_inversion" in features.columns:
        # Signal when curve inverts (potential recession warning)
        signals["curve_inversion"] = features["yield_inversion"]
        inversion_days = signals["curve_inversion"].sum()
        print(f"2. Curve Inversion: {inversion_days} days inverted")

    # Signal 3: Credit Stress Alert
    if "credit_stress" in features.columns:
        signals["credit_stress_signal"] = features["credit_stress"]
        stress_days = signals["credit_stress_signal"].sum()
        print(f"3. Credit Stress: {stress_days} days of elevated stress")

    # Signal 4: Multi-Factor Risk-Off Signal
    risk_factors = []
    if "vol_regime_high" in features.columns:
        risk_factors.append(features["vol_regime_high"])
    if "credit_stress" in features.columns:
        risk_factors.append(features["credit_stress"])
    if "yield_inversion" in features.columns:
        risk_factors.append(features["yield_inversion"])

    if risk_factors:
        signals["risk_off"] = sum(risk_factors) >= 2  # 2+ risk factors active
        risk_off_days = signals["risk_off"].sum()
        print(f"4. Multi-Factor Risk-Off: {risk_off_days} days")

    # Current signal status
    print("\n📊 Current Signal Status:")
    for col in signals.columns:
        if not signals[col].empty:
            current = signals[col].iloc[-1]
            status = "🔴 ACTIVE" if current else "🟢 INACTIVE"
            print(f"   {col}: {status}")

    return signals


def create_feature_summary():
    """
    Create summary of available FRED features.
    """
    print("\n📋 FRED Feature Categories")
    print("=" * 30)

    categories = {
        "Volatility": ["VIXCLS - VIX Index"],
        "Interest Rates": [
            "DGS1 - 1-Year Treasury",
            "DGS2 - 2-Year Treasury",
            "DGS10 - 10-Year Treasury",
            "DGS30 - 30-Year Treasury",
            "SOFR - Secured Overnight Financing Rate",
            "MORTGAGE30US - 30-Year Mortgage Rate",
        ],
        "Credit Risk": [
            "BAMLH0A0HYM2 - High Yield Credit Spread",
            "BAMLC0A0CM - Investment Grade Spread",
        ],
        "Currency": [
            "DTWEXBGS - Trade-Weighted Dollar Index",
            "DEXUSEU - USD/EUR Exchange Rate",
        ],
    }

    for category, indicators in categories.items():
        print(f"\n{category}:")
        for indicator in indicators:
            print(f"  • {indicator}")

    print(f"\n✅ Total: {sum(len(v) for v in categories.values())} economic indicators available")


def demonstrate_integration():
    """
    Demonstrate full FRED-ML integration.
    """
    print("🏦 FRED Economic Data ML Integration Demo")
    print("=" * 50)

    # Load data
    df = load_fred_ml_data()
    if df is None:
        return

    # Create features
    features = create_regime_features(df)

    # Generate signals
    signals = generate_trading_signals(df, features)

    # Feature summary
    create_feature_summary()

    # Integration recommendations
    print("\n💡 Integration Recommendations")
    print("=" * 35)
    print("1. TFT Model Integration:")
    print("   - Add regime features as static covariates")
    print("   - Use yield curve features as known future inputs")
    print("   - Include VIX as volatility proxy")

    print("\n2. Trading Strategy Integration:")
    print("   - Use risk-off signals for position sizing")
    print("   - Apply VIX regime filters to entry conditions")
    print("   - Adjust leverage based on credit stress levels")

    print("\n3. Risk Management:")
    print("   - Monitor yield curve inversion for recession risk")
    print("   - Scale down during high volatility regimes")
    print("   - Hedge credit exposure during stress periods")

    # Save processed features for ML pipeline
    output_file = "data/fred/fred_ml_features_processed.parquet"
    combined_features = pd.concat([features, signals], axis=1)
    combined_features.reset_index().to_parquet(output_file)
    print(f"\n💾 Saved processed features: {output_file}")
    print(f"   {len(combined_features)} rows, {len(combined_features.columns)} features")

    return combined_features


if __name__ == "__main__":
    features = demonstrate_integration()
