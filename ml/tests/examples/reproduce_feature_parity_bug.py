#!/usr/bin/env python3
"""
Reproduce the critical feature parity bug.
"""

import sys

sys.path.insert(0, "/home/nate/projects/nautilus_trader")

import numpy as np
import polars as pl
from ml.features.engineering import FeatureConfig, FeatureEngineer


def reproduce_feature_parity_bug():
    """
    Reproduce the feature count mismatch between batch and online modes.
    """
    # Create mock data
    n_bars = 100
    base_price = 1.1000
    rng = np.random.default_rng(42)

    bars_data = []
    for i in range(n_bars):
        price_var = rng.uniform(-0.001, 0.001)
        close_price = base_price + price_var
        high_price = close_price + abs(rng.uniform(0, 0.002))
        low_price = close_price - abs(rng.uniform(0, 0.002))
        volume = 1000000 + rng.integers(-100000, 100000)

        bars_data.append(
            {
                "close": close_price,
                "high": high_price,
                "low": low_price,
                "volume": float(volume),
                "ts_event": i * 60_000_000_000,  # 1 minute intervals in nanoseconds
            },
        )

    bars_df = pl.DataFrame(bars_data)

    print("=== Testing Default Configuration (L1 only) ===")
    # Default config (no microstructure/trade_flow)
    default_config = FeatureConfig()
    test_config_features(default_config, bars_df, bars_data, "Default Config")

    print("\n=== Testing with Microstructure Enabled (L1+L2) ===")
    # Config with microstructure enabled
    micro_config = FeatureConfig(include_microstructure=True)
    test_config_features(micro_config, bars_df, bars_data, "Microstructure Config")

    print("\n=== Testing with Both Microstructure and Trade Flow Enabled (L1+L2+L3) ===")
    # Config with both microstructure and trade flow enabled
    full_config = FeatureConfig(include_microstructure=True, include_trade_flow=True)
    test_config_features(full_config, bars_df, bars_data, "Full Config")


def test_config_features(config, bars_df, bars_data, config_name):
    """
    Test feature computation for a specific configuration.
    """
    print(f"\nTesting {config_name}:")
    print(f"  include_microstructure: {config.include_microstructure}")
    print(f"  include_trade_flow: {config.include_trade_flow}")

    # Initialize engineers
    batch_engineer = FeatureEngineer(config)
    online_engineer = FeatureEngineer(config)

    # Get feature names
    feature_names = config.get_feature_names()
    print(f"  Expected feature names ({len(feature_names)}): {feature_names}")

    try:
        # Compute batch features
        print("  Computing batch features...")
        batch_features, _ = batch_engineer.calculate_features_batch(bars_df)

        if hasattr(batch_features, "to_numpy"):
            batch_features_array = batch_features.to_numpy()
        else:
            batch_features_array = np.array(batch_features)

        batch_feature_count = (
            batch_features_array.shape[1]
            if batch_features_array.ndim > 1
            else len(batch_features_array)
        )
        print(f"  Batch features computed: {batch_feature_count}")

        # Compute online features for last few bars to warm up indicators
        print("  Computing online features...")
        online_features = []
        for i, bar_data in enumerate(bars_data):
            features = online_engineer.calculate_features_online(
                close_price=bar_data["close"],
                high_price=bar_data["high"],
                low_price=bar_data["low"],
                volume=bar_data["volume"],
            )
            if i >= len(bars_data) - 5:  # Keep last 5 for comparison
                online_features.append(features.copy())

        if online_features:
            online_feature_count = len(online_features[0])
            print(f"  Online features computed: {online_feature_count}")

            # Check for mismatch
            if batch_feature_count != online_feature_count:
                print(f"  ❌ FEATURE COUNT MISMATCH!")
                print(f"     Batch: {batch_feature_count} features")
                print(f"     Online: {online_feature_count} features")
                print(f"     Difference: {batch_feature_count - online_feature_count}")
                return False
            else:
                print(f"  ✅ Feature counts match: {batch_feature_count}")
                return True
        else:
            print("  ⚠️  No online features computed")
            return False

    except Exception as e:
        print(f"  ❌ Error during feature computation: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Reproducing Feature Parity Bug")
    print("=" * 50)

    success = reproduce_feature_parity_bug()

    print("\n" + "=" * 50)
    if success:
        print("All configurations passed!")
    else:
        print("Feature parity bug reproduced successfully!")
        sys.exit(1)
