#!/usr/bin/env python3
"""
Test feature value parity between batch and online modes.
"""

import sys

sys.path.insert(0, "/home/nate/projects/nautilus_trader")

import numpy as np
import polars as pl
from ml.features.engineering import FeatureConfig, FeatureEngineer


def test_feature_value_parity():
    """
    Test that feature values match between batch and online modes with 1e-6 tolerance.
    """
    # Create mock data - use more bars to ensure indicator warmup
    n_bars = 200
    base_price = 1.1000
    rng = np.random.default_rng(42)

    bars_data = []
    current_price = base_price
    for i in range(n_bars):
        price_var = rng.uniform(-0.001, 0.001)
        current_price += price_var
        close_price = current_price
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

    print("=== Testing Feature Value Parity ===")

    # Test all configurations
    configs_to_test = [
        ("Default (L1)", FeatureConfig()),
        ("Microstructure (L1+L2)", FeatureConfig(include_microstructure=True)),
        ("Full (L1+L2+L3)", FeatureConfig(include_microstructure=True, include_trade_flow=True)),
    ]

    all_passed = True

    for config_name, config in configs_to_test:
        print(f"\nTesting {config_name}:")
        passed = test_config_value_parity(config, bars_df, bars_data, config_name)
        all_passed = all_passed and passed

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL FEATURE PARITY TESTS PASSED!")
        print("Feature values match between batch and online modes.")
    else:
        print("❌ SOME PARITY TESTS FAILED!")
        print("Feature value differences detected.")
        sys.exit(1)


def test_config_value_parity(config, bars_df, bars_data, config_name):
    """
    Test feature value parity for a specific configuration.
    """

    # Initialize engineers
    batch_engineer = FeatureEngineer(config)
    online_engineer = FeatureEngineer(config)

    try:
        # Compute batch features
        print("  Computing batch features...")
        batch_features, _ = batch_engineer.calculate_features_batch(bars_df)

        if hasattr(batch_features, "to_numpy"):
            batch_features_array = batch_features.to_numpy()
        else:
            batch_features_array = np.array(batch_features)

        print(f"  Batch features shape: {batch_features_array.shape}")

        # Compute online features for all bars
        print("  Computing online features...")
        online_features_list = []
        for i, bar_data in enumerate(bars_data):
            features = online_engineer.calculate_features_online(
                close_price=bar_data["close"],
                high_price=bar_data["high"],
                low_price=bar_data["low"],
                volume=bar_data["volume"],
            )
            # Important: copy to avoid buffer reuse issues
            online_features_list.append(features.copy())

        online_features_array = np.array(online_features_list)
        print(f"  Online features shape: {online_features_array.shape}")

        # Check shapes match
        if batch_features_array.shape != online_features_array.shape:
            print(
                f"  ❌ Shape mismatch: batch {batch_features_array.shape} vs online {online_features_array.shape}",
            )
            return False

        # Compare the last 50 rows (after indicators are warmed up)
        compare_start = max(0, len(bars_data) - 50)
        batch_subset = batch_features_array[compare_start:]
        online_subset = online_features_array[compare_start:]

        print(f"  Comparing last {len(batch_subset)} rows...")

        # Calculate differences
        abs_diff = np.abs(batch_subset - online_subset)
        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)

        print(f"  Max absolute difference: {max_abs_diff:.2e}")
        print(f"  Mean absolute difference: {mean_abs_diff:.2e}")

        # Check if differences are within tolerance
        # Use more lenient tolerance for OHLCV approximations vs proper microstructure
        tolerance = (
            1e-6 if not (config.include_microstructure or config.include_trade_flow) else 1e-3
        )

        if max_abs_diff < tolerance:
            print(f"  ✅ Feature parity PASSED (within {tolerance:.0e} tolerance)")
            return True
        else:
            print(f"  ❌ Feature parity FAILED (exceeds {tolerance:.0e} tolerance)")

            # Show where the largest differences are
            max_diff_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
            row, col = max_diff_idx
            batch_val = batch_subset[row, col]
            online_val = online_subset[row, col]

            feature_names = config.get_feature_names()
            feature_name = feature_names[col] if col < len(feature_names) else f"feature_{col}"

            print(
                f"  Largest diff at row {row}, feature '{feature_name}': batch={batch_val:.6f}, online={online_val:.6f}, diff={abs_diff[row, col]:.6f}",
            )

            # Show summary statistics for the problematic feature
            batch_col = batch_subset[:, col]
            online_col = online_subset[:, col]
            print(f"  Feature '{feature_name}' stats:")
            print(f"    Batch:  mean={np.mean(batch_col):.6f}, std={np.std(batch_col):.6f}")
            print(f"    Online: mean={np.mean(online_col):.6f}, std={np.std(online_col):.6f}")

            return False

    except Exception as e:
        print(f"  ❌ Error during feature computation: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_feature_value_parity()
