#!/usr/bin/env python3

"""
Example: Extending the FeatureEngineer with custom features.

This example demonstrates how to:
1. Extend the base FeatureEngineer class with custom feature calculations
2. Maintain hot/cold path separation for performance
3. Add feature quality validation and monitoring
4. Ensure backward compatibility with existing code

IMPORTANT: The base FeatureEngineer in ml.features.engineering already includes
microstructure and trade flow features. This example shows how to add your own
custom features if needed.

"""

from __future__ import annotations

from typing import Any

import numpy as np

from ml._imports import HAS_POLARS
from ml.features import FeatureConfig
from ml.features import FeatureEngineer
from ml.common.safe_math import safe_divide


class CustomFeatureEngineer(FeatureEngineer):
    """
    Example of extending FeatureEngineer with custom domain-specific features.

    This example adds custom volatility and market regime features while
    maintaining the critical hot/cold path separation required for production.

    Custom Features Added
    ---------------------
    - Advanced volatility metrics (Parkinson, Garman-Klass)
    - Market regime detection features
    - Custom feature quality validation

    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        """
        Initialize custom feature engineer.

        Parameters
        ----------
        config : FeatureConfig, optional
            Configuration for feature engineering.

        """
        super().__init__(config)

        # Metrics for monitoring
        self.computation_times: dict[str, list[float]] = {}
        self.feature_quality_metrics: dict[str, dict[str, float]] = {}

        # Pre-allocate buffers for custom features
        self.volatility_buffer = np.zeros(3, dtype=np.float32)  # For 3 volatility features
        self.regime_buffer = np.zeros(2, dtype=np.float32)  # For 2 regime features

    def calculate_custom_volatility_features(
        self,
        bars_df: Any,  # pl.DataFrame in cold path
    ) -> dict[str, float]:
        """
        Calculate advanced volatility features (COLD PATH ONLY).

        This demonstrates how to add custom volatility calculations
        for batch processing during model training.

        Parameters
        ----------
        bars_df : pl.DataFrame
            DataFrame with OHLC data.

        Returns
        -------
        dict[str, float]
            Dictionary of volatility features.

        """
        features = {}

        if not HAS_POLARS or len(bars_df) < 20:
            return {
                "parkinson_volatility": 0.0,
                "garman_klass_volatility": 0.0,
                "volatility_ratio": 0.0,
            }

        # Parkinson volatility (using high-low range)
        log_hl = np.log(bars_df["high"] / bars_df["low"])
        parkinson_vol = np.sqrt(np.mean(log_hl**2) / (4 * np.log(2)))
        features["parkinson_volatility"] = float(parkinson_vol)

        # Garman-Klass volatility (using OHLC)
        log_hl = np.log(bars_df["high"] / bars_df["low"])
        log_co = np.log(bars_df["close"] / bars_df["open"])
        gk_vol = np.sqrt(np.mean(0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2))
        features["garman_klass_volatility"] = float(gk_vol) if not np.isnan(gk_vol) else 0.0

        # Volatility ratio (short-term vs long-term)
        if len(bars_df) >= 50:
            returns = bars_df["close"].pct_change().drop_nulls()
            if len(returns) >= 50:
                short_vol = returns.tail(10).std()
                long_vol = returns.tail(50).std()
                features["volatility_ratio"] = safe_divide(
                    float(short_vol) if short_vol is not None else 0.0,
                    float(long_vol) if long_vol is not None else 1.0,
                    1.0,
                )
            else:
                features["volatility_ratio"] = 1.0
        else:
            features["volatility_ratio"] = 1.0

        return features

    def calculate_features_online_with_custom(
        self,
        current_bar: dict[str, float],
        indicator_manager: Any,
        scaler: Any = None,
    ) -> np.ndarray:
        """
        Calculate features online including custom features (HOT PATH).

        This shows how to incorporate custom features in the hot path
        while maintaining low latency requirements.

        Parameters
        ----------
        current_bar : dict[str, float]
            Current OHLCV data.
        indicator_manager : IndicatorManager
            Indicator manager with state.
        scaler : Any, optional
            Pre-fitted scaler.

        Returns
        -------
        np.ndarray
            Feature array for prediction.

        """
        # Get base features (includes microstructure and trade flow if configured)
        base_features = self.calculate_features_online(
            current_bar,
            indicator_manager,
            scaler=None,  # We'll scale at the end
        )

        # Add simplified custom volatility features for hot path
        # In production, these would be pre-computed or use incremental updates
        high = current_bar["high"]
        low = current_bar["low"]
        close = current_bar["close"]
        open_ = current_bar["open"]

        # Simplified Parkinson volatility proxy
        if high > low > 0:
            self.volatility_buffer[0] = np.log(high / low)
        else:
            self.volatility_buffer[0] = 0.0

        # Simplified Garman-Klass proxy
        if high > low > 0 and close > 0 and open_ > 0:
            log_hl = np.log(high / low)
            log_co = np.log(close / open_)
            self.volatility_buffer[1] = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        else:
            self.volatility_buffer[1] = 0.0

        # Volatility ratio (would need historical buffer in production)
        self.volatility_buffer[2] = 1.0  # Placeholder

        # Combine all features
        n_features = len(base_features) + len(self.volatility_buffer)
        combined_features = np.zeros(n_features, dtype=np.float32)
        combined_features[: len(base_features)] = base_features
        combined_features[len(base_features) :] = self.volatility_buffer

        # Apply scaling if provided
        if scaler is not None:
            combined_features = scaler.transform(combined_features.reshape(1, -1))[0]

        return combined_features

    def validate_feature_quality(
        self,
        features_df: Any,  # pl.DataFrame or pd.DataFrame
    ) -> dict[str, dict[str, float]]:
        """
        Validate feature quality with custom metrics.

        Parameters
        ----------
        features_df : DataFrame
            Features to validate.

        Returns
        -------
        dict[str, dict[str, float]]
            Quality metrics per feature.

        """
        quality_metrics = super().validate_feature_quality(features_df)

        # Add custom validation logic
        for col in features_df.columns:
            if "volatility" in col.lower():
                # Check that volatility features are non-negative
                if col in quality_metrics:
                    negative_rate = (features_df[col] < 0).sum() / len(features_df)
                    quality_metrics[col]["negative_rate"] = float(negative_rate)

                    # Flag if too many negative values
                    if negative_rate > 0.01:  # More than 1% negative
                        quality_metrics[col]["warning"] = 1.0  # Flag as problematic

        self.feature_quality_metrics = quality_metrics
        return quality_metrics


# Example usage
def demonstrate_custom_feature_engineering() -> None:
    """
    Demonstrate how to use the custom feature engineer.
    """
    # Step 1: Create config (uses base FeatureConfig)
    config = FeatureConfig(
        # Standard features
        return_periods=[1, 5, 10, 20],
        rsi_period=14,
        bb_period=20,
        # Enable built-in advanced features
        include_microstructure=True,
        include_trade_flow=True,
    )

    # Step 2: Use custom engineer
    engineer = CustomFeatureEngineer(config)

    # Step 3: For batch processing (cold path)
    print("Example: Batch Feature Calculation (Cold Path)")
    print("===============================================")
    print("In training, you would:")
    print("1. Load historical data with Polars")
    print("2. Calculate all features including custom volatility")
    print("3. Validate feature quality")
    print("4. Save features to Parquet for model training")
    print()

    # Example feature names (would be generated from actual data)
    feature_names = engineer.get_feature_names()
    print(f"Base features: {len(feature_names)}")
    print(f"With custom volatility: {len(feature_names) + 3}")
    print()

    # Step 4: For online inference (hot path)
    print("Example: Online Feature Calculation (Hot Path)")
    print("==============================================")
    print("In live trading, you would:")
    print("1. Pre-allocate all buffers at startup")
    print("2. Update features incrementally on each bar")
    print("3. Use simplified calculations for custom features")
    print("4. Maintain <5ms latency for entire pipeline")
    print()

    # Example bar data
    current_bar = {
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.0,
        "volume": 1000000.0,
    }

    print(f"Example bar: {current_bar}")
    print()

    # Performance considerations
    print("Performance Best Practices:")
    print("===========================")
    print("✓ Use NumPy arrays exclusively in hot path")
    print("✓ Pre-allocate all buffers")
    print("✓ Avoid Pandas/Polars in hot path")
    print("✓ Use incremental updates instead of recalculation")
    print("✓ Profile and benchmark regularly")
    print("✓ Maintain feature parity between training and inference")


if __name__ == "__main__":
    demonstrate_custom_feature_engineering()
