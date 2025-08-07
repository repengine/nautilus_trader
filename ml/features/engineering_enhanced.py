# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Enhanced feature engineering implementation example.

This module shows how to extend the current FeatureEngineer with advanced features
from FeatureEngineerV2 while maintaining backward compatibility and hot/cold path separation.

NOTE: This is an EXAMPLE implementation showing the migration approach.

"""

from __future__ import annotations

from typing import Any

import numpy as np

from ml._imports import HAS_POLARS
from ml._imports import pl
from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.features.engineering import safe_divide


class EnhancedFeatureEngineer(FeatureEngineer):
    """
    Enhanced feature engineer with microstructure and trade flow features.

    This class extends the base FeatureEngineer to add advanced features
    while maintaining perfect backward compatibility and hot/cold path separation.

    Key Enhancements
    ----------------
    - Microstructure features (bid-ask spreads, size imbalance)
    - Trade flow features (VWAP, trade intensity)
    - Feature quality metrics tracking
    - Computation time monitoring

    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        """
        Initialize enhanced feature engineer.

        Parameters
        ----------
        config : FeatureConfig, optional
            Configuration for feature engineering.

        """
        super().__init__(config)

        # Metrics for monitoring
        self.computation_times: dict[str, list[float]] = {}
        self.feature_quality_metrics: dict[str, dict[str, float]] = {}

        # Pre-allocate additional buffers if needed
        if self.config.include_microstructure:
            self.microstructure_buffer = np.zeros(7, dtype=np.float32)

        if self.config.include_trade_flow:
            self.trade_flow_buffer = np.zeros(4, dtype=np.float32)

    def calculate_microstructure_features(
        self,
        quotes_df: Any,  # pl.DataFrame in cold path
    ) -> dict[str, float]:
        """
        Calculate microstructure features from quote data (COLD PATH ONLY).

        This method is designed for batch processing during training.
        For real-time inference, these features should be pre-computed
        or calculated using a separate microstructure Actor.

        Parameters
        ----------
        quotes_df : pl.DataFrame
            DataFrame with bid/ask quotes including columns:
            - bid: Bid price
            - ask: Ask price
            - bid_size: Bid size
            - ask_size: Ask size

        Returns
        -------
        dict[str, float]
            Dictionary of microstructure features.

        """
        features = {}

        if not HAS_POLARS or len(quotes_df) < 10:
            # Return default values if not enough data
            return {
                "spread_mean": 0.0,
                "spread_std": 0.0,
                "spread_relative": 0.0,
                "size_imbalance_mean": 0.0,
                "size_imbalance_std": 0.0,
                "mid_return_std": 0.0,
                "mid_return_autocorr": 0.0,
            }

        # Calculate spreads
        spreads = quotes_df["ask"] - quotes_df["bid"]
        spread_mean = spreads.mean()
        spread_std = spreads.std()
        features["spread_mean"] = float(spread_mean) if spread_mean is not None else 0.0
        features["spread_std"] = float(spread_std) if spread_std is not None else 0.0

        # Relative spread
        mid_prices = (quotes_df["bid"] + quotes_df["ask"]) / 2
        mid_mean = mid_prices.mean()
        features["spread_relative"] = safe_divide(
            float(spread_mean) if spread_mean is not None else 0.0,
            float(mid_mean) if mid_mean is not None else 1.0,
            0.0,
        )

        # Size imbalance
        size_imbalance = (quotes_df["bid_size"] - quotes_df["ask_size"]) / (
            quotes_df["bid_size"] + quotes_df["ask_size"]
        )
        imb_mean = size_imbalance.mean()
        imb_std = size_imbalance.std()
        features["size_imbalance_mean"] = float(imb_mean) if imb_mean is not None else 0.0
        features["size_imbalance_std"] = float(imb_std) if imb_std is not None else 0.0

        # Mid-price returns
        mid_returns = mid_prices.pct_change().drop_nulls()
        if len(mid_returns) > 0:
            mid_std = mid_returns.std()
            features["mid_return_std"] = float(mid_std) if mid_std is not None else 0.0

            # Autocorrelation
            if len(mid_returns) > 20:
                returns_array = mid_returns.to_numpy()
                if len(returns_array) > 1:
                    # Simple autocorrelation at lag 1
                    autocorr = np.corrcoef(returns_array[:-1], returns_array[1:])[0, 1]
                    if not np.isnan(autocorr):
                        features["mid_return_autocorr"] = float(autocorr)
                    else:
                        features["mid_return_autocorr"] = 0.0
                else:
                    features["mid_return_autocorr"] = 0.0
            else:
                features["mid_return_autocorr"] = 0.0
        else:
            features["mid_return_std"] = 0.0
            features["mid_return_autocorr"] = 0.0

        return features

    def calculate_trade_flow_features(
        self,
        trades_df: Any,  # pl.DataFrame in cold path
    ) -> dict[str, float]:
        """
        Calculate trade flow features from trade data (COLD PATH ONLY).

        Parameters
        ----------
        trades_df : pl.DataFrame
            DataFrame with trade data including:
            - price: Trade price
            - size: Trade size
            - aggressor_side: 1 for buy, -1 for sell
            - timestamp: Trade timestamp (optional)

        Returns
        -------
        dict[str, float]
            Dictionary of trade flow features.

        """
        features = {}

        if not HAS_POLARS or len(trades_df) < 10:
            return {
                "trade_flow_imbalance": 0.0,
                "vwap": 0.0,
                "trade_intensity": 0.0,
                "avg_price_impact": 0.0,
            }

        # Trade flow imbalance
        buy_volume = trades_df.filter(pl.col("aggressor_side") > 0)["size"].sum()
        sell_volume = trades_df.filter(pl.col("aggressor_side") < 0)["size"].sum()
        total_volume = buy_volume + sell_volume

        if total_volume and total_volume > 0:
            features["trade_flow_imbalance"] = float((buy_volume - sell_volume) / total_volume)
        else:
            features["trade_flow_imbalance"] = 0.0

        # VWAP
        total_size = trades_df["size"].sum()
        if total_size is not None and total_size > 0:
            vwap = (trades_df["price"] * trades_df["size"]).sum() / total_size
            features["vwap"] = float(vwap) if vwap is not None else 0.0
        else:
            features["vwap"] = 0.0

        # Trade intensity
        if "timestamp" in trades_df.columns:
            max_time = trades_df["timestamp"].max()
            min_time = trades_df["timestamp"].min()
            if max_time is not None and min_time is not None:
                # Handle both datetime and numeric timestamps
                diff = max_time - min_time
                if hasattr(diff, "total_seconds"):
                    time_diff = diff.total_seconds()
                else:
                    # Assume nanoseconds
                    time_diff = float(diff) / 1e9

                if time_diff > 0:
                    features["trade_intensity"] = float(len(trades_df) / time_diff)
                else:
                    features["trade_intensity"] = 0.0
            else:
                features["trade_intensity"] = 0.0
        else:
            features["trade_intensity"] = 0.0

        # Average price impact
        prices = trades_df["price"]
        price_changes = prices.diff().abs()
        avg_impact_series = price_changes / trades_df["size"]
        avg_impact = avg_impact_series.mean()
        features["avg_price_impact"] = float(avg_impact) if avg_impact is not None else 0.0

        return features

    def calculate_features_online_with_microstructure(
        self,
        current_bar: dict[str, float],
        indicator_manager: Any,
        current_quote: dict[str, float] | None = None,
        recent_trades: list[dict[str, float]] | None = None,
        scaler: Any = None,
    ) -> np.ndarray:
        """
        Calculate features online including microstructure (HOT PATH).

        This method shows how to incorporate pre-computed microstructure
        features in the hot path. The actual microstructure calculations
        should be done by a separate Actor to maintain performance.

        Parameters
        ----------
        current_bar : dict[str, float]
            Current OHLCV data.
        indicator_manager : IndicatorManager
            Indicator manager with state.
        current_quote : dict[str, float], optional
            Current bid/ask quote.
        recent_trades : list[dict[str, float]], optional
            Recent trades for trade flow features.
        scaler : Any, optional
            Pre-fitted scaler.

        Returns
        -------
        np.ndarray
            Feature array for prediction.

        """
        # Get base features
        base_features = self.calculate_features_online(
            current_bar,
            indicator_manager,
            scaler=None,  # We'll scale at the end
        )

        # Determine total feature count
        n_features = len(base_features)

        # Add microstructure features if configured and data available
        if self.config.include_microstructure and current_quote is not None:
            # In hot path, use pre-computed or simplified calculations
            bid = current_quote.get("bid", 0.0)
            ask = current_quote.get("ask", 0.0)
            bid_size = current_quote.get("bid_size", 0.0)
            ask_size = current_quote.get("ask_size", 0.0)

            # Simple spread calculation
            spread = ask - bid if ask > bid else 0.0
            mid = (bid + ask) / 2 if ask > 0 and bid > 0 else current_bar["close"]

            # Fill microstructure buffer
            self.microstructure_buffer[0] = spread  # spread_mean proxy
            self.microstructure_buffer[1] = 0.0  # spread_std (need history)
            self.microstructure_buffer[2] = safe_divide(spread, mid)  # spread_relative

            # Size imbalance
            total_size = bid_size + ask_size
            if total_size > 0:
                self.microstructure_buffer[3] = (bid_size - ask_size) / total_size
            else:
                self.microstructure_buffer[3] = 0.0

            self.microstructure_buffer[4] = 0.0  # size_imbalance_std (need history)
            self.microstructure_buffer[5] = 0.0  # mid_return_std (need history)
            self.microstructure_buffer[6] = 0.0  # mid_return_autocorr (need history)

            n_features += 7

        # Add trade flow features if configured and data available
        if self.config.include_trade_flow and recent_trades:
            # In hot path, use simplified calculations
            buy_volume = sum(t["size"] for t in recent_trades if t.get("side", 0) > 0)
            sell_volume = sum(t["size"] for t in recent_trades if t.get("side", 0) < 0)
            total_volume = buy_volume + sell_volume

            if total_volume > 0:
                self.trade_flow_buffer[0] = (buy_volume - sell_volume) / total_volume
            else:
                self.trade_flow_buffer[0] = 0.0

            # Simple VWAP
            if recent_trades:
                total_value = sum(t["price"] * t["size"] for t in recent_trades)
                total_size = sum(t["size"] for t in recent_trades)
                self.trade_flow_buffer[1] = safe_divide(
                    total_value,
                    total_size,
                    current_bar["close"],
                )
            else:
                self.trade_flow_buffer[1] = current_bar["close"]

            # Trade intensity (trades per second)
            self.trade_flow_buffer[2] = len(recent_trades) / 60.0  # Assume 1-minute window

            # Avg price impact (simplified)
            self.trade_flow_buffer[3] = 0.0  # Would need proper calculation

            n_features += 4

        # Combine all features
        combined_features = np.zeros(n_features, dtype=np.float32)
        combined_features[: len(base_features)] = base_features

        idx = len(base_features)
        if self.config.include_microstructure and current_quote is not None:
            combined_features[idx : idx + 7] = self.microstructure_buffer
            idx += 7

        if self.config.include_trade_flow and recent_trades:
            combined_features[idx : idx + 4] = self.trade_flow_buffer

        # Apply scaling if provided
        if scaler is not None:
            combined_features = scaler.transform(combined_features.reshape(1, -1))[0]

        return combined_features

    def validate_feature_quality(
        self,
        features_df: Any,  # pl.DataFrame or pd.DataFrame
    ) -> dict[str, dict[str, float]]:
        """
        Validate feature quality metrics.

        Parameters
        ----------
        features_df : DataFrame
            Features to validate.

        Returns
        -------
        dict[str, dict[str, float]]
            Quality metrics per feature.

        """
        quality_metrics = {}

        if not HAS_POLARS:
            return quality_metrics

        for col in features_df.columns:
            if col in ["timestamp", "entity_id", "symbol"]:
                continue

            metrics = {
                "null_rate": features_df[col].null_count() / len(features_df),
                "zero_rate": (features_df[col] == 0).sum() / len(features_df),
                "unique_ratio": features_df[col].n_unique() / len(features_df),
            }

            # Check for infinities in numeric columns
            if features_df[col].dtype in [pl.Float32, pl.Float64]:
                metrics["inf_rate"] = features_df[col].is_infinite().sum() / len(features_df)

                # Calculate outlier rate using IQR
                q1 = features_df[col].quantile(0.25)
                q3 = features_df[col].quantile(0.75)
                if q1 is not None and q3 is not None:
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outliers = ((features_df[col] < lower) | (features_df[col] > upper)).sum()
                    metrics["outlier_rate"] = outliers / len(features_df)
                else:
                    metrics["outlier_rate"] = 0.0
            else:
                metrics["inf_rate"] = 0.0
                metrics["outlier_rate"] = 0.0

            quality_metrics[col] = metrics

        self.feature_quality_metrics = quality_metrics
        return quality_metrics

    def get_computation_stats(self) -> dict[str, dict[str, float]]:
        """
        Get computation statistics.

        Returns
        -------
        dict[str, dict[str, float]]
            Statistics per computation type.

        """
        stats = {}

        for comp_type, times in self.computation_times.items():
            if times:
                stats[comp_type] = {
                    "mean": float(np.mean(times)),
                    "std": float(np.std(times)),
                    "min": float(np.min(times)),
                    "max": float(np.max(times)),
                    "p50": float(np.percentile(times, 50)),
                    "p95": float(np.percentile(times, 95)),
                    "p99": float(np.percentile(times, 99)),
                }

        return stats


# Example usage and migration path
def migrate_to_enhanced_features():
    """
    Example showing how to migrate from basic to enhanced features.
    """
    # Step 1: Create enhanced config (backward compatible)
    config = FeatureConfig(
        # Existing parameters work as before
        return_periods=[1, 5, 10, 20],
        rsi_period=14,
        bb_period=20,
        # New optional features (default False)
        include_microstructure=True,
        include_trade_flow=True,
    )

    # Step 2: Use enhanced engineer (drop-in replacement)
    engineer = EnhancedFeatureEngineer(config)

    # Step 3: Works exactly like before for basic features
    # feature_names = engineer.get_feature_names()

    # Step 4: Can now use advanced features when data available
    # For batch processing (cold path):
    # features_df, scaler = engineer.calculate_features_batch(df)

    # With microstructure data:
    # microstructure_features = engineer.calculate_microstructure_features(quotes_df)

    # For online inference (hot path):
    # features = engineer.calculate_features_online_with_microstructure(
    #     current_bar,
    #     indicator_manager,
    #     current_quote=latest_quote,
    #     recent_trades=trades_window
    # )

    return engineer
