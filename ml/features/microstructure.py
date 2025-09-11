"""
Enhanced microstructure feature engineering for L2/L3 data.

This module provides advanced microstructure features computed from order book depth and
trade flow data collected from Databento. Implements features from academic literature
on market microstructure.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from ml._imports import pd
from ml._imports import pl
from ml.common.safe_math import safe_divide


if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


class L2MicrostructureFeatures:
    """
    Compute microstructure features from L2 order book data.

    Features include:
    - Bid-ask spread metrics
    - Order book imbalance
    - Depth-weighted midpoint
    - Price impact measures
    - Order book shape metrics

    """

    def __init__(
        self,
        n_levels: int = 10,
        lookback_window: int = 20,
        compute_ratios: bool = True,
    ) -> None:
        """
        Initialize L2 microstructure feature calculator.

        Parameters
        ----------
        n_levels : int, default 10
            Number of order book levels to use
        lookback_window : int, default 20
            Lookback window for rolling calculations
        compute_ratios : bool, default True
            Whether to compute ratio-based features

        """
        self.n_levels = n_levels
        self.lookback_window = lookback_window
        self.compute_ratios = compute_ratios

    def compute_spread_features(
        self,
        bid_prices: npt.NDArray[np.float64],
        ask_prices: npt.NDArray[np.float64],
        bid_sizes: npt.NDArray[np.float64],
        ask_sizes: npt.NDArray[np.float64],
    ) -> dict[str, float]:
        """
        Compute spread-related features.

        Parameters
        ----------
        bid_prices : np.ndarray
            Bid prices for each level (n_samples, n_levels)
        ask_prices : np.ndarray
            Ask prices for each level (n_samples, n_levels)
        bid_sizes : np.ndarray
            Bid sizes for each level (n_samples, n_levels)
        ask_sizes : np.ndarray
            Ask sizes for each level (n_samples, n_levels)

        Returns
        -------
        dict[str, float]
            Spread features

        """
        features: dict[str, float] = {}

        # Basic spread
        best_bid = bid_prices[:, 0]
        best_ask = ask_prices[:, 0]
        spread = best_ask - best_bid
        midpoint = (best_ask + best_bid) / 2.0

        features["spread"] = float(spread[-1])
        features["spread_bps"] = float(safe_divide(spread[-1], midpoint[-1], 0.0) * 10000.0)
        features["spread_mean"] = float(np.mean(spread[-self.lookback_window :]))
        features["spread_std"] = float(np.std(spread[-self.lookback_window :]))

        # Weighted spread (by size)
        total_size = bid_sizes[:, 0] + ask_sizes[:, 0]
        weighted_spread = spread * total_size
        features["weighted_spread"] = float(weighted_spread[-1])

        # Effective spread (using actual trades if available)
        denom = float(bid_sizes[-1, 0] + ask_sizes[-1, 0])
        weighted_price = safe_divide(
            float(best_bid[-1] * bid_sizes[-1, 0] + best_ask[-1] * ask_sizes[-1, 0]),
            denom,
            default=float(midpoint[-1]),
        )
        features["effective_spread_proxy"] = float(2.0 * abs(float(midpoint[-1]) - weighted_price))

        # Spread volatility
        if len(spread) > 1:
            spread_returns = np.diff(spread) / spread[:-1]
            features["spread_volatility"] = float(np.std(spread_returns))
        else:
            features["spread_volatility"] = 0.0

        return features

    def compute_imbalance_features(
        self,
        bid_sizes: npt.NDArray[np.float64],
        ask_sizes: npt.NDArray[np.float64],
        bid_prices: npt.NDArray[np.float64] | None = None,
        ask_prices: npt.NDArray[np.float64] | None = None,
    ) -> dict[str, float]:
        """
        Compute order book imbalance features.

        Parameters
        ----------
        bid_sizes : np.ndarray
            Bid sizes for each level
        ask_sizes : np.ndarray
            Ask sizes for each level
        bid_prices : np.ndarray, optional
            Bid prices for weighted imbalance
        ask_prices : np.ndarray, optional
            Ask prices for weighted imbalance

        Returns
        -------
        dict[str, float]
            Imbalance features

        """
        features: dict[str, float] = {}

        # Level 1 imbalance
        bid_size_l1 = bid_sizes[:, 0]
        ask_size_l1 = ask_sizes[:, 0]
        imbalance_l1 = (bid_size_l1 - ask_size_l1) / (bid_size_l1 + ask_size_l1 + 1e-10)

        # Scalar last-value using safe division
        features["imbalance_l1"] = float(
            safe_divide(
                float(bid_size_l1[-1] - ask_size_l1[-1]),
                float(bid_size_l1[-1] + ask_size_l1[-1]),
                0.0,
            ),
        )
        features["imbalance_l1_mean"] = float(np.mean(imbalance_l1[-self.lookback_window :]))
        features["imbalance_l1_std"] = float(np.std(imbalance_l1[-self.lookback_window :]))

        # Multi-level imbalance (top 5 levels)
        top_levels = min(5, self.n_levels)
        bid_size_top = np.sum(bid_sizes[:, :top_levels], axis=1)
        ask_size_top = np.sum(ask_sizes[:, :top_levels], axis=1)
        imbalance_top = (bid_size_top - ask_size_top) / (bid_size_top + ask_size_top + 1e-10)

        features["imbalance_top5"] = float(imbalance_top[-1])
        features["imbalance_top5_mean"] = float(np.mean(imbalance_top[-self.lookback_window :]))

        # Full book imbalance
        bid_size_total = np.sum(bid_sizes, axis=1)
        ask_size_total = np.sum(ask_sizes, axis=1)
        imbalance_total = (bid_size_total - ask_size_total) / (
            bid_size_total + ask_size_total + 1e-10
        )

        features["imbalance_total"] = float(imbalance_total[-1])

        # Weighted imbalance (by inverse price distance from mid)
        if bid_prices is not None and ask_prices is not None:
            midpoint = (bid_prices[:, 0] + ask_prices[:, 0]) / 2.0

            # Weight by inverse distance from midpoint
            bid_weights = 1.0 / (np.abs(bid_prices - midpoint[:, np.newaxis]) + 1e-10)
            ask_weights = 1.0 / (np.abs(ask_prices - midpoint[:, np.newaxis]) + 1e-10)

            weighted_bid_size = np.sum(bid_sizes * bid_weights, axis=1)
            weighted_ask_size = np.sum(ask_sizes * ask_weights, axis=1)

            weighted_imbalance = (weighted_bid_size - weighted_ask_size) / (
                weighted_bid_size + weighted_ask_size + 1e-10
            )

            features["imbalance_weighted"] = float(weighted_imbalance[-1])

        # Imbalance ratios at different levels
        if self.compute_ratios:
            for level in [1, 3, 5]:
                if level <= self.n_levels:
                    level_idx = level - 1
                    ratio = safe_divide(
                        float(bid_sizes[-1, level_idx]),
                        float(ask_sizes[-1, level_idx]),
                        0.0,
                    )
                    features[f"bid_ask_ratio_l{level}"] = float(np.log(ratio + 1e-10))

        return features

    def compute_depth_features(
        self,
        bid_sizes: npt.NDArray[np.float64],
        ask_sizes: npt.NDArray[np.float64],
        bid_prices: npt.NDArray[np.float64],
        ask_prices: npt.NDArray[np.float64],
    ) -> dict[str, float]:
        """
        Compute order book depth features.

        Parameters
        ----------
        bid_sizes : np.ndarray
            Bid sizes for each level
        ask_sizes : np.ndarray
            Ask sizes for each level
        bid_prices : np.ndarray
            Bid prices for each level
        ask_prices : np.ndarray
            Ask prices for each level

        Returns
        -------
        dict[str, float]
            Depth features

        """
        features: dict[str, float] = {}

        # Total depth
        total_bid_depth = np.sum(bid_sizes, axis=1)
        total_ask_depth = np.sum(ask_sizes, axis=1)

        features["bid_depth_total"] = float(total_bid_depth[-1])
        features["ask_depth_total"] = float(total_ask_depth[-1])
        features["depth_ratio"] = float(
            safe_divide(float(total_bid_depth[-1]), float(total_ask_depth[-1]), 0.0),
        )

        # Depth concentration (how much volume in top levels)
        bid_concentration = bid_sizes[:, 0] / (total_bid_depth + 1e-10)
        ask_concentration = ask_sizes[:, 0] / (total_ask_depth + 1e-10)

        features["bid_concentration_l1"] = float(bid_concentration[-1])
        features["ask_concentration_l1"] = float(ask_concentration[-1])

        # Depth-weighted average price (VWAP-like for order book)
        bid_vwap = np.sum(bid_prices * bid_sizes, axis=1) / (total_bid_depth + 1e-10)
        ask_vwap = np.sum(ask_prices * ask_sizes, axis=1) / (total_ask_depth + 1e-10)

        features["bid_vwap"] = float(bid_vwap[-1])
        features["ask_vwap"] = float(ask_vwap[-1])
        features["vwap_spread"] = float(ask_vwap[-1] - bid_vwap[-1])

        # Depth slope (how quickly depth drops off)
        if self.n_levels > 1:
            # Calculate average size decay per level
            bid_slope = np.polyfit(range(self.n_levels), bid_sizes[-1, :], 1)[0]
            ask_slope = np.polyfit(range(self.n_levels), ask_sizes[-1, :], 1)[0]

            features["bid_depth_slope"] = float(bid_slope)
            features["ask_depth_slope"] = float(ask_slope)

        return features

    def compute_shape_features(
        self,
        bid_sizes: npt.NDArray[np.float64],
        ask_sizes: npt.NDArray[np.float64],
        bid_prices: npt.NDArray[np.float64],
        ask_prices: npt.NDArray[np.float64],
    ) -> dict[str, float]:
        """
        Compute order book shape features.

        Parameters
        ----------
        bid_sizes : np.ndarray
            Bid sizes for each level
        ask_sizes : np.ndarray
            Ask sizes for each level
        bid_prices : np.ndarray
            Bid prices for each level
        ask_prices : np.ndarray
            Ask prices for each level

        Returns
        -------
        dict[str, float]
            Shape features

        """
        features: dict[str, float] = {}

        # Skewness of order book
        total_bid = np.sum(bid_sizes[-1, :])
        total_ask = np.sum(ask_sizes[-1, :])

        features["book_skewness"] = float(
            safe_divide(float(total_bid - total_ask), float(total_bid + total_ask), 0.0),
        )

        # Kurtosis proxy (concentration measure)
        bid_kurtosis = np.sum((bid_sizes[-1, :] / (total_bid + 1e-10)) ** 4)
        ask_kurtosis = np.sum((ask_sizes[-1, :] / (total_ask + 1e-10)) ** 4)

        features["bid_kurtosis"] = float(bid_kurtosis)
        features["ask_kurtosis"] = float(ask_kurtosis)

        # Price range covered by book
        bid_range = bid_prices[-1, 0] - bid_prices[-1, -1]
        ask_range = ask_prices[-1, -1] - ask_prices[-1, 0]

        features["bid_price_range"] = float(bid_range)
        features["ask_price_range"] = float(ask_range)

        # Liquidity concentration zones

        # Find levels with high liquidity (>20% of average level size)
        avg_bid_size = np.mean(bid_sizes[-1, :])
        avg_ask_size = np.mean(ask_sizes[-1, :])

        high_bid_levels = np.sum(bid_sizes[-1, :] > 1.2 * avg_bid_size)
        high_ask_levels = np.sum(ask_sizes[-1, :] > 1.2 * avg_ask_size)

        features["bid_liquidity_zones"] = float(high_bid_levels)
        features["ask_liquidity_zones"] = float(high_ask_levels)

        return features

    def compute_all_features(
        self,
        df: pl.DataFrame | pd.DataFrame,
    ) -> dict[str, npt.NDArray[np.float64]]:
        """
        Compute all L2 microstructure features.

        Parameters
        ----------
        df : DataFrame
            DataFrame with L2 order book data

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary of feature arrays

        """
        # Extract order book data
        bid_prices, ask_prices, bid_sizes, ask_sizes = self._extract_l2_data(df)

        all_features: dict[str, list[float]] = {}

        # Compute features for each timestamp
        for i in range(self.lookback_window, len(bid_prices)):
            window_slice = slice(max(0, i - self.lookback_window), i + 1)

            # Compute different feature groups
            spread_features = self.compute_spread_features(
                bid_prices[window_slice],
                ask_prices[window_slice],
                bid_sizes[window_slice],
                ask_sizes[window_slice],
            )

            imbalance_features = self.compute_imbalance_features(
                bid_sizes[window_slice],
                ask_sizes[window_slice],
                bid_prices[window_slice],
                ask_prices[window_slice],
            )

            depth_features = self.compute_depth_features(
                bid_sizes[window_slice],
                ask_sizes[window_slice],
                bid_prices[window_slice],
                ask_prices[window_slice],
            )

            shape_features = self.compute_shape_features(
                bid_sizes[window_slice],
                ask_sizes[window_slice],
                bid_prices[window_slice],
                ask_prices[window_slice],
            )

            # Combine all features
            timestamp_features = {
                **spread_features,
                **imbalance_features,
                **depth_features,
                **shape_features,
            }

            # Add to arrays
            for key, value in timestamp_features.items():
                if key not in all_features:
                    all_features[key] = []
                all_features[key].append(value)

        # Convert lists to arrays
        return {key: np.array(values) for key, values in all_features.items()}

    def _extract_l2_data(
        self,
        df: pl.DataFrame | pd.DataFrame,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """
        Extract L2 order book data from DataFrame.

        Parameters
        ----------
        df : DataFrame
            Input dataframe with L2 data

        Returns
        -------
        tuple
            bid_prices, ask_prices, bid_sizes, ask_sizes arrays

        """
        if isinstance(df, pl.DataFrame):
            # Extract from Polars DataFrame
            bid_prices = []
            ask_prices = []
            bid_sizes = []
            ask_sizes = []

            for level in range(self.n_levels):
                bid_col = f"bid_price_{level}"
                ask_col = f"ask_price_{level}"
                bid_size_col = f"bid_size_{level}"
                ask_size_col = f"ask_size_{level}"

                if bid_col in df.columns:
                    bid_prices.append(df[bid_col].to_numpy())
                    ask_prices.append(df[ask_col].to_numpy())
                    bid_sizes.append(df[bid_size_col].to_numpy())
                    ask_sizes.append(df[ask_size_col].to_numpy())

            # Stack into (n_samples, n_levels) arrays
            bid_prices_arr = np.column_stack(bid_prices)
            ask_prices_arr = np.column_stack(ask_prices)
            bid_sizes_arr = np.column_stack(bid_sizes)
            ask_sizes_arr = np.column_stack(ask_sizes)

        else:
            # Extract from Pandas DataFrame
            bid_prices = []
            ask_prices = []
            bid_sizes = []
            ask_sizes = []

            for level in range(self.n_levels):
                bid_col = f"bid_price_{level}"
                ask_col = f"ask_price_{level}"
                bid_size_col = f"bid_size_{level}"
                ask_size_col = f"ask_size_{level}"

                if bid_col in df.columns:
                    bid_prices.append(np.asarray(df[bid_col].values))
                    ask_prices.append(np.asarray(df[ask_col].values))
                    bid_sizes.append(np.asarray(df[bid_size_col].values))
                    ask_sizes.append(np.asarray(df[ask_size_col].values))

            bid_prices_arr = np.column_stack(bid_prices)
            ask_prices_arr = np.column_stack(ask_prices)
            bid_sizes_arr = np.column_stack(bid_sizes)
            ask_sizes_arr = np.column_stack(ask_sizes)

        return bid_prices_arr, ask_prices_arr, bid_sizes_arr, ask_sizes_arr


class L3TradeFlowFeatures:
    """
    Compute trade flow features from L3 trade data.

    Features include:
    - Trade imbalance
    - Volume-weighted average price (VWAP)
    - Trade intensity
    - Price impact measures
    - Aggressive vs passive flow

    """

    def __init__(
        self,
        lookback_window: int = 100,
        volume_buckets: int = 10,
    ) -> None:
        """
        Initialize L3 trade flow feature calculator.

        Parameters
        ----------
        lookback_window : int, default 100
            Number of trades to look back
        volume_buckets : int, default 10
            Number of volume buckets for VPIN calculation

        """
        self.lookback_window = lookback_window
        self.volume_buckets = volume_buckets

    def compute_trade_imbalance(
        self,
        prices: npt.NDArray[np.float64],
        volumes: npt.NDArray[np.float64],
        sides: npt.NDArray[np.int64],
    ) -> dict[str, float]:
        """
        Compute trade imbalance features.

        Parameters
        ----------
        prices : np.ndarray
            Trade prices
        volumes : np.ndarray
            Trade volumes
        sides : np.ndarray
            Trade sides (1 for buy, -1 for sell)

        Returns
        -------
        dict[str, float]
            Trade imbalance features

        """
        features: dict[str, float] = {}

        # Signed volume
        signed_volume = volumes * sides

        # Trade imbalance
        buy_volume = np.sum(volumes[sides == 1])
        sell_volume = np.sum(volumes[sides == -1])
        total_volume = buy_volume + sell_volume

        features["trade_imbalance"] = float(
            safe_divide(float(buy_volume - sell_volume), float(total_volume), 0.0),
        )

        # Dollar volume imbalance
        buy_dollar_volume = np.sum(prices[sides == 1] * volumes[sides == 1])
        sell_dollar_volume = np.sum(prices[sides == -1] * volumes[sides == -1])
        total_dollar_volume = buy_dollar_volume + sell_dollar_volume

        features["dollar_imbalance"] = float(
            safe_divide(
                float(buy_dollar_volume - sell_dollar_volume),
                float(total_dollar_volume),
                0.0,
            ),
        )

        # Trade count imbalance
        buy_count = np.sum(sides == 1)
        sell_count = np.sum(sides == -1)
        total_count = buy_count + sell_count

        features["trade_count_imbalance"] = float(
            safe_divide(float(buy_count - sell_count), float(total_count), 0.0),
        )

        # Cumulative signed volume (momentum indicator)
        cumulative_signed_volume = np.cumsum(signed_volume)
        features["cumulative_flow"] = float(cumulative_signed_volume[-1])
        features["flow_acceleration"] = float(
            (
                cumulative_signed_volume[-1]
                - cumulative_signed_volume[len(cumulative_signed_volume) // 2]
                if len(cumulative_signed_volume) > 1
                else 0
            ),
        )

        return features

    def compute_vwap_features(
        self,
        prices: npt.NDArray[np.float64],
        volumes: npt.NDArray[np.float64],
        sides: npt.NDArray[np.int64] | None = None,
    ) -> dict[str, float]:
        """
        Compute VWAP-related features.

        Parameters
        ----------
        prices : np.ndarray
            Trade prices
        volumes : np.ndarray
            Trade volumes
        sides : np.ndarray, optional
            Trade sides for sided VWAP

        Returns
        -------
        dict[str, float]
            VWAP features

        """
        features: dict[str, float] = {}

        # Overall VWAP
        total_volume = np.sum(volumes)
        if total_volume > 0:
            vwap = np.sum(prices * volumes) / total_volume
            features["vwap"] = float(vwap)

            # Price deviation from VWAP
            current_price = prices[-1]
            features["price_vs_vwap"] = float(safe_divide(float(current_price - vwap), float(vwap), 0.0))

            # VWAP variance (measure of price dispersion)
            vwap_variance = np.sum(volumes * (prices - vwap) ** 2) / total_volume
            features["vwap_variance"] = float(vwap_variance)
        else:
            features["vwap"] = float(prices[-1]) if len(prices) > 0 else 0.0
            features["price_vs_vwap"] = 0.0
            features["vwap_variance"] = 0.0

        # Sided VWAP
        if sides is not None:
            buy_mask = sides == 1
            sell_mask = sides == -1

            buy_volume = np.sum(volumes[buy_mask])
            if buy_volume > 0:
                buy_vwap = np.sum(prices[buy_mask] * volumes[buy_mask]) / buy_volume
                features["buy_vwap"] = float(buy_vwap)
            else:
                features["buy_vwap"] = features["vwap"]

            sell_volume = np.sum(volumes[sell_mask])
            if sell_volume > 0:
                sell_vwap = np.sum(prices[sell_mask] * volumes[sell_mask]) / sell_volume
                features["sell_vwap"] = float(sell_vwap)
            else:
                features["sell_vwap"] = features["vwap"]

            # VWAP spread (buy vs sell)
            features["vwap_spread"] = float(features["buy_vwap"] - features["sell_vwap"])

        return features

    def compute_intensity_features(
        self,
        timestamps: npt.NDArray[np.int64],
        volumes: npt.NDArray[np.float64],
        prices: npt.NDArray[np.float64],
    ) -> dict[str, float]:
        """
        Compute trade intensity features.

        Parameters
        ----------
        timestamps : np.ndarray
            Trade timestamps in nanoseconds
        volumes : np.ndarray
            Trade volumes
        prices : np.ndarray
            Trade prices

        Returns
        -------
        dict[str, float]
            Trade intensity features

        """
        features: dict[str, float] = {}

        if len(timestamps) < 2:
            features["trade_rate"] = 0.0
            features["volume_rate"] = 0.0
            features["dollar_rate"] = 0.0
            features["avg_trade_size"] = float(volumes[0]) if len(volumes) > 0 else 0.0
            return features

        # Time span in seconds
        time_span_ns = timestamps[-1] - timestamps[0]
        time_span_s = time_span_ns / 1e9

        # Rates per second (safe division to avoid spikes on tiny windows)
        features["trade_rate"] = float(safe_divide(float(len(timestamps)), float(time_span_s), 0.0))
        features["volume_rate"] = float(
            safe_divide(float(np.sum(volumes)), float(time_span_s), 0.0),
        )
        features["dollar_rate"] = float(
            safe_divide(float(np.sum(volumes * prices)), float(time_span_s), 0.0),
        )

        # Average trade size
        features["avg_trade_size"] = float(np.mean(volumes))
        features["trade_size_std"] = float(np.std(volumes))

        # Trade clustering (using inter-trade times)
        inter_trade_times = np.diff(timestamps) / 1e9  # Convert to seconds
        if len(inter_trade_times) > 0:
            features["avg_inter_trade_time"] = float(np.mean(inter_trade_times))
            features["inter_trade_time_std"] = float(np.std(inter_trade_times))

            # Clustering coefficient (low inter-trade time variance = high clustering)
            ratio = safe_divide(
                float(features["inter_trade_time_std"]),
                float(features["avg_inter_trade_time"]),
                0.0,
            )
            features["trade_clustering"] = float(1.0 / (1.0 + ratio))
        else:
            features["avg_inter_trade_time"] = 0.0
            features["inter_trade_time_std"] = 0.0
            features["trade_clustering"] = 0.0

        return features

    def compute_price_impact(
        self,
        prices: npt.NDArray[np.float64],
        volumes: npt.NDArray[np.float64],
        sides: npt.NDArray[np.int64],
    ) -> dict[str, float]:
        """
        Compute price impact features.

        Parameters
        ----------
        prices : np.ndarray
            Trade prices
        volumes : np.ndarray
            Trade volumes
        sides : np.ndarray
            Trade sides

        Returns
        -------
        dict[str, float]
            Price impact features

        """
        features: dict[str, float] = {}

        if len(prices) < 2:
            features["avg_price_impact"] = 0.0
            features["kyle_lambda"] = 0.0
            return features

        # Simple price impact: price change per unit volume
        price_changes = np.diff(prices)
        volumes_slice = volumes[1:]  # Align with price changes
        sides_slice = sides[1:]

        # Signed price changes (buy trades should push price up)
        signed_price_changes = price_changes * sides_slice

        # Average price impact per unit volume
        if np.sum(volumes_slice) > 0:
            avg_impact = np.sum(np.abs(signed_price_changes)) / np.sum(volumes_slice)
            features["avg_price_impact"] = float(avg_impact)
        else:
            features["avg_price_impact"] = 0.0

        # Kyle's lambda (regression coefficient)
        signed_volumes = volumes * sides
        if len(price_changes) > 1 and np.std(signed_volumes[1:]) > 0:
            # Regress price changes on signed volumes
            coef = np.polyfit(signed_volumes[1:], price_changes, 1)[0]
            features["kyle_lambda"] = float(coef)
        else:
            features["kyle_lambda"] = 0.0

        # Temporary vs permanent impact (simplified)
        if len(prices) > 10:
            # Immediate impact (next trade)
            immediate_impacts = []
            for i in range(len(prices) - 1):
                immediate_impact = (prices[i + 1] - prices[i]) * sides[i]
                immediate_impacts.append(immediate_impact)

            # Permanent impact (price after 10 trades)
            permanent_impacts = []
            for i in range(len(prices) - 10):
                permanent_impact = (prices[i + 10] - prices[i]) * sides[i]
                permanent_impacts.append(permanent_impact)

            features["immediate_impact"] = float(np.mean(np.abs(immediate_impacts)))
            features["permanent_impact"] = float(np.mean(np.abs(permanent_impacts)))

            # Temporary impact = immediate - permanent
            features["temporary_impact"] = float(
                features["immediate_impact"] - features["permanent_impact"],
            )
        else:
            features["immediate_impact"] = features["avg_price_impact"]
            features["permanent_impact"] = features["avg_price_impact"]
            features["temporary_impact"] = 0.0

        return features

    def compute_all_features(
        self,
        df: pl.DataFrame | pd.DataFrame,
    ) -> dict[str, npt.NDArray[np.float64]]:
        """
        Compute all L3 trade flow features.

        Parameters
        ----------
        df : DataFrame
            DataFrame with L3 trade data

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary of feature arrays

        """
        # Extract trade data
        timestamps, prices, volumes, sides = self._extract_l3_data(df)

        all_features: dict[str, list[float]] = {}

        # Compute features for rolling windows
        for i in range(self.lookback_window, len(prices)):
            window_slice = slice(max(0, i - self.lookback_window), i + 1)

            # Compute different feature groups
            imbalance_features = self.compute_trade_imbalance(
                prices[window_slice],
                volumes[window_slice],
                sides[window_slice],
            )

            vwap_features = self.compute_vwap_features(
                prices[window_slice],
                volumes[window_slice],
                sides[window_slice],
            )

            intensity_features = self.compute_intensity_features(
                timestamps[window_slice],
                volumes[window_slice],
                prices[window_slice],
            )

            impact_features = self.compute_price_impact(
                prices[window_slice],
                volumes[window_slice],
                sides[window_slice],
            )

            # Combine all features
            timestamp_features = {
                **imbalance_features,
                **vwap_features,
                **intensity_features,
                **impact_features,
            }

            # Add to arrays
            for key, value in timestamp_features.items():
                if key not in all_features:
                    all_features[key] = []
                all_features[key].append(value)

        # Convert lists to arrays
        return {key: np.array(values) for key, values in all_features.items()}

    def _extract_l3_data(
        self,
        df: pl.DataFrame | pd.DataFrame,
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
    ]:
        """
        Extract L3 trade data from DataFrame.

        Parameters
        ----------
        df : DataFrame
            Input dataframe with L3 data

        Returns
        -------
        tuple
            timestamps, prices, volumes, sides arrays

        """
        if isinstance(df, pl.DataFrame):
            # Extract from Polars DataFrame
            timestamps = df["ts_event"].to_numpy()
            prices = df["price"].to_numpy()
            volumes = df["volume"].to_numpy()

            # Convert side to numeric (1 for buy, -1 for sell)
            if "side" in df.columns:
                side_str = df["side"].to_numpy()
                sides = np.where(side_str == "BUY", 1, -1)
            elif "aggressor_side" in df.columns:
                side_str = df["aggressor_side"].to_numpy()
                sides = np.where(side_str == "BUY", 1, -1)
            else:
                # Infer from price movement
                price_changes = np.diff(prices, prepend=prices[0])
                sides = np.where(price_changes >= 0, 1, -1)
        else:
            # Extract from Pandas DataFrame
            timestamps = np.asarray(df["ts_event"].values, dtype=np.int64)
            prices = np.asarray(df["price"].values, dtype=np.float64)
            volumes = np.asarray(df["volume"].values, dtype=np.float64)

            # Convert side to numeric
            if "side" in df.columns:
                side_str = np.asarray(df["side"].values)
                sides = np.where(side_str == "BUY", 1, -1)
            elif "aggressor_side" in df.columns:
                side_str = np.asarray(df["aggressor_side"].values)
                sides = np.where(side_str == "BUY", 1, -1)
            else:
                # Infer from price movement
                price_changes = np.diff(prices, prepend=prices[0])
                sides = np.where(price_changes >= 0, 1, -1)

        return (
            cast(npt.NDArray[np.int64], timestamps),
            cast(npt.NDArray[np.float64], prices),
            cast(npt.NDArray[np.float64], volumes),
            cast(npt.NDArray[np.int64], sides),
        )
