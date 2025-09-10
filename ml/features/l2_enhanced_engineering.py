"""
Enhanced FeatureEngineer with L2/L3 hot path support.

This module provides patches and enhancements to enable real-time L2/L3
microstructure features in the hot path while maintaining performance requirements.

Key Enhancements:
- Proper L2/L3 feature computation in online mode
- Feature parity between batch and online modes (37 features)
- Order book data integration for MLSignalActor
- Performance optimizations for <5ms target

Usage:
    Replace calls to standard FeatureEngineer with L2FeatureEngineer
    for hot path L2/L3 microstructure features.

"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.features.engineering import IndicatorManager
from nautilus_trader.model.book import OrderBook


if TYPE_CHECKING:
    pass


class L2IndicatorManager(IndicatorManager):
    """
    Enhanced indicator manager with L2/L3 order book integration.

    Extends the base IndicatorManager to track order book state alongside price/volume
    indicators for microstructure feature computation.

    """

    def __init__(self, config: FeatureConfig) -> None:
        """
        Initialize with L2/L3 tracking capabilities.
        """
        super().__init__(config)

        # Order book state tracking
        self.order_book: OrderBook | None = None
        self.book_levels = 10  # Track top 10 levels
        self.last_book_update_ns = 0

        # L2 feature buffers (pre-allocated for zero allocation)
        self.bid_prices = np.zeros(self.book_levels, dtype=np.float32)
        self.ask_prices = np.zeros(self.book_levels, dtype=np.float32)
        self.bid_sizes = np.zeros(self.book_levels, dtype=np.float32)
        self.ask_sizes = np.zeros(self.book_levels, dtype=np.float32)

        # Historical L2 features for windowed calculations
        self.spread_history = np.zeros(20, dtype=np.float32)
        self.imbalance_history = np.zeros(20, dtype=np.float32)
        self.microprice_history = np.zeros(20, dtype=np.float32)
        self.l2_history_idx = 0
        self.l2_history_filled = 0

    def update_order_book(self, book: OrderBook) -> None:
        """
        Update internal order book state.
        """
        self.order_book = book
        self.last_book_update_ns = time.perf_counter_ns()

        if book is None or book.count == 0:
            return

        # Extract price/size arrays efficiently
        levels = min(book.count, self.book_levels)

        # Reset arrays
        self.bid_prices.fill(0.0)
        self.ask_prices.fill(0.0)
        self.bid_sizes.fill(0.0)
        self.ask_sizes.fill(0.0)

        # Fill from order book
        for i, order in enumerate(book.bids.orders()[:levels]):
            self.bid_prices[i] = float(order.price)
            self.bid_sizes[i] = float(order.size)

        for i, order in enumerate(book.asks.orders()[:levels]):
            self.ask_prices[i] = float(order.price)
            self.ask_sizes[i] = float(order.size)

    def has_valid_l2_data(self, max_age_ms: float = 1000.0) -> bool:
        """
        Check if L2 data is available and not stale.
        """
        if self.last_book_update_ns == 0:
            return False

        age_ms = (time.perf_counter_ns() - self.last_book_update_ns) / 1_000_000
        return (
            age_ms <= max_age_ms
            and self.bid_prices[0] > 0
            and self.ask_prices[0] > 0
            and self.order_book is not None
        )


class L2FeatureEngineer(FeatureEngineer):
    """
    Enhanced FeatureEngineer with L2/L3 microstructure features.

    Provides feature parity between batch and online modes by enabling real-time L2/L3
    computation in the hot path.

    """

    def __init__(self, config: FeatureConfig) -> None:
        """
        Initialize with L2/L3 capabilities.
        """
        super().__init__(config)

        # Enable L2/L3 features in online mode
        self._enable_l2_online = True
        self._l2_feature_count = 11  # Number of L2/L3 features added

        # Update feature count to include L2/L3 features
        if config.include_microstructure or config.include_trade_flow:
            self.n_features += self._l2_feature_count

        # Pre-allocate enhanced feature buffer
        self.feature_buffer = np.zeros(self.n_features, dtype=np.float32)

    def create_indicator_manager(self) -> L2IndicatorManager:
        """
        Create L2-enhanced indicator manager.
        """
        return L2IndicatorManager(self.config)

    def calculate_features_online(
        self,
        current_bar: dict[str, float] | None = None,
        indicator_manager: IndicatorManager | None = None,
        scaler: Any | None = None,
        *,
        close_price: float | None = None,
        high_price: float | None = None,
        low_price: float | None = None,
        volume: float | None = None,
        order_book: OrderBook | None = None,
    ) -> npt.NDArray[np.float32]:
        """
        Enhanced online feature calculation with L2/L3 support.

        Parameters
        ----------
        order_book : OrderBook | None
            Current order book state for L2/L3 features

        All other parameters same as base class.

        """
        # Handle L2 indicator manager
        if isinstance(indicator_manager, L2IndicatorManager) and order_book is not None:
            indicator_manager.update_order_book(order_book)

        # Compute base L1 features first
        base_features = super().calculate_features_online(
            current_bar=current_bar,
            indicator_manager=indicator_manager,
            scaler=None,  # Don't scale yet - need to add L2/L3 features first
            close_price=close_price,
            high_price=high_price,
            low_price=low_price,
            volume=volume,
        )

        # If L2/L3 not requested, return base features
        if not (self.config.include_microstructure or self.config.include_trade_flow):
            if scaler is not None:
                features_2d = base_features.reshape(1, -1)
                scaled = scaler.transform(features_2d)
                return np.asarray(scaled[0], dtype=np.float32)
            return base_features

        # Add L2/L3 features
        extended_features = self._add_l2_features_online(
            base_features,
            indicator_manager,
            current_bar,
        )

        # Apply scaling to complete feature set
        if scaler is not None:
            features_2d = extended_features.reshape(1, -1)
            scaled = scaler.transform(features_2d)
            return np.asarray(scaled[0], dtype=np.float32)

        return extended_features

    def _add_l2_features_online(
        self,
        base_features: npt.NDArray[np.float32],
        indicator_manager: IndicatorManager | None,
        current_bar: dict[str, float] | None,
    ) -> npt.NDArray[np.float32]:
        """
        Add L2/L3 features to base feature set.
        """
        # Extend feature buffer
        total_features = len(base_features) + self._l2_feature_count
        if len(self.feature_buffer) < total_features:
            self.feature_buffer = np.zeros(total_features, dtype=np.float32)

        # Copy base features
        self.feature_buffer[: len(base_features)] = base_features
        feature_idx = len(base_features)

        # Check if we have L2 data
        has_l2_data = (
            isinstance(indicator_manager, L2IndicatorManager)
            and indicator_manager.has_valid_l2_data()
        )

        if has_l2_data:
            # Compute real L2/L3 features
            feature_idx = self._compute_real_l2_features(
                indicator_manager,
                feature_idx,
            )
        else:
            # Use approximations from OHLCV
            feature_idx = self._compute_l2_approximations(
                current_bar,
                indicator_manager,
                feature_idx,
            )

        return self.feature_buffer[:feature_idx]

    def _compute_real_l2_features(
        self,
        l2_manager: L2IndicatorManager,
        feature_idx: int,
    ) -> int:
        """
        Compute real L2/L3 features from order book data.
        """
        # Get L2 data
        bid_prices = l2_manager.bid_prices
        ask_prices = l2_manager.ask_prices
        bid_sizes = l2_manager.bid_sizes
        ask_sizes = l2_manager.ask_sizes

        # Core calculations
        best_bid = bid_prices[0]
        best_ask = ask_prices[0]

        if best_bid <= 0 or best_ask <= 0:
            # Fallback to approximations
            return self._compute_l2_approximations(None, l2_manager, feature_idx)

        mid_price = (best_bid + best_ask) * 0.5
        bid_size_0 = bid_sizes[0]
        ask_size_0 = ask_sizes[0]

        # 1. Spread (bps)
        spread_bps = 10000.0 * (best_ask - best_bid) / mid_price if mid_price > 0 else 0.0
        self.feature_buffer[feature_idx] = np.float32(spread_bps)
        feature_idx += 1

        # 2. Microprice (bps from mid)
        total_size_0 = bid_size_0 + ask_size_0
        if total_size_0 > 0:
            microprice = (best_ask * bid_size_0 + best_bid * ask_size_0) / total_size_0
            microprice_bps = (
                10000.0 * (microprice - mid_price) / mid_price if mid_price > 0 else 0.0
            )
        else:
            microprice_bps = 0.0
        self.feature_buffer[feature_idx] = np.float32(microprice_bps)
        feature_idx += 1

        # 3. L1 imbalance
        if total_size_0 > 0:
            imbalance = (bid_size_0 - ask_size_0) / total_size_0
        else:
            imbalance = 0.0
        self.feature_buffer[feature_idx] = np.float32(imbalance)
        feature_idx += 1

        # 4. Depth imbalance (top 3)
        top3_bid = np.sum(bid_sizes[:3])
        top3_ask = np.sum(ask_sizes[:3])
        top3_total = top3_bid + top3_ask
        if top3_total > 0:
            depth_imb_3 = (top3_bid - top3_ask) / top3_total
        else:
            depth_imb_3 = 0.0
        self.feature_buffer[feature_idx] = np.float32(depth_imb_3)
        feature_idx += 1

        # 5. Depth imbalance (top 5)
        top5_bid = np.sum(bid_sizes[:5])
        top5_ask = np.sum(ask_sizes[:5])
        top5_total = top5_bid + top5_ask
        if top5_total > 0:
            depth_imb_5 = (top5_bid - top5_ask) / top5_total
        else:
            depth_imb_5 = 0.0
        self.feature_buffer[feature_idx] = np.float32(depth_imb_5)
        feature_idx += 1

        # 6. Bid slope
        if bid_prices[4] > 0:
            bid_slope = (bid_prices[4] - bid_prices[0]) / 4.0
            bid_slope_bps = 10000.0 * bid_slope / mid_price if mid_price > 0 else 0.0
        else:
            bid_slope_bps = 0.0
        self.feature_buffer[feature_idx] = np.float32(bid_slope_bps)
        feature_idx += 1

        # 7. Ask slope
        if ask_prices[4] > 0:
            ask_slope = (ask_prices[4] - ask_prices[0]) / 4.0
            ask_slope_bps = 10000.0 * ask_slope / mid_price if mid_price > 0 else 0.0
        else:
            ask_slope_bps = 0.0
        self.feature_buffer[feature_idx] = np.float32(ask_slope_bps)
        feature_idx += 1

        # Update historical buffers
        self._update_l2_history(l2_manager, spread_bps, imbalance, microprice_bps)

        # 8. Spread volatility (20-period)
        if l2_manager.l2_history_filled >= 5:
            window_size = min(l2_manager.l2_history_filled, 20)
            spread_vol = np.std(l2_manager.spread_history[:window_size])
        else:
            spread_vol = 0.0
        self.feature_buffer[feature_idx] = np.float32(spread_vol)
        feature_idx += 1

        # 9. Imbalance momentum
        if l2_manager.l2_history_filled >= 3:
            recent_imb = l2_manager.imbalance_history[: min(5, l2_manager.l2_history_filled)]
            if len(recent_imb) > 2:
                imb_momentum = np.mean(np.diff(recent_imb))
            else:
                imb_momentum = 0.0
        else:
            imb_momentum = 0.0
        self.feature_buffer[feature_idx] = np.float32(imb_momentum)
        feature_idx += 1

        # 10. Trade flow proxy
        flow_proxy = (
            (ask_size_0 - bid_size_0) / total_size_0 * spread_bps * 0.01
            if total_size_0 > 0
            else 0.0
        )
        self.feature_buffer[feature_idx] = np.float32(flow_proxy)
        feature_idx += 1

        # 11. Liquidity concentration (top 3 vs total)
        total_depth = np.sum(bid_sizes) + np.sum(ask_sizes)
        if total_depth > 0:
            concentration = top3_total / total_depth
        else:
            concentration = 0.0
        self.feature_buffer[feature_idx] = np.float32(concentration)
        feature_idx += 1

        return feature_idx

    def _compute_l2_approximations(
        self,
        current_bar: dict[str, float] | None,
        indicator_manager: IndicatorManager | None,
        feature_idx: int,
    ) -> int:
        """
        Compute L2/L3 approximations using OHLCV data.
        """
        # Use existing approximation methods
        if current_bar is not None:
            close = current_bar["close"]
            high = current_bar["high"]
            low = current_bar["low"]
            volume = current_bar["volume"]

            # Approximate spread from high-low range
            mid_approx = (high + low) * 0.5
            spread_approx = (high - low) / mid_approx * 10000.0 if mid_approx > 0 else 0.0

            # Price position within range
            if high > low:
                price_pos = (close - low) / (high - low)
                imbalance_approx = (price_pos - 0.5) * 2.0  # Convert to [-1, 1]
            else:
                imbalance_approx = 0.0

        else:
            spread_approx = 0.0
            imbalance_approx = 0.0

        # Fill approximation features
        approximation_features = [
            spread_approx,  # spread_bps
            0.0,  # microprice_bps
            imbalance_approx,  # imbalance
            imbalance_approx,  # depth_imb_3
            imbalance_approx,  # depth_imb_5
            0.0,  # bid_slope_bps
            0.0,  # ask_slope_bps
            spread_approx * 0.1,  # spread_vol
            0.0,  # imb_momentum
            imbalance_approx * spread_approx * 0.01,  # flow_proxy
            0.5,  # concentration
        ]

        for i, val in enumerate(approximation_features):
            self.feature_buffer[feature_idx + i] = np.float32(val)

        return feature_idx + len(approximation_features)

    def _update_l2_history(
        self,
        l2_manager: L2IndicatorManager,
        spread_bps: float,
        imbalance: float,
        microprice_bps: float,
    ) -> None:
        """
        Update L2 historical buffers.
        """
        i = l2_manager.l2_history_idx
        l2_manager.spread_history[i] = spread_bps
        l2_manager.imbalance_history[i] = imbalance
        l2_manager.microprice_history[i] = microprice_bps

        l2_manager.l2_history_idx = (i + 1) % 20
        l2_manager.l2_history_filled = min(l2_manager.l2_history_filled + 1, 20)


def create_l2_feature_engineer(config: FeatureConfig) -> L2FeatureEngineer:
    """
    Factory function to create L2-enhanced feature engineer.

    This is the recommended way to create a FeatureEngineer when L2/L3 features are
    required in hot path processing.

    """
    return L2FeatureEngineer(config)
