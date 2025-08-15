"""
ML Trading Strategy implementation with multi-model support.

This module provides a production-ready ML strategy that can:
- Handle signals from multiple ML models
- Filter signals by model_id
- Aggregate signals for consensus trading
- Track model performance over time

"""

from __future__ import annotations

from typing import Any

import numpy as np

from ml.actors.base import MLSignal
from ml.strategies.base import BaseMLStrategy
from nautilus_trader.model.enums import OrderSide


class MLTradingStrategy(BaseMLStrategy):
    """
    Production ML trading strategy with multi-model support.

    This strategy can:
    - Filter signals by specific model IDs
    - Aggregate signals from multiple models
    - Track performance per model
    - Execute trades based on consensus or individual signals

    Configuration options (via MLStrategyConfig):
    - target_model_ids: List of model IDs to listen to (None = all)
    - aggregation_mode: "voting", "weighted_average", or None
    - required_models: Number of models needed for aggregation
    - time_window_ms: Time window for signal aggregation
    - conflict_resolution: How to handle conflicting signals
    - model_weights: Dict of model_id -> weight for weighted averaging
    - track_performance: Whether to track per-model performance

    """

    def _process_ml_signal(self, signal: MLSignal) -> None:
        """
        Process ML signal and execute trading logic.

        This implementation:
        1. Checks current position
        2. Determines trade direction from signal
        3. Manages position changes
        4. Tracks model performance if enabled

        Parameters
        ----------
        signal : MLSignal
            The ML signal to process (may be aggregated).

        """
        current_position = self._get_current_position()

        # Determine target side based on prediction
        # Using 0.5 as threshold for binary classification
        if signal.prediction > 0.5:
            target_side = OrderSide.BUY
            signal_direction = "LONG"
        else:
            target_side = OrderSide.SELL
            signal_direction = "SHORT"

        # Log signal details
        model_id = getattr(signal, "model_id", None) or signal.metadata.get("model_id", "unknown")
        self.log.info(
            f"Processing ML signal from {model_id}: "
            f"prediction={signal.prediction:.3f}, "
            f"confidence={signal.confidence:.3f}, "
            f"direction={signal_direction}",
        )

        # Check if we need to change position
        if current_position is None:
            # No position, enter new one
            self._enter_position(target_side, signal)

        elif self._should_reverse_position(current_position, target_side):
            # Position exists but signal suggests opposite direction
            self._reverse_position(current_position, target_side, signal)

        else:
            # Position aligns with signal
            self.log.debug(
                f"Position already {current_position.side.name}, aligns with signal direction {signal_direction}",
            )

            # Could add logic here to increase position size or adjust stops

    def _enter_position(self, side: OrderSide, signal: MLSignal) -> None:
        """
        Enter a new position based on ML signal.

        Parameters
        ----------
        side : OrderSide
            The side to enter (BUY or SELL).
        signal : MLSignal
            The signal triggering the entry.

        """
        quantity = self._calculate_position_size()
        if quantity is None:
            self.log.warning(
                f"Cannot enter position due to sizing failure for {signal.instrument_id}",
            )
            return

        # Place the order
        order_id = self._place_market_order(side, quantity)
        self._active_positions += 1

        # Track the signal that triggered this trade
        model_id = getattr(signal, "model_id", None) or signal.metadata.get("model_id", "unknown")
        self.log.info(
            f"Entering {side.name} position: {quantity} units based on signal from {model_id}",
        )

        # Store signal info for performance tracking
        if self.track_performance:
            self._track_trade_entry(model_id, signal, order_id)

    def _reverse_position(
        self,
        current_position: Any,
        target_side: OrderSide,
        signal: MLSignal,
    ) -> None:
        """
        Reverse an existing position based on ML signal.

        Parameters
        ----------
        current_position : Position
            The current position to reverse.
        target_side : OrderSide
            The new target side.
        signal : MLSignal
            The signal triggering the reversal.

        """
        # Close current position first
        close_side = OrderSide.SELL if current_position.side.name == "LONG" else OrderSide.BUY

        self.log.info(f"Reversing position from {current_position.side.name} to {target_side.name}")

        # Close existing position
        self._place_market_order(
            close_side,
            current_position.quantity,
            reduce_only=True,
        )

        # Open new position in opposite direction
        quantity = self._calculate_position_size()
        if quantity is None:
            self.log.warning("Closed position but cannot open new one due to sizing failure")
            return

        order_id = self._place_market_order(target_side, quantity)

        # Track the reversal
        model_id = getattr(signal, "model_id", None) or signal.metadata.get("model_id", "unknown")
        if self.track_performance:
            self._track_trade_entry(model_id, signal, order_id)

    def _should_reverse_position(
        self,
        current_position: Any,
        target_side: OrderSide,
    ) -> bool:
        """
        Check if position should be reversed.

        Parameters
        ----------
        current_position : Position
            The current position.
        target_side : OrderSide
            The target side from signal.

        Returns
        -------
        bool
            True if position should be reversed.

        """
        return bool(
            (current_position.side.name == "LONG" and target_side == OrderSide.SELL)
            or (current_position.side.name == "SHORT" and target_side == OrderSide.BUY),
        )

    def _track_trade_entry(
        self,
        model_id: str,
        signal: MLSignal,
        order_id: Any,
    ) -> None:
        """
        Track trade entry for performance analysis.

        Parameters
        ----------
        model_id : str
            The model that generated the signal.
        signal : MLSignal
            The signal that triggered the trade.
        order_id : ClientOrderId
            The order ID for tracking.

        """
        # Store mapping of order_id to model_id for later performance tracking
        if not hasattr(self, "_order_to_model"):
            self._order_to_model = {}

        self._order_to_model[str(order_id)] = {
            "model_id": model_id,
            "signal": signal,
            "entry_time": self.clock.timestamp_ns(),
        }

    def on_order_filled(self, event: Any) -> None:
        """
        Handle order fills and track model performance.

        Parameters
        ----------
        event : OrderFilled
            The order filled event.

        """
        super().on_order_filled(event)

        # Track performance if this was a closing order
        if hasattr(self, "_order_to_model") and self.track_performance:
            order_id = str(event.client_order_id)

            # Check if this is a tracked order
            if order_id in self._order_to_model:
                order_info = self._order_to_model[order_id]
                model_id = order_info["model_id"]

                # Calculate P&L (simplified - real implementation would use actual fills)
                # This is just for demonstration
                if event.order_side.name == "SELL":
                    # Assuming we're closing a long position
                    pnl = float(event.last_px.as_double()) - float(event.avg_px.as_double())
                else:
                    # Assuming we're closing a short position
                    pnl = float(event.avg_px.as_double()) - float(event.last_px.as_double())

                # Update model performance
                self._update_model_performance(model_id, pnl)

                self.log.info(f"Trade completed for model {model_id}: P&L = {pnl:.2f}")

                # Clean up tracking
                del self._order_to_model[order_id]


class MultiModelMLStrategy(MLTradingStrategy):
    """
    Extended ML strategy specifically designed for multi-model aggregation.

    This strategy extends the base MLTradingStrategy with additional
    features for handling multiple models:
    - Weighted consensus decisions
    - Model performance-based dynamic weighting
    - Conflict resolution strategies

    """

    def __init__(self, config: Any) -> None:
        """
        Initialize multi-model strategy.

        Parameters
        ----------
        config : MLStrategyConfig
            The strategy configuration.

        """
        super().__init__(config)

        # Enable performance tracking for dynamic weighting
        self.track_performance = True
        self.use_dynamic_weights = getattr(config, "use_dynamic_weights", False)

    def _get_dynamic_model_weights(self) -> dict[str, float]:
        """
        Calculate dynamic weights based on model performance.

        Returns
        -------
        dict[str, float]
            Model weights based on historical performance.

        """
        if not self._model_performance:
            # No performance data yet, use equal weights
            return {}

        weights = {}
        for model_id, perf in self._model_performance.items():
            # Weight based on accuracy and total profit
            accuracy = perf.get("accuracy", 0.5)
            total_profit = perf.get("total_profit", 0.0)
            total_trades = perf.get("total_trades", 1)

            # Combine accuracy and profit per trade
            profit_per_trade = total_profit / max(total_trades, 1)

            # Simple weighting formula (can be customized)
            weight = accuracy * (1.0 + np.tanh(profit_per_trade / 100.0))
            weights[model_id] = max(weight, 0.1)  # Minimum weight of 0.1

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def _aggregate_signal(self, signal: MLSignal) -> None:
        """
        Override aggregation to use dynamic weights if enabled.

        Parameters
        ----------
        signal : MLSignal
            The signal to aggregate.

        """
        # Update model weights if using dynamic weighting
        if self.use_dynamic_weights:
            self.model_weights = self._get_dynamic_model_weights()

        # Call parent aggregation
        super()._aggregate_signal(signal)
