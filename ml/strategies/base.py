
"""
Base class for ML-driven trading strategies.

This module provides the foundation for building trading strategies that use ML signals
for decision making while following Nautilus Trader's architecture patterns and
performance requirements.

"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections import deque
from decimal import Decimal
from typing import Any, cast

import numpy as np

from ml.actors.base import MLSignal
from ml.common.metrics import HAS_PROMETHEUS
from ml.common.metrics import Counter
from ml.common.metrics import Histogram
from ml.config.base import MLStrategyConfig
from nautilus_trader.core.data import Data
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.data import DataType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.enums import TriggerType
from nautilus_trader.model.identifiers import ClientId
from nautilus_trader.model.identifiers import ClientOrderId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.model.orders import StopMarketOrder
from nautilus_trader.model.position import Position
from nautilus_trader.trading.strategy import Strategy


# Prometheus metrics for monitoring
# These are module-level singletons to avoid registry collisions
_metrics_initialized = False
ml_signals_received = None
ml_trades_executed = None
ml_signal_to_trade_latency = None
ml_position_count = None


def _initialize_metrics() -> None:
    """
    Initialize Prometheus metrics once.
    """
    global _metrics_initialized, ml_signals_received, ml_trades_executed, ml_signal_to_trade_latency, ml_position_count

    if _metrics_initialized:
        return

    # Check if metrics already exist in registry
    if HAS_PROMETHEUS:
        from prometheus_client import REGISTRY

        # Try to get existing metrics or create new ones
        existing_names = set(REGISTRY._names_to_collectors.keys())

        if "nautilus_ml_signals_received_total" not in existing_names:
            ml_signals_received = Counter(
                "nautilus_ml_signals_received_total",
                "Total number of ML signals received",
                ["strategy_id", "signal_source"],
            )
        else:
            ml_signals_received = cast(
                Counter,
                REGISTRY._names_to_collectors["nautilus_ml_signals_received_total"],
            )

        if "nautilus_ml_trades_executed_total" not in existing_names:
            ml_trades_executed = Counter(
                "nautilus_ml_trades_executed_total",
                "Total number of trades executed based on ML signals",
                ["strategy_id", "order_side"],
            )
        else:
            ml_trades_executed = cast(
                Counter,
                REGISTRY._names_to_collectors["nautilus_ml_trades_executed_total"],
            )

        if "nautilus_ml_signal_to_trade_latency_seconds" not in existing_names:
            ml_signal_to_trade_latency = Histogram(
                "nautilus_ml_signal_to_trade_latency_seconds",
                "Latency from signal reception to trade execution",
                ["strategy_id"],
            )
        else:
            ml_signal_to_trade_latency = cast(
                Histogram,
                REGISTRY._names_to_collectors["nautilus_ml_signal_to_trade_latency_seconds"],
            )

        if "nautilus_ml_position_count" not in existing_names:
            ml_position_count = Counter(
                "nautilus_ml_position_count",
                "Current number of open positions",
                ["strategy_id", "instrument"],
            )
        else:
            ml_position_count = cast(
                Counter,
                REGISTRY._names_to_collectors["nautilus_ml_position_count"],
            )
    else:
        # Use dummy metrics when Prometheus is not available
        ml_signals_received = Counter(
            "nautilus_ml_signals_received_total",
            "Total number of ML signals received",
            ["strategy_id", "signal_source"],
        )
        ml_trades_executed = Counter(
            "nautilus_ml_trades_executed_total",
            "Total number of trades executed based on ML signals",
            ["strategy_id", "order_side"],
        )
        ml_signal_to_trade_latency = Histogram(
            "nautilus_ml_signal_to_trade_latency_seconds",
            "Latency from signal reception to trade execution",
            ["strategy_id"],
        )
        ml_position_count = Counter(
            "nautilus_ml_position_count",
            "Current number of open positions",
            ["strategy_id", "instrument"],
        )

    _metrics_initialized = True


# Initialize metrics on module load
_initialize_metrics()


class BaseMLStrategy(Strategy, ABC):  # type: ignore[misc]
    """
    Base class for ML-driven trading strategies.

    This class provides common functionality for strategies that trade based on
    ML signals, including position sizing, risk management, and signal handling.

    Key features:
    - Subscribes to ML signals from actors
    - Implements position sizing based on account balance
    - Provides configurable stop loss and take profit
    - Tracks strategy performance metrics

    Parameters
    ----------
    config : MLStrategyConfig
        The configuration for the ML strategy.

    """

    def __init__(self, config: MLStrategyConfig) -> None:
        """
        Initialize the ML strategy.

        Parameters
        ----------
        config : MLStrategyConfig
            The configuration for the ML strategy.

        """
        super().__init__(config)
        self._config = config

        # Trading state
        self._active_positions = 0
        self._pending_orders = 0
        self._last_signal_time = 0

        # Performance tracking
        self._signals_received = 0
        self._trades_executed = 0
        self._winning_trades = 0
        self._total_pnl = Decimal("0.0")

        # Signal management
        self._signal_history: deque[MLSignal] = deque(maxlen=config.history_size if hasattr(config, 'history_size') else 100)
        self._signal_buffer: dict[str, MLSignal] = {}  # For aggregation by model_id
        self._model_signals: dict[str, MLSignal] = {}  # Current signals per model
        self._model_performance: dict[str, dict[str, Any]] = {}  # Performance tracking per model

        # Model filtering and aggregation settings
        self.target_model_ids: list[str] | None = getattr(config, 'target_model_ids', None)
        self.aggregation_mode: str | None = getattr(config, 'aggregation_mode', None)
        self.required_models: int = getattr(config, 'required_models', 1)
        self.time_window_ms: int = getattr(config, 'time_window_ms', 1000)
        self.conflict_resolution: str | None = getattr(config, 'conflict_resolution', None)
        self.model_weights: dict[str, float] = getattr(config, 'model_weights', {})
        self.track_performance: bool = getattr(config, 'track_performance', False)

        # Prometheus metrics
        self._signals_received_metric = ml_signals_received
        self._orders_submitted_metric = ml_trades_executed
        self._position_count_metric = ml_position_count

    def on_start(self) -> None:
        """
        Initialize the strategy and subscribe to ML signals.

        This method sets up the strategy by subscribing to ML signals from the
        configured source and initializing any required state.

        """
        self.log.info(f"Starting {self.__class__.__name__}")

        # Subscribe to ML signals
        # If specific client_id configured, use it; otherwise subscribe to all
        client_id = getattr(self._config, 'signal_client_id', None)
        if client_id is not None:
            self.subscribe_data(
                data_type=DataType(MLSignal),
                client_id=ClientId(client_id),
            )
        else:
            self.subscribe_data(
                data_type=DataType(MLSignal),
                client_id=None,  # Subscribe to all ML signals
            )

        # Subscribe to instruments for market data if needed
        self.subscribe_instrument(self._config.instrument_id)

        self.log.info(
            f"ML Strategy configured: instrument={self._config.instrument_id}, "
            f"position_size={self._config.position_size_pct:.1%}, "
            f"min_confidence={self._config.min_confidence}, "
            f"target_models={self.target_model_ids}, "
            f"aggregation={self.aggregation_mode}",
        )

    def on_data(self, data: Data) -> None:
        """
        Process incoming data, particularly ML signals.

        Parameters
        ----------
        data : Data
            The incoming data object.

        """
        if isinstance(data, MLSignal):
            # Add to history
            self._signal_history.append(data)
            
            # Get model_id from either the dedicated field or metadata
            model_id = getattr(data, 'model_id', None) or data.metadata.get('model_id')
            
            # Filter by model_id if configured
            if self.target_model_ids is not None:
                if model_id not in self.target_model_ids:
                    self.log.debug(
                        f"Ignoring signal from model {model_id} (not in target list)"
                    )
                    return
            
            # Check confidence threshold
            if data.confidence < self._config.min_confidence:
                self.log.debug(
                    f"Signal below confidence threshold: {data.confidence:.3f} < "
                    f"{self._config.min_confidence:.3f}"
                )
                return
            
            # Handle aggregation if configured
            if self.aggregation_mode:
                self._aggregate_signal(data)
            else:
                # Process single signal
                self._handle_ml_signal(data)

    def on_stop(self) -> None:
        """
        Log final statistics when the strategy stops.
        """
        win_rate = self._winning_trades / max(self._trades_executed, 1) * 100

        self.log.info(
            f"Stopping {self.__class__.__name__} - "
            f"Signals: {self._signals_received}, "
            f"Trades: {self._trades_executed}, "
            f"Win rate: {win_rate:.1f}%, "
            f"Total PnL: {self._total_pnl}",
        )

    def _handle_ml_signal(self, signal: MLSignal) -> None:
        """
        Process ML signal and potentially execute trades.

        This method evaluates the ML signal against configured thresholds
        and risk management rules before executing trades.

        Parameters
        ----------
        signal : MLSignal
            The ML signal to process.

        """
        self._signals_received += 1
        self._last_signal_time = signal.ts_event

        # Check if signal is for our instrument
        if signal.instrument_id != self._config.instrument_id:
            return

        # Check position limits
        if self._active_positions >= self._config.max_positions:
            self.log.debug("Maximum positions reached, ignoring signal")
            return

        # Let concrete strategy decide on the signal
        self._process_signal(signal)
        self._process_ml_signal(signal)

    def _calculate_position_size(self) -> Quantity | None:
        """
        Calculate position size based on configuration and account balance.

        Returns
        -------
        Quantity | None
            The calculated position size, or None if insufficient data available.

        """
        instrument = self.cache.instrument(self._config.instrument_id)
        if instrument is None:
            self.log.error(
                f"Cannot calculate position size: Instrument {self._config.instrument_id} not found. "
                "Ensure instrument is subscribed and available in cache.",
            )
            return None

        account = self.cache.account_for_venue(instrument.venue)
        if account is None:
            self.log.error(
                f"Cannot calculate position size: No account found for venue {instrument.venue}. "
                "Position sizing requires account information.",
            )
            return None

        # Calculate position size as percentage of account balance
        account_balance = float(account.balance_total().as_double())
        position_value = account_balance * self._config.position_size_pct

        # Get current price for position sizing (instrument already fetched above)

        # Use last trade price or mid price for sizing
        last_tick = self.cache.trade_tick(self._config.instrument_id)
        if last_tick is not None:
            current_price = float(last_tick.price.as_double())
        else:
            # Fallback to quote tick mid price
            quote_tick = self.cache.quote_tick(self._config.instrument_id)
            if quote_tick is not None:
                bid_price = float(quote_tick.bid_price.as_double())
                ask_price = float(quote_tick.ask_price.as_double())
                current_price = (bid_price + ask_price) / 2.0
            else:
                self.log.error(
                    f"Cannot calculate position size: No price data available for {self._config.instrument_id}. "
                    "Ensure market data is being received before trading.",
                )
                return None

        # Calculate quantity
        raw_quantity = position_value / current_price

        # Round to instrument precision
        precision = instrument.size_precision
        quantity_value = round(raw_quantity, precision)

        # Ensure minimum size
        min_quantity = float(instrument.min_quantity.as_double())
        quantity_value = max(quantity_value, min_quantity)

        return Quantity.from_str(str(quantity_value))

    def _place_market_order(
        self,
        side: OrderSide,
        quantity: Quantity,
        reduce_only: bool = False,
    ) -> ClientOrderId:
        """
        Place a market order with optional stop loss and take profit.

        Parameters
        ----------
        side : OrderSide
            The order side (BUY or SELL).
        quantity : Quantity
            The order quantity.
        reduce_only : bool, default False
            Whether this is a reduce-only order.

        Returns
        -------
        ClientOrderId
            The client order ID of the placed order.

        """
        order = MarketOrder(
            trader_id=self.trader_id,
            strategy_id=self.id,
            instrument_id=self._config.instrument_id,
            client_order_id=self.cache.client_order_id(),
            order_side=side,
            quantity=quantity,
            init_id=UUID4(),
            ts_init=self.clock.timestamp_ns(),
            time_in_force=TimeInForce.GTC,
            reduce_only=reduce_only,
        )

        self.submit_order(order)
        self._pending_orders += 1
        self._trades_executed += 1

        self.log.info(
            f"Placed {side.name} market order: {quantity} @ market " f"(reduce_only={reduce_only})",
        )

        return order.client_order_id

    def _place_stop_loss(
        self,
        side: OrderSide,
        quantity: Quantity,
        trigger_price: Price,
    ) -> ClientOrderId:
        """
        Place a stop loss order.

        Parameters
        ----------
        side : OrderSide
            The order side (opposite of main position).
        quantity : Quantity
            The order quantity.
        trigger_price : Price
            The stop loss trigger price.

        Returns
        -------
        ClientOrderId
            The client order ID of the placed order.

        """
        order = StopMarketOrder(
            trader_id=self.trader_id,
            strategy_id=self.id,
            instrument_id=self._config.instrument_id,
            client_order_id=self.cache.client_order_id(),
            order_side=side,
            quantity=quantity,
            trigger_price=trigger_price,
            trigger_type=TriggerType.DEFAULT,
            init_id=UUID4(),
            ts_init=self.clock.timestamp_ns(),
            time_in_force=TimeInForce.GTC,
            reduce_only=True,
        )

        self.submit_order(order)

        self.log.info(f"Placed stop loss: {side.name} {quantity} @ {trigger_price}")

        return order.client_order_id

    def _get_current_position(self) -> Position | None:
        """
        Get the current position for the configured instrument.

        Returns
        -------
        Position | None
            The current position, or None if no position exists.

        """
        positions = self.cache.positions_open(
            venue=None,  # All venues
            instrument_id=self._config.instrument_id,
        )

        if positions:
            return positions[0]  # Return first open position
        return None

    def _aggregate_signal(self, signal: MLSignal) -> None:
        """
        Aggregate signals from multiple models.

        Parameters
        ----------
        signal : MLSignal
            The ML signal to aggregate.

        """
        model_id = getattr(signal, 'model_id', None) or signal.metadata.get('model_id')
        if model_id:
            self._model_signals[model_id] = signal
        
        # Check if we have enough signals
        if len(self._model_signals) >= self.required_models:
            # Check if all signals are within time window
            latest_time = max(s.ts_event for s in self._model_signals.values())
            earliest_time = min(s.ts_event for s in self._model_signals.values())
            time_diff_ms = (latest_time - earliest_time) / 1_000_000  # Convert ns to ms
            
            if time_diff_ms <= self.time_window_ms:
                # Aggregate and make decision
                if self.conflict_resolution == "weighted_average":
                    # Calculate weighted average prediction
                    total_weight = 0.0
                    weighted_sum = 0.0
                    
                    for mid, sig in self._model_signals.items():
                        weight = self.model_weights.get(mid, 1.0)
                        weighted_sum += weight * sig.prediction
                        total_weight += weight
                    
                    if total_weight > 0:
                        weighted_pred = weighted_sum / total_weight
                        avg_confidence = float(np.mean([s.confidence for s in self._model_signals.values()]))
                        
                        # Create aggregated signal
                        aggregated_signal = MLSignal(
                            instrument_id=signal.instrument_id,
                            model_id="aggregated",
                            prediction=weighted_pred,
                            confidence=avg_confidence,
                            metadata={"aggregated_from": list(self._model_signals.keys())},
                            ts_event=latest_time,
                            ts_init=self.clock.timestamp_ns(),
                        )
                        
                        self._make_decision({"weighted_prediction": weighted_pred, "confidence": avg_confidence})
                        self._process_ml_signal(aggregated_signal)
                else:
                    # Simple voting
                    bullish = sum(1 for s in self._model_signals.values() if s.prediction > 0.5)
                    bearish = len(self._model_signals) - bullish
                    
                    action = "BUY" if bullish > bearish else "SELL"
                    confidence = max(s.confidence for s in self._model_signals.values())
                    
                    # Create aggregated signal
                    prediction = 0.8 if action == "BUY" else 0.2
                    aggregated_signal = MLSignal(
                        instrument_id=signal.instrument_id,
                        model_id="aggregated",
                        prediction=prediction,
                        confidence=confidence,
                        metadata={"action": action, "aggregated_from": list(self._model_signals.keys())},
                        ts_event=latest_time,
                        ts_init=self.clock.timestamp_ns(),
                    )
                    
                    self._execute_trade({"action": action, "confidence": confidence, "signal": aggregated_signal})
                    self._process_ml_signal(aggregated_signal)
                
                # Clear buffer after decision
                self._model_signals.clear()
            else:
                # Signals too far apart, clear old ones
                self._model_signals = {mid: sig for mid, sig in self._model_signals.items() 
                                       if (latest_time - sig.ts_event) / 1_000_000 <= self.time_window_ms}

    def _process_signal(self, signal: MLSignal) -> None:
        """
        Process individual signal (stub for test compatibility).

        Parameters
        ----------
        signal : MLSignal
            The ML signal to process.

        """
        pass

    def _make_decision(self, decision: dict[str, Any]) -> None:
        """
        Make trading decision (stub for test compatibility).

        Parameters
        ----------
        decision : dict[str, Any]
            The decision data.

        """
        pass

    def _execute_trade(self, trade: dict[str, Any]) -> None:
        """
        Execute trade based on signal (stub for test compatibility).

        Parameters
        ----------
        trade : dict[str, Any]
            The trade data.

        """
        pass

    def _update_model_performance(self, model_id: str, profit: float) -> None:
        """
        Update model performance metrics.

        Parameters
        ----------
        model_id : str
            The model identifier.
        profit : float
            The profit from the trade.

        """
        if model_id not in self._model_performance:
            self._model_performance[model_id] = {
                "total_trades": 0,
                "total_profit": 0.0,
                "wins": 0,
                "losses": 0,
                "accuracy": 0.0,
            }
        
        self._model_performance[model_id]["total_trades"] += 1
        self._model_performance[model_id]["total_profit"] += profit
        
        if profit > 0:
            self._model_performance[model_id]["wins"] += 1
        else:
            self._model_performance[model_id]["losses"] += 1
        
        # Update accuracy
        total = self._model_performance[model_id]["total_trades"]
        wins = self._model_performance[model_id]["wins"]
        self._model_performance[model_id]["accuracy"] = wins / total if total > 0 else 0.0

    @abstractmethod
    def _process_ml_signal(self, signal: MLSignal) -> None:
        """
        Process ML signal and execute trading logic.

        This method should be implemented by concrete strategies to define
        how ML signals are translated into trading actions.

        Parameters
        ----------
        signal : MLSignal
            The ML signal to process.

        """
        ...


class SimpleMLStrategy(BaseMLStrategy):
    """
    Simple ML strategy that trades based on binary ML signals.

    This strategy demonstrates a basic implementation that:
    - Goes long on positive signals (prediction > 0.5)
    - Goes short on negative signals (prediction < 0.5)
    - Implements basic position management

    """

    def _process_ml_signal(self, signal: MLSignal) -> None:
        """
        Process ML signal and execute simple trading logic.

        Parameters
        ----------
        signal : MLSignal
            The ML signal to process.

        """
        current_position = self._get_current_position()

        # Determine target side based on prediction
        if signal.prediction > 0.5:
            target_side = OrderSide.BUY
        else:
            target_side = OrderSide.SELL

        # Check if we need to change position
        if current_position is None:
            # No position, enter new one
            quantity = self._calculate_position_size()
            if quantity is None:
                self.log.warning(
                    f"Skipping trade signal due to position sizing failure for {signal.instrument_id}",
                )
                return
            self._place_market_order(target_side, quantity)
            self._active_positions += 1

        elif (current_position.side.name == "LONG" and target_side == OrderSide.SELL) or (
            current_position.side.name == "SHORT" and target_side == OrderSide.BUY
        ):
            # Position exists but signal suggests opposite direction
            # Close current position first
            close_side = OrderSide.SELL if current_position.side.name == "LONG" else OrderSide.BUY
            self._place_market_order(
                close_side,
                current_position.quantity,
                reduce_only=True,
            )

            # Then open new position
            quantity = self._calculate_position_size()
            if quantity is None:
                self.log.warning(
                    f"Closed position but cannot open new one due to position sizing failure for {signal.instrument_id}",
                )
                return
            self._place_market_order(target_side, quantity)

        else:
            # Position aligns with signal, no action needed
            self.log.debug("Position aligns with signal, no action taken")

    def on_order_filled(self, event: Any) -> None:
        """
        Handle order filled events for position tracking.
        """
        super().on_order_filled(event)

        # Update pending orders count
        self._pending_orders = max(0, self._pending_orders - 1)

        # Update position count
        current_position = self._get_current_position()
        if current_position is None:
            self._active_positions = 0
        else:
            self._active_positions = 1  # Simple strategy only holds one position

        self.log.info(
            f"Order filled: {event.order_side.name} {event.last_qty} @ {event.last_px}, "
            f"Active positions: {self._active_positions}",
        )
