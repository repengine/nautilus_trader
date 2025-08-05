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
Base class for ML-driven trading strategies.

This module provides the foundation for building trading strategies that use ML signals
for decision making while following Nautilus Trader's architecture patterns and
performance requirements.

"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from decimal import Decimal
from typing import Any

from ml.actors.base import MLSignal
from ml.common.metrics import Counter
from ml.common.metrics import Histogram
from ml.config.base import MLStrategyConfig
from nautilus_trader.core.data import Data
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.data import DataType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.enums import TriggerType
from nautilus_trader.model.identifiers import ClientOrderId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.model.orders import StopMarketOrder
from nautilus_trader.model.position import Position
from nautilus_trader.trading.strategy import Strategy


# Prometheus metrics for monitoring
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


class BaseMLStrategy(Strategy, ABC):
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

        # Prometheus metrics
        self._signals_received_metric = ml_signals_received
        self._orders_submitted_metric = ml_trades_executed
        self._position_count_metric = Counter(
            "nautilus_ml_position_count",
            "Current number of open positions",
            ["strategy_id", "instrument"],
        )

    def on_start(self) -> None:
        """
        Initialize the strategy and subscribe to ML signals.

        This method sets up the strategy by subscribing to ML signals from the
        configured source and initializing any required state.

        """
        self.log.info(f"Starting {self.__class__.__name__}")

        # Subscribe to ML signals
        self.subscribe_data(
            data_type=DataType(MLSignal),
            client_id=None,  # Subscribe to all ML signals
        )

        # Subscribe to instruments for market data if needed
        self.subscribe_instrument(self._config.instrument_id)

        self.log.info(
            f"ML Strategy configured: instrument={self._config.instrument_id}, "
            f"position_size={self._config.position_size_pct:.1%}, "
            f"min_confidence={self._config.min_confidence}",
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

        # Check confidence threshold
        if signal.confidence < self._config.min_confidence:
            self.log.debug(
                f"Signal below confidence threshold: {signal.confidence:.3f} < "
                f"{self._config.min_confidence:.3f}",
            )
            return

        # Check position limits
        if self._active_positions >= self._config.max_positions:
            self.log.debug("Maximum positions reached, ignoring signal")
            return

        # Let concrete strategy decide on the signal
        self._process_ml_signal(signal)

    def _calculate_position_size(self) -> Quantity:
        """
        Calculate position size based on configuration and account balance.

        Returns
        -------
        Quantity
            The calculated position size.

        """
        instrument = self.cache.instrument(self._config.instrument_id)
        if instrument is None:
            self.log.warning("Instrument not found, using minimum position size")
            return Quantity.from_int(1)

        account = self.cache.account_for_venue(instrument.venue)
        if account is None:
            self.log.warning("No account found, using minimum position size")
            return Quantity.from_int(1)

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
                self.log.warning("No price data available, using minimum position size")
                return Quantity.from_int(1)

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
