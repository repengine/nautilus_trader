"""
Order submission component for MLTradingStrategy decomposition.

This component extracts order creation and submission logic from BaseMLStrategy
following the Protocol-First Interface Design pattern.

Responsibility:
- Create and submit market orders
- Create and submit smart orders using OrderExecutor
- Place stop loss orders
- Handle circuit breaker backpressure
- Track order metrics

"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:

    from nautilus_trader.model.identifiers import ClientOrderId
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.objects import Price
    from nautilus_trader.model.objects import Quantity

    from ml.actors.base import MLSignal
    from nautilus_trader.model.enums import OrderSide


@runtime_checkable
class CircuitBreakerProtocol(Protocol):
    """Protocol for circuit breaker functionality."""

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        ...

    def record_success(self) -> None:
        """Record a successful execution."""
        ...

    def record_failure(self) -> None:
        """Record a failed execution."""
        ...


@runtime_checkable
class CacheProtocol(Protocol):
    """Protocol for cache access."""

    def instrument(self, instrument_id: Any) -> Any:
        """Get instrument by ID."""
        ...

    def quote_tick(self, instrument_id: Any) -> Any:
        """Get latest quote tick for instrument."""
        ...

    def client_order_id(self) -> Any:
        """Generate a new client order ID."""
        ...


@runtime_checkable
class OrderExecutorProtocol(Protocol):
    """Protocol for order execution."""

    def create_order(
        self,
        side: Any,
        quantity: Any,
        signal: Any,
        market_state: dict[str, float],
        instrument: Any,
        *,
        trader_id: Any = None,
        strategy_id: Any = None,
        client_order_id: Any = None,
        init_id: Any = None,
        ts_init: int | None = None,
    ) -> Any:
        """Create an order based on signal and market conditions."""
        ...


@runtime_checkable
class PerformanceTrackerProtocol(Protocol):
    """Protocol for performance tracking."""

    def record_order(self, order: Any, signal: Any) -> None:
        """Record an order placement."""
        ...


@runtime_checkable
class LoggerProtocol(Protocol):
    """Protocol for logging interface."""

    def debug(self, *args: object, **kwargs: object) -> None:
        """Log debug message."""
        ...

    def info(self, *args: object, **kwargs: object) -> None:
        """Log info message."""
        ...

    def warning(self, *args: object, **kwargs: object) -> None:
        """Log warning message."""
        ...

    def error(self, *args: object, **kwargs: object) -> None:
        """Log error message."""
        ...


class _NoOpLogger:
    """No-op logger for when no logger is provided."""

    def debug(self, *args: object, **kwargs: object) -> None:
        """No-op debug."""
        del args, kwargs

    def info(self, *args: object, **kwargs: object) -> None:
        """No-op info."""
        del args, kwargs

    def warning(self, *args: object, **kwargs: object) -> None:
        """No-op warning."""
        del args, kwargs

    def error(self, *args: object, **kwargs: object) -> None:
        """No-op error."""
        del args, kwargs


class OrderSubmissionComponent:
    """
    Manages order creation and submission with circuit breaker protection.

    This component is extracted from BaseMLStrategy to provide focused,
    testable order submission functionality following the facade pattern.

    Responsibilities:
    - Create and submit market orders
    - Create and submit smart orders using OrderExecutor
    - Place stop loss orders
    - Handle circuit breaker backpressure
    - Track order metrics

    Parameters
    ----------
    strategy_id : str
        Strategy identifier for metrics and logging.
    order_executor : OrderExecutorProtocol | None, optional
        Smart order executor for advanced order types.
    circuit_breaker : CircuitBreakerProtocol | None, optional
        Circuit breaker for backpressure handling.
    performance_tracker : PerformanceTrackerProtocol | None, optional
        Performance tracker for order recording.
    cache : CacheProtocol | None, optional
        Cache for instrument/quote data access.
    submit_order_callback : Callable | None, optional
        Callback function to submit orders to the trading system.
    log : LoggerProtocol | None, optional
        Logger instance for debug output.
    instrument_id : Any, optional
        Target instrument ID for orders.
    trader_id : Any, optional
        Trader ID for order creation.
    clock : Any, optional
        Clock for timestamp generation.
    orders_submitted_metric : Any, optional
        Prometheus metric for orders submitted counter.

    Examples
    --------
    >>> component = OrderSubmissionComponent(
    ...     strategy_id="my_strategy",
    ...     order_executor=executor,
    ...     circuit_breaker=breaker,
    ...     cache=cache,
    ...     submit_order_callback=strategy.submit_order,
    ... )
    >>> order_id = component.place_market_order(
    ...     instrument_id=instrument_id,
    ...     side=OrderSide.BUY,
    ...     quantity=Quantity.from_str("10.0"),
    ... )

    """

    def __init__(
        self,
        strategy_id: str,
        order_executor: OrderExecutorProtocol | None = None,
        circuit_breaker: CircuitBreakerProtocol | None = None,
        performance_tracker: PerformanceTrackerProtocol | None = None,
        cache: CacheProtocol | None = None,
        submit_order_callback: Callable[[Any], None] | None = None,
        log: Any = None,
        instrument_id: Any = None,
        trader_id: Any = None,
        clock: Any = None,
        orders_submitted_metric: Any = None,
    ) -> None:
        """Initialize the order submission component."""
        self._strategy_id = strategy_id
        self._order_executor = order_executor
        self._circuit_breaker = circuit_breaker
        self._performance_tracker = performance_tracker
        self._cache = cache
        self._submit_order_callback = submit_order_callback
        self._log = log if log is not None else _NoOpLogger()
        self._instrument_id = instrument_id
        self._trader_id = trader_id
        self._clock = clock
        self._orders_submitted_metric = orders_submitted_metric

        # Internal state
        self._dry_run_trades: int = 0
        self._trades_executed: int = 0
        self._pending_orders: int = 0

    # -------------------------------------------------------------------------
    # Public Properties
    # -------------------------------------------------------------------------

    @property
    def strategy_id(self) -> str:
        """Get the strategy ID."""
        return self._strategy_id

    @property
    def order_executor(self) -> OrderExecutorProtocol | None:
        """Get the order executor."""
        return self._order_executor

    @property
    def circuit_breaker(self) -> CircuitBreakerProtocol | None:
        """Get the circuit breaker."""
        return self._circuit_breaker

    @property
    def dry_run_trades(self) -> int:
        """Get the count of dry run trades."""
        return self._dry_run_trades

    @property
    def trades_executed(self) -> int:
        """Get the count of trades executed."""
        return self._trades_executed

    @property
    def pending_orders(self) -> int:
        """Get the count of pending orders."""
        return self._pending_orders

    # -------------------------------------------------------------------------
    # Configuration Update Methods
    # -------------------------------------------------------------------------

    def update_config(
        self,
        *,
        instrument_id: Any = None,
        cache: CacheProtocol | None = None,
        submit_order_callback: Callable[[Any], None] | None = None,
        trader_id: Any = None,
        clock: Any = None,
    ) -> None:
        """
        Update component configuration.

        Parameters
        ----------
        instrument_id : Any, optional
            Updated target instrument ID.
        cache : CacheProtocol | None, optional
            Updated cache instance.
        submit_order_callback : Callable | None, optional
            Updated submit order callback.
        trader_id : Any, optional
            Updated trader ID.
        clock : Any, optional
            Updated clock instance.

        """
        if instrument_id is not None:
            self._instrument_id = instrument_id
        if cache is not None:
            self._cache = cache
        if submit_order_callback is not None:
            self._submit_order_callback = submit_order_callback
        if trader_id is not None:
            self._trader_id = trader_id
        if clock is not None:
            self._clock = clock

    # -------------------------------------------------------------------------
    # Circuit Breaker Check
    # -------------------------------------------------------------------------

    def _check_circuit_breaker(self) -> bool:
        """
        Check if circuit breaker allows execution.

        Returns
        -------
        bool
            True if execution is allowed, False if suppressed.

        """
        try:
            cb = self._circuit_breaker
            if cb is not None and not cb.can_execute():
                self._dry_run_trades += 1
                self._log.info(
                    "Order submission suppressed by circuit breaker (DRY-RUN)",
                )
                return False
        except Exception as breaker_exc:
            self._log.debug(
                "ml_strategy.order_breaker_check_failed",
                strategy_id=self._strategy_id,
                exc_info=True,
                error=str(breaker_exc),
            )
        return True

    # -------------------------------------------------------------------------
    # Market Order Submission
    # -------------------------------------------------------------------------

    def place_market_order(
        self,
        instrument_id: InstrumentId,
        side: OrderSide,
        quantity: Quantity,
        reduce_only: bool = False,
    ) -> ClientOrderId | None:
        """
        Place a market order with optional circuit breaker protection.

        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument to trade.
        side : OrderSide
            The order side (BUY or SELL).
        quantity : Quantity
            The order quantity.
        reduce_only : bool, default False
            Whether this is a reduce-only order.

        Returns
        -------
        ClientOrderId | None
            The client order ID of the placed order, or None if suppressed.

        Examples
        --------
        >>> order_id = component.place_market_order(
        ...     instrument_id=InstrumentId.from_str("EURUSD.SIM"),
        ...     side=OrderSide.BUY,
        ...     quantity=Quantity.from_str("10.0"),
        ... )

        """
        from nautilus_trader.core.uuid import UUID4
        from nautilus_trader.model.enums import TimeInForce
        from nautilus_trader.model.orders import MarketOrder

        # Check circuit breaker
        if not self._check_circuit_breaker():
            # Return a fresh client order id without submitting
            if self._cache is not None:
                return self._cache.client_order_id()
            return None

        if self._cache is None:
            self._log.error("Cannot place market order: Cache not available")
            return None

        # Get timestamp
        ts_init = self._clock.timestamp_ns() if self._clock else time.time_ns()

        # Resolve trader_id and strategy_id to proper types
        from nautilus_trader.model.identifiers import StrategyId
        from nautilus_trader.model.identifiers import TraderId

        trader_id = self._trader_id
        if isinstance(trader_id, str):
            trader_id = TraderId(trader_id)

        strategy_id = self._strategy_id
        if isinstance(strategy_id, str):
            strategy_id = StrategyId(strategy_id)

        # Create market order
        order = MarketOrder(
            trader_id=trader_id,
            strategy_id=strategy_id,
            instrument_id=instrument_id,
            client_order_id=self._cache.client_order_id(),
            order_side=side,
            quantity=quantity,
            init_id=UUID4(),
            ts_init=ts_init,
            time_in_force=TimeInForce.GTC,
            reduce_only=reduce_only,
        )

        # Submit order
        if self._submit_order_callback is not None:
            self._submit_order_callback(order)

        self._pending_orders += 1
        self._trades_executed += 1

        # Record orders submitted metric
        if self._orders_submitted_metric is not None:
            try:
                self._orders_submitted_metric.labels(
                    strategy_id=str(self._strategy_id),
                    order_side=side.name,
                ).inc()
            except Exception:
                pass  # Metrics are non-critical

        self._log.info(
            f"Placed {side.name} market order: {quantity} @ market "
            f"(reduce_only={reduce_only})",
        )

        return order.client_order_id

    # -------------------------------------------------------------------------
    # Smart Order Submission
    # -------------------------------------------------------------------------

    def submit_smart_order(
        self,
        signal: MLSignal,
        side: OrderSide,
        quantity: Quantity,
        instrument: Any,
        reduce_only: bool = False,
    ) -> ClientOrderId | None:
        """
        Create and submit an order using the smart executor when available.

        Falls back to market orders when executor is not configured or declines.

        Parameters
        ----------
        signal : MLSignal
            The ML signal triggering the order.
        side : OrderSide
            The order side (BUY or SELL).
        quantity : Quantity
            The order quantity.
        instrument : Any
            The instrument to trade.
        reduce_only : bool, default False
            Whether this is a reduce-only order.

        Returns
        -------
        ClientOrderId | None
            The client order ID of the placed order, or None if suppressed.

        Examples
        --------
        >>> order_id = component.submit_smart_order(
        ...     signal=ml_signal,
        ...     side=OrderSide.BUY,
        ...     quantity=Quantity.from_str("10.0"),
        ...     instrument=instrument,
        ... )

        """
        from nautilus_trader.core.uuid import UUID4

        # Check circuit breaker first
        try:
            cb = self._circuit_breaker
            if cb is not None and not cb.can_execute():
                self._dry_run_trades += 1
                # Attempt to publish degraded event
                self._publish_degraded_event(signal, side)
                return None
        except Exception as breaker_exc:
            self._log.debug(
                "ml_strategy.order_breaker_guard_failed",
                strategy_id=self._strategy_id,
                exc_info=True,
                error=str(breaker_exc),
            )

        if instrument is None:
            return None

        # Try smart executor if available
        if self._order_executor is not None and self._cache is not None:
            try:
                # Build market state snapshot (lightweight)
                market_state = self._build_market_state(signal.instrument_id)

                client_order_id = self._cache.client_order_id()
                init_id = UUID4()
                ts_init = (
                    self._clock.timestamp_ns() if self._clock else time.time_ns()
                )

                order = self._order_executor.create_order(
                    side=side,
                    quantity=quantity,
                    signal=signal,
                    market_state=market_state,
                    instrument=instrument,
                    trader_id=self._trader_id,
                    strategy_id=self._strategy_id,
                    client_order_id=client_order_id,
                    init_id=init_id,
                    ts_init=ts_init,
                )

                if order is not None:
                    # Unwrap OrderResult if needed
                    core_order = (
                        order.unwrap() if hasattr(order, "unwrap") else order
                    )

                    # Submit order
                    if self._submit_order_callback is not None:
                        self._submit_order_callback(core_order)

                    # Record metric
                    if self._orders_submitted_metric is not None:
                        try:
                            self._orders_submitted_metric.labels(
                                strategy_id=str(self._strategy_id),
                                order_side=side.name,
                            ).inc()
                        except Exception:
                            pass  # Metrics are non-critical

                    # Record performance
                    if self._performance_tracker is not None:
                        try:
                            self._performance_tracker.record_order(
                                core_order, signal
                            )
                        except Exception as perf_exc:
                            self._log.debug(
                                f"ml_strategy.performance_record_order_failed "
                                f"strategy_id={self._strategy_id} error={perf_exc}",
                            )

                    return order.client_order_id

            except Exception as exc:
                # Log and continue to fallback
                self._log.error(
                    f"ml_strategy.smart_order_creation_failed "
                    f"strategy_id={self._strategy_id} "
                    f"order_side={side.name} error={exc}",
                )

        # Fallback to market order (outside try to avoid masking errors)
        return self.place_market_order(
            instrument_id=signal.instrument_id,
            side=side,
            quantity=quantity,
            reduce_only=reduce_only,
        )

    # -------------------------------------------------------------------------
    # Stop Loss Order Placement
    # -------------------------------------------------------------------------

    def place_stop_loss(
        self,
        instrument_id: InstrumentId,
        side: OrderSide,
        quantity: Quantity,
        trigger_price: Price,
    ) -> ClientOrderId | None:
        """
        Place a stop loss order.

        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument for the stop loss.
        side : OrderSide
            The order side (opposite of main position).
        quantity : Quantity
            The order quantity.
        trigger_price : Price
            The stop loss trigger price.

        Returns
        -------
        ClientOrderId | None
            The client order ID of the placed order, or None on failure.

        Examples
        --------
        >>> order_id = component.place_stop_loss(
        ...     instrument_id=InstrumentId.from_str("EURUSD.SIM"),
        ...     side=OrderSide.SELL,
        ...     quantity=Quantity.from_str("10.0"),
        ...     trigger_price=Price.from_str("1.0950"),
        ... )

        """
        from nautilus_trader.core.uuid import UUID4
        from nautilus_trader.model.enums import TimeInForce
        from nautilus_trader.model.enums import TriggerType
        from nautilus_trader.model.orders import StopMarketOrder

        if self._cache is None:
            self._log.error("Cannot place stop loss: Cache not available")
            return None

        # Get timestamp
        ts_init = self._clock.timestamp_ns() if self._clock else time.time_ns()

        # Resolve trader_id and strategy_id to proper types
        from nautilus_trader.model.identifiers import StrategyId
        from nautilus_trader.model.identifiers import TraderId

        trader_id = self._trader_id
        if isinstance(trader_id, str):
            trader_id = TraderId(trader_id)

        strategy_id = self._strategy_id
        if isinstance(strategy_id, str):
            strategy_id = StrategyId(strategy_id)

        # Create stop market order
        order = StopMarketOrder(
            trader_id=trader_id,
            strategy_id=strategy_id,
            instrument_id=instrument_id,
            client_order_id=self._cache.client_order_id(),
            order_side=side,
            quantity=quantity,
            trigger_price=trigger_price,
            trigger_type=TriggerType.DEFAULT,
            init_id=UUID4(),
            ts_init=ts_init,
            time_in_force=TimeInForce.GTC,
            reduce_only=True,
        )

        # Submit order
        if self._submit_order_callback is not None:
            self._submit_order_callback(order)

        # Record orders submitted metric for stop loss
        if self._orders_submitted_metric is not None:
            try:
                self._orders_submitted_metric.labels(
                    strategy_id=str(self._strategy_id),
                    order_side=side.name,
                ).inc()
            except Exception:
                pass  # Metrics are non-critical

        self._log.info(
            f"Placed stop loss: {side.name} {quantity} @ {trigger_price}",
        )

        return order.client_order_id

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _build_market_state(
        self, instrument_id: InstrumentId
    ) -> dict[str, float]:
        """
        Build market state snapshot for smart order execution.

        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument to get market state for.

        Returns
        -------
        dict[str, float]
            Market state with bid, ask, and spread_bps.

        """
        bid = ask = 0.0
        spread_bps = 0.0

        if self._cache is not None:
            try:
                qt = self._cache.quote_tick(instrument_id)
                if qt is not None:
                    bid = float(qt.bid_price.as_double())
                    ask = float(qt.ask_price.as_double())
                    mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else 0.0
                    if mid > 0 and ask >= bid > 0:
                        spread_bps = ((ask - bid) / mid) * 10_000
            except (AttributeError, TypeError):
                pass

        return {
            "bid": bid,
            "ask": ask,
            "spread_bps": spread_bps,
        }

    def _publish_degraded_event(
        self,
        signal: MLSignal,
        side: OrderSide,
    ) -> None:
        """
        Publish a degraded event when circuit breaker suppresses order.

        This is called when the circuit breaker prevents order submission.
        The event signals that the system is operating in degraded mode.

        Parameters
        ----------
        signal : MLSignal
            The signal that triggered the order attempt.
        side : OrderSide
            The intended order side.

        """
        # Import lazily to avoid circular imports
        try:
            from ml.config.bus import MessageBusConfig as _MBC

            _ = _MBC.from_env()  # Validate config availability

            # Note: This requires a bus_publisher which we don't have in component
            # In practice, this would be wired through a callback or the facade
            self._log.warning(
                "ml_strategy.degraded_mode_activated",
                strategy_id=self._strategy_id,
                instrument_id=str(signal.instrument_id),
                order_side=side.name,
            )
        except Exception as pub_exc:
            self._log.warning(
                "ml_strategy.degraded_publish_failed",
                strategy_id=self._strategy_id,
                instrument_id=str(signal.instrument_id),
                order_side=side.name,
                exc_info=True,
                error=str(pub_exc),
            )


__all__ = [
    "CacheProtocol",
    "CircuitBreakerProtocol",
    "LoggerProtocol",
    "OrderExecutorProtocol",
    "OrderSubmissionComponent",
    "PerformanceTrackerProtocol",
]
