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

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from nautilus_trader.model.identifiers import ClientOrderId
from nautilus_trader.model.identifiers import VenueOrderId

from ml.common.metrics_bootstrap import get_counter
from ml.config.events import Source
from ml.strategies.common.decision_persistence import _NoOpLogger
from ml.strategies.common.decision_persistence import _SafeLogger
from nautilus_trader.core.uuid import UUID4


if TYPE_CHECKING:
    from nautilus_trader.model.identifiers import ClientOrderId
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.objects import Price
    from nautilus_trader.model.objects import Quantity

    from ml.actors.base import MLSignal
    from nautilus_trader.model.enums import OrderSide


_UNSET: object = object()


quote_tick_stale_total = get_counter(
    "ml_strategy_quote_tick_stale_total",
    "Total stale quote ticks encountered during order submission",
    labels=["strategy_id"],
)


def resolve_order_intent_path(order_intent_path: str | None) -> Path | None:
    """
    Resolve the JSONL output path for order intent serialization.

    Falls back to ``ML_FILE_STORE_PATH`` when an explicit path is not provided.
    Returns ``None`` when no path can be resolved.

    """
    if order_intent_path:
        return Path(order_intent_path)
    file_store_path = os.getenv("ML_FILE_STORE_PATH")
    if file_store_path:
        return Path(file_store_path) / "orders" / "order_intents.jsonl"
    return None


def _enum_to_str(value: Any) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    if name is not None:
        return str(name)
    raw = getattr(value, "value", None)
    if raw is not None:
        return str(raw)
    return str(value)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if hasattr(value, "as_double"):
            return float(value.as_double())
        return float(value)
    except Exception:
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _build_order_intent_record(order: Any, *, is_live: bool) -> dict[str, Any]:
    side = getattr(order, "side", getattr(order, "order_side", None))
    quantity = getattr(order, "quantity", None)
    return {
        "strategy_id": str(getattr(order, "strategy_id", "")) or None,
        "trader_id": str(getattr(order, "trader_id", "")) or None,
        "instrument_id": str(getattr(order, "instrument_id", "")) or None,
        "client_order_id": str(getattr(order, "client_order_id", "")) or None,
        "order_type": _enum_to_str(getattr(order, "order_type", None)),
        "side": _enum_to_str(side),
        "quantity": _to_float(quantity),
        "time_in_force": _enum_to_str(getattr(order, "time_in_force", None)),
        "reduce_only": getattr(order, "is_reduce_only", getattr(order, "reduce_only", None)),
        "ts_init": _to_int(getattr(order, "ts_init", None)),
        "is_live": bool(is_live),
        "source": Source.LIVE.value if is_live else Source.HISTORICAL.value,
        "order_class": type(order).__name__,
    }


class OrderIntentWriter:
    """
    JSONL serializer for broker order intents.

    Used as a safe stub for order submission during dry-run validation.

    """

    def __init__(
        self,
        path: Path,
        *,
        log: LoggerProtocol | None = None,
    ) -> None:
        """
        Initialize the order intent writer.
        """
        self._path = path
        self._log = _SafeLogger(log if log is not None else _NoOpLogger())
        self._available = True
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._available = False
            self._log.error(
                "ml_strategy.order_intent_path_unavailable",
                exc_info=True,
                error=str(exc),
            )

    @property
    def path(self) -> Path:
        """
        Return the configured JSONL output path.
        """
        return self._path

    def write(self, order: Any, *, is_live: bool) -> None:
        """
        Append a single order intent record to JSONL.

        This is a best-effort operation; failures are logged and suppressed.

        """
        if not self._available:
            return
        try:
            record = _build_order_intent_record(order, is_live=is_live)
            payload = json.dumps(record, separators=(",", ":"))
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(payload + "\n")
        except Exception as exc:
            self._log.debug(
                "ml_strategy.order_intent_write_failed",
                exc_info=True,
                error=str(exc),
            )


@runtime_checkable
class CircuitBreakerProtocol(Protocol):
    """
    Protocol for circuit breaker functionality.
    """

    def can_execute(self) -> bool:
        """
        Check if execution is allowed.
        """
        ...

    def record_success(self) -> None:
        """
        Record a successful execution.
        """
        ...

    def record_failure(self) -> None:
        """
        Record a failed execution.
        """
        ...


@runtime_checkable
class CacheProtocol(Protocol):
    """
    Protocol for cache access.
    """

    def instrument(self, instrument_id: Any) -> Any:
        """
        Get instrument by ID.
        """
        ...

    def quote_tick(self, instrument_id: Any) -> Any:
        """
        Get latest quote tick for instrument.
        """
        ...

    def client_order_id(self, venue_order_id: Any | None = None) -> Any:
        """
        Generate a new client order ID (optionally from a venue order ID).
        """
        ...


@runtime_checkable
class OrderExecutorProtocol(Protocol):
    """
    Protocol for order execution.
    """

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
        """
        Create an order based on signal and market conditions.
        """
        ...


@runtime_checkable
class PerformanceTrackerProtocol(Protocol):
    """
    Protocol for performance tracking.
    """

    def record_order(self, order: Any, signal: Any) -> None:
        """
        Record an order placement.
        """
        ...


@runtime_checkable
class LoggerProtocol(Protocol):
    """
    Protocol for logging interface.
    """

    def debug(self, *args: object, **kwargs: object) -> None:
        """
        Log debug message.
        """
        ...

    def info(self, *args: object, **kwargs: object) -> None:
        """
        Log info message.
        """
        ...

    def warning(self, *args: object, **kwargs: object) -> None:
        """
        Log warning message.
        """
        ...

    def error(self, *args: object, **kwargs: object) -> None:
        """
        Log error message.
        """
        ...


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
    max_quote_age_ms : int | None, optional
        Maximum quote age in milliseconds allowed for execution market state.
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
        max_quote_age_ms: int | None = None,
        orders_submitted_metric: Any = None,
    ) -> None:
        """
        Initialize the order submission component.
        """
        self._strategy_id = strategy_id
        self._order_executor = order_executor
        self._circuit_breaker = circuit_breaker
        self._performance_tracker = performance_tracker
        self._cache = cache
        self._submit_order_callback = submit_order_callback
        self._log = _SafeLogger(log if log is not None else _NoOpLogger())
        self._instrument_id = instrument_id
        self._trader_id = trader_id
        self._clock = clock
        if max_quote_age_ms is None:
            self._max_quote_age_ns = None
        elif max_quote_age_ms < 0:
            self._log.debug(
                "ml_strategy.max_quote_age_invalid",
                strategy_id=self._strategy_id,
                error=str(max_quote_age_ms),
            )
            self._max_quote_age_ns = None
        else:
            self._max_quote_age_ns = int(max_quote_age_ms) * 1_000_000
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
        """
        Get the strategy ID.
        """
        return self._strategy_id

    @property
    def order_executor(self) -> OrderExecutorProtocol | None:
        """
        Get the order executor.
        """
        return self._order_executor

    @property
    def circuit_breaker(self) -> CircuitBreakerProtocol | None:
        """
        Get the circuit breaker.
        """
        return self._circuit_breaker

    @property
    def dry_run_trades(self) -> int:
        """
        Get the count of dry run trades.
        """
        return self._dry_run_trades

    @property
    def trades_executed(self) -> int:
        """
        Get the count of trades executed.
        """
        return self._trades_executed

    @property
    def pending_orders(self) -> int:
        """
        Get the count of pending orders.
        """
        return self._pending_orders

    def _timestamp_ns(self) -> int:
        """
        Return a nanosecond timestamp with fallback for unimplemented clocks.
        """
        if self._clock is None:
            return time.time_ns()
        try:
            return int(self._clock.timestamp_ns())
        except Exception as exc:
            self._log.debug(
                "ml_strategy.clock_timestamp_failed",
                exc_info=True,
                error=str(exc),
            )
            return time.time_ns()

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
        order_executor: OrderExecutorProtocol | None | object = _UNSET,
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
        order_executor : OrderExecutorProtocol | None, optional
            Updated order executor (pass None to disable).

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
        if order_executor is not _UNSET:
            self._order_executor = cast(OrderExecutorProtocol | None, order_executor)

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

    def _resolve_ids(self) -> tuple[Any, Any]:
        """
        Resolve trader and strategy identifiers to Nautilus types when possible.
        """
        from nautilus_trader.model.identifiers import StrategyId
        from nautilus_trader.model.identifiers import TraderId

        trader_id = self._trader_id
        if isinstance(trader_id, str):
            trader_id = TraderId(trader_id)

        strategy_id = self._strategy_id
        if isinstance(strategy_id, str):
            strategy_id = StrategyId(strategy_id)

        return trader_id, strategy_id

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
            order_id = self._resolve_client_order_id()
            if order_id is not None:
                return order_id
            return None

        if self._cache is None:
            self._log.error("Cannot place market order: Cache not available")
            return None

        # Get timestamp
        ts_init = self._timestamp_ns()

        trader_id, strategy_id = self._resolve_ids()
        if trader_id is None or strategy_id is None:
            self._log.error(
                "Cannot place market order: missing trader_id or strategy_id",
                strategy_id=str(self._strategy_id),
            )
            return None

        client_order_id = self._resolve_client_order_id()
        if client_order_id is None:
            self._log.error(
                "Cannot place market order: client_order_id unavailable",
                strategy_id=str(self._strategy_id),
            )
            return None

        # Create market order
        order = MarketOrder(
            trader_id=trader_id,
            strategy_id=strategy_id,
            instrument_id=instrument_id,
            client_order_id=client_order_id,
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
            except Exception as exc:
                self._log.debug(
                    "ml_strategy.orders_submitted_metric_failed",
                    strategy_id=self._strategy_id,
                    order_side=side.name,
                    exc_info=True,
                    error=str(exc),
                )

        self._log.info(
            f"Placed {side.name} market order: {quantity} @ market " f"(reduce_only={reduce_only})",
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
                market_state, is_stale = self._build_market_state(
                    signal.instrument_id,
                    reference_ts=signal.ts_event,
                )
                if is_stale:
                    return self.place_market_order(
                        instrument_id=signal.instrument_id,
                        side=side,
                        quantity=quantity,
                        reduce_only=reduce_only,
                    )

                trader_id, strategy_id = self._resolve_ids()
                if trader_id is None or strategy_id is None:
                    raise ValueError("Missing trader_id or strategy_id")

                client_order_id = self._resolve_client_order_id()
                if client_order_id is None:
                    raise ValueError("Missing client_order_id")
                init_id = UUID4()
                ts_init = self._timestamp_ns()

                order = self._order_executor.create_order(
                    side=side,
                    quantity=quantity,
                    signal=signal,
                    market_state=market_state,
                    instrument=instrument,
                    trader_id=trader_id,
                    strategy_id=strategy_id,
                    client_order_id=client_order_id,
                    init_id=init_id,
                    ts_init=ts_init,
                )

                if order is not None:
                    # Unwrap OrderResult if needed
                    core_order = order.unwrap() if hasattr(order, "unwrap") else order

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
                        except Exception as exc:
                            self._log.debug(
                                "ml_strategy.orders_submitted_metric_failed",
                                strategy_id=self._strategy_id,
                                order_side=side.name,
                                exc_info=True,
                                error=str(exc),
                            )

                    # Record performance
                    if self._performance_tracker is not None:
                        try:
                            self._performance_tracker.record_order(
                                core_order,
                                signal,
                            )
                        except Exception as perf_exc:
                            self._log.debug(
                                f"ml_strategy.performance_record_order_failed "
                                f"strategy_id={self._strategy_id} error={perf_exc}",
                                exc_info=True,
                            )

                    return order.client_order_id

            except Exception as exc:
                # Log and continue to fallback
                self._log.error(
                    f"ml_strategy.smart_order_creation_failed "
                    f"strategy_id={self._strategy_id} "
                    f"order_side={side.name} error={exc}",
                    exc_info=True,
                )

        # Fallback to market order (outside try to avoid masking errors)
        return self.place_market_order(
            instrument_id=signal.instrument_id,
            side=side,
            quantity=quantity,
            reduce_only=reduce_only,
        )

    def _resolve_client_order_id(self) -> ClientOrderId | None:
        """
        Resolve a ClientOrderId across Nautilus cache API variants.

        Falls back to generating a UUID-derived ClientOrderId when cache helpers are
        unavailable or require a VenueOrderId in newer versions.

        """
        if self._cache is not None:
            try:
                cached_id = self._cache.client_order_id()
                if cached_id is not None:
                    return cached_id
                self._log.debug(
                    "ml_strategy.client_order_id_empty",
                    error="cache_returned_none",
                )
            except TypeError as exc:
                self._log.debug(
                    "ml_strategy.client_order_id_signature_mismatch",
                    exc_info=True,
                    error=str(exc),
                )
                try:
                    cached_id = self._cache.client_order_id(VenueOrderId(str(UUID4())))
                    if cached_id is not None:
                        return cached_id
                    self._log.debug(
                        "ml_strategy.client_order_id_empty",
                        error="cache_returned_none_with_venue",
                    )
                except Exception as inner_exc:
                    self._log.debug(
                        "ml_strategy.client_order_id_with_venue_failed",
                        exc_info=True,
                        error=str(inner_exc),
                    )
            except Exception as exc:
                self._log.debug(
                    "ml_strategy.client_order_id_failed",
                    exc_info=True,
                    error=str(exc),
                )

        try:
            return ClientOrderId(str(UUID4()))
        except Exception as exc:
            self._log.debug(
                "ml_strategy.client_order_id_fallback_failed",
                exc_info=True,
                error=str(exc),
            )
            return None

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
        ts_init = self._timestamp_ns()

        trader_id, strategy_id = self._resolve_ids()
        if trader_id is None or strategy_id is None:
            self._log.error(
                "Cannot place stop loss: missing trader_id or strategy_id",
                strategy_id=str(self._strategy_id),
            )
            return None

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
            except Exception as exc:
                self._log.debug(
                    "ml_strategy.orders_submitted_metric_failed",
                    strategy_id=self._strategy_id,
                    order_side=side.name,
                    exc_info=True,
                    error=str(exc),
                )

        self._log.info(
            f"Placed stop loss: {side.name} {quantity} @ {trigger_price}",
        )

        return order.client_order_id

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _build_market_state(
        self,
        instrument_id: InstrumentId,
        *,
        reference_ts: int | None = None,
    ) -> tuple[dict[str, float], bool]:
        """
        Build market state snapshot for smart order execution.

        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument to get market state for.
        reference_ts : int | None, optional
            Reference timestamp to evaluate quote staleness.

        Returns
        -------
        tuple[dict[str, float], bool]
            Market state with bid, ask, spread_bps, and a staleness flag.

        """
        bid = ask = 0.0
        spread_bps = 0.0
        is_stale = False

        if self._cache is not None:
            try:
                qt = self._cache.quote_tick(instrument_id)
                if qt is not None:
                    if self._max_quote_age_ns is not None and reference_ts is not None:
                        quote_age_ns = reference_ts - int(qt.ts_event)
                        if quote_age_ns > self._max_quote_age_ns:
                            self._log.debug(
                                "ml_strategy.quote_tick_stale",
                                strategy_id=self._strategy_id,
                                instrument_id=str(instrument_id),
                                quote_ts_event=int(qt.ts_event),
                                reference_ts=reference_ts,
                                max_quote_age_ns=self._max_quote_age_ns,
                            )
                            try:
                                quote_tick_stale_total.labels(
                                    strategy_id=str(self._strategy_id),
                                ).inc()
                            except Exception as exc:
                                self._log.debug(
                                    "ml_strategy.quote_tick_stale_metric_failed",
                                    strategy_id=self._strategy_id,
                                    exc_info=True,
                                    error=str(exc),
                                )
                            is_stale = True
                            return {
                                "bid": 0.0,
                                "ask": 0.0,
                                "spread_bps": 0.0,
                            }, is_stale

                    bid = float(qt.bid_price.as_double())
                    ask = float(qt.ask_price.as_double())
                    mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else 0.0
                    if mid > 0 and ask >= bid > 0:
                        spread_bps = ((ask - bid) / mid) * 10_000
            except (AttributeError, TypeError) as exc:
                self._log.debug(
                    "ml_strategy.market_state_unavailable",
                    strategy_id=self._strategy_id,
                    instrument_id=str(instrument_id),
                    exc_info=True,
                    error=str(exc),
                )

        return {
            "bid": bid,
            "ask": ask,
            "spread_bps": spread_bps,
        }, is_stale

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
