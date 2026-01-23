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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from nautilus_trader.model.identifiers import ClientOrderId
from nautilus_trader.model.identifiers import VenueOrderId

from ml.common.metrics_bootstrap import get_counter
from ml.config.base import LimitPriceConfig
from ml.config.base import LimitPriceSource
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
QuoteMetadata = dict[str, object]
ExecutionMetadata = dict[str, object]


@dataclass(slots=True)
class _PendingLimitOrder:
    order: Any
    signal: Any
    side: Any
    quantity: Any
    instrument: Any
    reduce_only: bool
    next_action_ns: int
    ttl_seconds: float
    cadence_seconds: float
    attempts_remaining: int


quote_tick_stale_total = get_counter(
    "ml_strategy_quote_tick_stale_total",
    "Total stale quote ticks encountered during order submission",
    labels=["strategy_id"],
)

execution_total = get_counter(
    "ml_strategy_execution_total",
    "Total executions by mode and fallback reason",
    labels=["strategy_id", "mode", "fallback_reason"],
)

risk_halt_total = get_counter(
    "ml_strategy_risk_halt_total",
    "Total order submissions gated by risk halts",
    labels=["strategy_id", "event", "reason"],
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


def _build_order_intent_record(
    order: Any,
    *,
    is_live: bool,
    positions_metadata: dict[str, object] | None = None,
    quote_metadata: QuoteMetadata | None = None,
    exit_metadata: dict[str, object] | None = None,
    execution_metadata: ExecutionMetadata | None = None,
) -> dict[str, Any]:
    side = getattr(order, "side", getattr(order, "order_side", None))
    quantity = getattr(order, "quantity", None)
    record: dict[str, Any] = {
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
    if positions_metadata is not None:
        record["positions"] = dict(positions_metadata)
    if quote_metadata is not None:
        record["quote"] = dict(quote_metadata)
    if exit_metadata is not None:
        record["exit"] = dict(exit_metadata)
    if execution_metadata is not None:
        record["execution"] = dict(execution_metadata)
    return record


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

    def write(
        self,
        order: Any,
        *,
        is_live: bool,
        positions_metadata: dict[str, object] | None = None,
        quote_metadata: QuoteMetadata | None = None,
        exit_metadata: dict[str, object] | None = None,
        execution_metadata: ExecutionMetadata | None = None,
    ) -> None:
        """
        Append a single order intent record to JSONL.

        This is a best-effort operation; failures are logged and suppressed.

        """
        if not self._available:
            return
        try:
            record = _build_order_intent_record(
                order,
                is_live=is_live,
                positions_metadata=positions_metadata,
                quote_metadata=quote_metadata,
                exit_metadata=exit_metadata,
                execution_metadata=execution_metadata,
            )
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
class RiskHaltProviderProtocol(Protocol):
    """
    Protocol for exposing risk halt state.
    """

    def is_trading_halted(self) -> bool:
        """
        Return True when trading is halted by risk controls.
        """
        ...

    def get_halt_reason(self) -> str | None:
        """
        Return a stable reason label for the current halt.
        """
        ...

    def allow_reduce_only_when_halted(self) -> bool:
        """
        Return True when reduce-only orders may bypass a halt.
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

    def trade_tick(self, instrument_id: Any) -> Any:
        """
        Get latest trade tick for instrument.
        """
        ...

    def price(self, instrument_id: Any, price_type: Any) -> Any:
        """
        Get latest cached price for instrument.
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
    risk_halt_provider : RiskHaltProviderProtocol | None, optional
        Provider for risk halt state; blocks orders when halted.
    performance_tracker : PerformanceTrackerProtocol | None, optional
        Performance tracker for order recording.
    cache : CacheProtocol | None, optional
        Cache for instrument/quote data access.
    submit_order_callback : Callable | None, optional
        Callback function to submit orders to the trading system.
    cancel_order_callback : Callable | None, optional
        Callback function to cancel submitted orders.
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
        risk_halt_provider: RiskHaltProviderProtocol | None = None,
        performance_tracker: PerformanceTrackerProtocol | None = None,
        cache: CacheProtocol | None = None,
        submit_order_callback: Callable[[Any], None] | None = None,
        cancel_order_callback: Callable[[Any], None] | None = None,
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
        self._risk_halt_provider = risk_halt_provider
        self._performance_tracker = performance_tracker
        self._cache = cache
        self._submit_order_callback = submit_order_callback
        self._cancel_order_callback = cancel_order_callback
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
        self._last_quote_metadata: QuoteMetadata | None = None
        self._last_execution_metadata: ExecutionMetadata | None = None
        self._dry_run_trades: int = 0
        self._trades_executed: int = 0
        self._pending_orders: int = 0
        self._pending_limit_orders: list[_PendingLimitOrder] = []
        self._last_risk_halt_state: bool | None = None
        self._last_risk_halt_reason: str | None = None

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

    def pop_last_quote_metadata(self) -> QuoteMetadata | None:
        """
        Pop quote metadata captured during the last market state build.

        Returns
        -------
        QuoteMetadata | None
            Copy of the last quote metadata, or None if unavailable.

        Examples
        --------
        >>> metadata = component.pop_last_quote_metadata()
        >>> metadata is None or metadata["available"] in (True, False)

        """
        metadata = self._last_quote_metadata
        self._last_quote_metadata = None
        if metadata is None:
            return None
        return dict(metadata)

    def pop_last_execution_metadata(self) -> ExecutionMetadata | None:
        """
        Pop execution metadata captured during the last submission attempt.

        Returns
        -------
        ExecutionMetadata | None
            Copy of the last execution metadata, or None if unavailable.

        Examples
        --------
        >>> metadata = component.pop_last_execution_metadata()
        >>> metadata is None or metadata["mode"] in ("market", "smart")

        """
        metadata = self._last_execution_metadata
        self._last_execution_metadata = None
        if metadata is None:
            return None
        return dict(metadata)

    def _build_execution_metadata(
        self,
        *,
        mode: str,
        fallback_reason: str | None,
        ttl_plan: dict[str, object] | None = None,
    ) -> ExecutionMetadata:
        """
        Build execution metadata payload for order intents.
        """
        metadata: ExecutionMetadata = {
            "mode": mode,
            "fallback_reason": fallback_reason,
        }
        if ttl_plan is not None:
            metadata["ttl_plan"] = dict(ttl_plan)
        return metadata

    def _read_ttl_plan(self) -> dict[str, object] | None:
        """
        Safely read TTL plan from the order executor when available.
        """
        if self._order_executor is None or not hasattr(self._order_executor, "get_last_ttl_plan"):
            return None
        try:
            return cast(dict[str, object] | None, self._order_executor.get_last_ttl_plan())
        except Exception as exc:
            self._log.debug(
                "ml_strategy.ttl_plan_read_failed",
                exc_info=True,
                error=str(exc),
            )
            return None

    def _resolve_limit_price_config(self) -> LimitPriceConfig | None:
        executor = self._order_executor
        if executor is None:
            return None
        cfg = getattr(executor, "config", None)
        limit_cfg = getattr(cfg, "limit_price_config", None)
        if isinstance(limit_cfg, LimitPriceConfig):
            return limit_cfg
        return None

    def _allow_limit_price_fallback(self, market_state: dict[str, float]) -> bool:
        limit_cfg = self._resolve_limit_price_config()
        if limit_cfg is None:
            return False
        fallback_sources = {
            LimitPriceSource.LAST_TRADE,
            LimitPriceSource.CACHE_LAST,
        }
        if not any(source in fallback_sources for source in limit_cfg.source_priority):
            return False
        return any(
            float(market_state.get(key, 0.0) or 0.0) > 0.0
            for key in ("last_trade", "cache_last")
        )

    def _interval_ns_from_plan(self, ttl_plan: dict[str, object]) -> int:
        """
        Resolve the TTL/cadence interval (nanoseconds) for a pending limit order.
        """
        ttl_seconds = _to_float(ttl_plan.get("ttl_seconds")) or 0.0
        cadence_seconds = _to_float(ttl_plan.get("cadence_seconds")) or 0.0
        ttl_ns = int(max(ttl_seconds, 0.0) * 1_000_000_000)
        cadence_ns = int(max(cadence_seconds, 0.0) * 1_000_000_000)
        if cadence_ns <= 0:
            return ttl_ns
        if ttl_ns <= 0:
            return cadence_ns
        return max(ttl_ns, cadence_ns)

    def _record_execution_metric(self) -> None:
        """
        Emit execution metrics for the last recorded execution metadata.
        """
        metadata = self._last_execution_metadata
        if metadata is None:
            return
        mode = str(metadata.get("mode") or "unknown")
        fallback_reason = metadata.get("fallback_reason")
        fallback_label = str(fallback_reason) if fallback_reason else "none"
        try:
            execution_total.labels(
                strategy_id=str(self._strategy_id),
                mode=mode,
                fallback_reason=fallback_label,
            ).inc()
        except Exception as exc:
            self._log.debug(
                "ml_strategy.execution_metric_failed",
                strategy_id=self._strategy_id,
                mode=mode,
                fallback_reason=fallback_label,
                exc_info=True,
                error=str(exc),
            )

    def _submit_core_order(
        self,
        core_order: Any,
        *,
        side: OrderSide,
        signal: MLSignal,
    ) -> ClientOrderId | None:
        """
        Submit a prepared order and record metrics/performance.
        """
        if self._submit_order_callback is not None:
            self._submit_order_callback(core_order)

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

        if self._performance_tracker is not None:
            try:
                self._performance_tracker.record_order(core_order, signal)
            except Exception as perf_exc:
                self._log.debug(
                    f"ml_strategy.performance_record_order_failed "
                    f"strategy_id={self._strategy_id} error={perf_exc}",
                    exc_info=True,
                )

        self._record_execution_metric()
        return getattr(core_order, "client_order_id", None)

    def _maybe_track_limit_order(
        self,
        core_order: Any,
        *,
        signal: MLSignal,
        side: OrderSide,
        quantity: Quantity,
        instrument: Any,
        reduce_only: bool,
        ttl_plan: dict[str, object] | None,
    ) -> None:
        """
        Track resting limit orders for TTL management.
        """
        if ttl_plan is None:
            return
        attempts = _to_int(ttl_plan.get("attempts")) or 0
        if attempts <= 0:
            return
        interval_ns = self._interval_ns_from_plan(ttl_plan)
        if interval_ns <= 0:
            return
        time_in_force = getattr(core_order, "time_in_force", None)
        try:
            from nautilus_trader.model.enums import TimeInForce

            if time_in_force != TimeInForce.GTC:
                return
        except Exception:
            return

        ts_init = getattr(core_order, "ts_init", None)
        try:
            submitted_ns = int(ts_init) if ts_init is not None else self._timestamp_ns()
        except Exception:
            submitted_ns = self._timestamp_ns()

        self._pending_limit_orders.append(
            _PendingLimitOrder(
                order=core_order,
                signal=signal,
                side=side,
                quantity=quantity,
                instrument=instrument,
                reduce_only=reduce_only,
                next_action_ns=submitted_ns + interval_ns,
                ttl_seconds=_to_float(ttl_plan.get("ttl_seconds")) or 0.0,
                cadence_seconds=_to_float(ttl_plan.get("cadence_seconds")) or 0.0,
                attempts_remaining=attempts,
            ),
        )

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
        cancel_order_callback: Callable[[Any], None] | None = None,
        trader_id: Any = None,
        clock: Any = None,
        order_executor: OrderExecutorProtocol | None | object = _UNSET,
        risk_halt_provider: RiskHaltProviderProtocol | None | object = _UNSET,
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
        cancel_order_callback : Callable | None, optional
            Updated cancel order callback.
        trader_id : Any, optional
            Updated trader ID.
        clock : Any, optional
            Updated clock instance.
        order_executor : OrderExecutorProtocol | None, optional
            Updated order executor (pass None to disable).
        risk_halt_provider : RiskHaltProviderProtocol | None, optional
            Updated risk halt provider (pass None to disable).

        """
        if instrument_id is not None:
            self._instrument_id = instrument_id
        if cache is not None:
            self._cache = cache
        if submit_order_callback is not None:
            self._submit_order_callback = submit_order_callback
        if cancel_order_callback is not None:
            self._cancel_order_callback = cancel_order_callback
        if trader_id is not None:
            self._trader_id = trader_id
        if clock is not None:
            self._clock = clock
        if order_executor is not _UNSET:
            self._order_executor = cast(OrderExecutorProtocol | None, order_executor)
        if risk_halt_provider is not _UNSET:
            self._risk_halt_provider = cast(RiskHaltProviderProtocol | None, risk_halt_provider)

    # -------------------------------------------------------------------------
    # Circuit Breaker Check
    # -------------------------------------------------------------------------

    def _normalize_halt_reason(self, reason: str | None) -> str:
        if reason is None:
            return "unknown"
        token = str(reason).strip()
        return token or "unknown"

    def _emit_risk_halt_metric(self, *, event: str, reason: str) -> None:
        try:
            risk_halt_total.labels(
                strategy_id=str(self._strategy_id),
                event=event,
                reason=reason,
            ).inc()
        except Exception as exc:
            self._log.debug(
                "ml_strategy.risk_halt_metric_failed",
                strategy_id=self._strategy_id,
                event=event,
                reason=reason,
                exc_info=True,
                error=str(exc),
            )

    def _check_risk_halt(self, *, reduce_only: bool) -> bool:
        provider = self._risk_halt_provider
        if provider is None:
            return True
        try:
            halted = bool(provider.is_trading_halted())
            reason = self._normalize_halt_reason(provider.get_halt_reason())
        except Exception as exc:
            self._log.debug(
                "ml_strategy.risk_halt_check_failed",
                strategy_id=self._strategy_id,
                exc_info=True,
                error=str(exc),
            )
            return True

        if halted:
            allow_reduce_only = False
            if reduce_only:
                try:
                    allow_reduce_only = bool(provider.allow_reduce_only_when_halted())
                except Exception as exc:
                    self._log.debug(
                        "ml_strategy.risk_halt_allow_reduce_only_failed",
                        strategy_id=self._strategy_id,
                        exc_info=True,
                        error=str(exc),
                    )
                    allow_reduce_only = False
            if allow_reduce_only:
                self._last_risk_halt_state = True
                self._last_risk_halt_reason = reason
                self._emit_risk_halt_metric(event="bypassed", reason=reason)
                self._log.info(
                    "ml_strategy.order_submission_halt_bypassed",
                    strategy_id=self._strategy_id,
                    reason=reason,
                )
                return True

            self._last_risk_halt_state = True
            self._last_risk_halt_reason = reason
            self._dry_run_trades += 1
            self._emit_risk_halt_metric(event="blocked", reason=reason)
            self._log.warning(
                "ml_strategy.order_submission_halted",
                strategy_id=self._strategy_id,
                reason=reason,
            )
            return False

        if self._last_risk_halt_state:
            resume_reason = self._last_risk_halt_reason or reason
            self._emit_risk_halt_metric(event="resumed", reason=resume_reason)
            self._last_risk_halt_reason = None
        self._last_risk_halt_state = False
        return True

    def _check_circuit_breaker(self, *, reduce_only: bool) -> bool:
        """
        Check if circuit breaker allows execution.

        Parameters
        ----------
        reduce_only : bool
            Whether the order is reduce-only (may bypass risk halts when configured).

        Returns
        -------
        bool
            True if execution is allowed, False if suppressed.

        """
        if not self._check_risk_halt(reduce_only=reduce_only):
            return False
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
        preserve_quote_metadata: bool = False,
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
        preserve_quote_metadata : bool, default False
            Whether to preserve the last quote metadata snapshot.

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

        if not preserve_quote_metadata:
            self._last_quote_metadata = None

        # Check circuit breaker
        if not self._check_circuit_breaker(reduce_only=reduce_only):
            # Return a fresh client order id without submitting
            order_id = self._resolve_client_order_id()
            if order_id is not None:
                return order_id
            return None

        if self._cache is None:
            self._log.error("Cannot place market order: Cache not available")
            return None

        if self._last_execution_metadata is None:
            self._last_execution_metadata = self._build_execution_metadata(
                mode="market",
                fallback_reason=None,
            )

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

        self._record_execution_metric()
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

        self._last_quote_metadata = None
        self._last_execution_metadata = None

        # Check circuit breaker and risk halt first
        if not self._check_circuit_breaker(reduce_only=reduce_only):
            self._publish_degraded_event(signal, side)
            return None

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
                fallback_reason: str | None = None
                quote_missing = (
                    self._last_quote_metadata is not None
                    and self._last_quote_metadata.get("available") is False
                )
                if quote_missing or is_stale:
                    fallback_reason = "quote_unavailable" if quote_missing else "stale_quote"
                    if not self._allow_limit_price_fallback(market_state):
                        self._last_execution_metadata = self._build_execution_metadata(
                            mode="market",
                            fallback_reason=fallback_reason,
                        )
                        order_id = self.place_market_order(
                            instrument_id=signal.instrument_id,
                            side=side,
                            quantity=quantity,
                            reduce_only=reduce_only,
                            preserve_quote_metadata=True,
                        )
                        if order_id is None:
                            self._last_quote_metadata = None
                            self._last_execution_metadata = None
                        return order_id

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
                    ttl_plan = self._read_ttl_plan()
                    self._last_execution_metadata = self._build_execution_metadata(
                        mode="smart",
                        fallback_reason=fallback_reason,
                        ttl_plan=ttl_plan,
                    )

                    order_id = self._submit_core_order(
                        core_order,
                        side=side,
                        signal=signal,
                    )
                    self._maybe_track_limit_order(
                        core_order,
                        signal=signal,
                        side=side,
                        quantity=quantity,
                        instrument=instrument,
                        reduce_only=reduce_only,
                        ttl_plan=ttl_plan,
                    )

                    return order_id or getattr(order, "client_order_id", None)
                self._last_execution_metadata = self._build_execution_metadata(
                    mode="market",
                    fallback_reason="executor_no_order",
                )

            except Exception as exc:
                self._last_execution_metadata = self._build_execution_metadata(
                    mode="market",
                    fallback_reason="executor_error",
                )
                # Log and continue to fallback
                self._log.error(
                    f"ml_strategy.smart_order_creation_failed "
                    f"strategy_id={self._strategy_id} "
                    f"order_side={side.name} error={exc}",
                    exc_info=True,
                )
        else:
            if self._order_executor is None:
                self._last_execution_metadata = self._build_execution_metadata(
                    mode="market",
                    fallback_reason="executor_unavailable",
                )
            elif self._cache is None:
                self._last_execution_metadata = self._build_execution_metadata(
                    mode="market",
                    fallback_reason="cache_unavailable",
                )

        # Fallback to market order (outside try to avoid masking errors)
        preserve_metadata = self._last_quote_metadata is not None
        order_id = self.place_market_order(
            instrument_id=signal.instrument_id,
            side=side,
            quantity=quantity,
            reduce_only=reduce_only,
            preserve_quote_metadata=preserve_metadata,
        )
        if order_id is None:
            self._last_execution_metadata = None
            if not preserve_metadata:
                self._last_quote_metadata = None
        return order_id

    def process_pending_limit_orders(self, *, now_ns: int | None = None) -> None:
        """
        Evaluate pending limit orders and trigger TTL cancel-replace handling.

        Parameters
        ----------
        now_ns : int | None, optional
            Override the current timestamp (nanoseconds) for testing.

        Examples
        --------
        >>> component.process_pending_limit_orders()

        """
        if not self._pending_limit_orders:
            return
        if self._cancel_order_callback is None or self._submit_order_callback is None:
            self._log.debug(
                "ml_strategy.pending_limit_orders_skipped",
                strategy_id=self._strategy_id,
                error="missing_callbacks",
            )
            return

        current_ns = self._timestamp_ns() if now_ns is None else int(now_ns)
        remaining: list[_PendingLimitOrder] = []

        for pending in self._pending_limit_orders:
            if pending.attempts_remaining <= 0:
                continue
            if current_ns < pending.next_action_ns:
                remaining.append(pending)
                continue
            try:
                self._cancel_order_callback(pending.order)
            except Exception as exc:
                self._log.debug(
                    "ml_strategy.pending_limit_order_cancel_failed",
                    strategy_id=self._strategy_id,
                    exc_info=True,
                    error=str(exc),
                )
                remaining.append(pending)
                continue

            replacement = self._create_replacement_order(pending, current_ns)
            if replacement is None:
                continue

            self._last_execution_metadata = self._build_execution_metadata(
                mode="smart",
                fallback_reason="ttl_replace",
            )
            self._submit_core_order(
                replacement,
                side=pending.side,
                signal=pending.signal,
            )

            attempts_remaining = pending.attempts_remaining - 1
            interval_ns = self._interval_ns_from_plan(
                {
                    "ttl_seconds": pending.ttl_seconds,
                    "cadence_seconds": pending.cadence_seconds,
                },
            )
            if attempts_remaining > 0 and interval_ns > 0:
                remaining.append(
                    _PendingLimitOrder(
                        order=replacement,
                        signal=pending.signal,
                        side=pending.side,
                        quantity=pending.quantity,
                        instrument=pending.instrument,
                        reduce_only=pending.reduce_only,
                        next_action_ns=current_ns + interval_ns,
                        ttl_seconds=pending.ttl_seconds,
                        cadence_seconds=pending.cadence_seconds,
                        attempts_remaining=attempts_remaining,
                    ),
                )

        self._pending_limit_orders = remaining

    def _create_replacement_order(
        self,
        pending: _PendingLimitOrder,
        now_ns: int,
    ) -> Any | None:
        """
        Build a replacement order for a pending limit order.
        """
        if self._order_executor is None or self._cache is None:
            self._log.debug(
                "ml_strategy.pending_limit_order_replace_unavailable",
                strategy_id=self._strategy_id,
                error="executor_or_cache_unavailable",
            )
            return None

        instrument_id = getattr(pending.signal, "instrument_id", None) or self._instrument_id
        if instrument_id is None:
            self._log.debug(
                "ml_strategy.pending_limit_order_replace_failed",
                strategy_id=self._strategy_id,
                error="instrument_id_missing",
            )
            return None

        market_state, is_stale = self._build_market_state(
            instrument_id,
            reference_ts=now_ns,
        )
        fallback_reason: str | None = None
        quote_missing = (
            self._last_quote_metadata is not None
            and self._last_quote_metadata.get("available") is False
        )
        if quote_missing or is_stale:
            fallback_reason = "quote_unavailable" if quote_missing else "stale_quote"
            if not self._allow_limit_price_fallback(market_state):
                self._last_execution_metadata = self._build_execution_metadata(
                    mode="market",
                    fallback_reason=fallback_reason,
                )
                self.place_market_order(
                    instrument_id=instrument_id,
                    side=pending.side,
                    quantity=pending.quantity,
                    reduce_only=pending.reduce_only,
                    preserve_quote_metadata=True,
                )
                return None

        trader_id, strategy_id = self._resolve_ids()
        if trader_id is None or strategy_id is None:
            self._log.debug(
                "ml_strategy.pending_limit_order_replace_failed",
                strategy_id=str(self._strategy_id),
                error="missing_trader_or_strategy_id",
            )
            return None

        client_order_id = self._resolve_client_order_id()
        if client_order_id is None:
            self._log.debug(
                "ml_strategy.pending_limit_order_replace_failed",
                strategy_id=str(self._strategy_id),
                error="client_order_id_unavailable",
            )
            return None

        try:
            instrument = pending.instrument
            if instrument is None:
                instrument = self._cache.instrument(instrument_id)
            order = self._order_executor.create_order(
                side=pending.side,
                quantity=pending.quantity,
                signal=pending.signal,
                market_state=market_state,
                instrument=instrument,
                trader_id=trader_id,
                strategy_id=strategy_id,
                client_order_id=client_order_id,
                init_id=UUID4(),
                ts_init=now_ns,
            )
        except Exception as exc:
            self._log.error(
                "ml_strategy.pending_limit_order_replace_failed",
                strategy_id=self._strategy_id,
                exc_info=True,
                error=str(exc),
            )
            return None

        return order.unwrap() if hasattr(order, "unwrap") else order

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

        self._last_quote_metadata = None

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
            Market state with quote, trade, cached prices, and a staleness flag.

        """
        bid = ask = mid = 0.0
        spread_bps = 0.0
        last_trade = 0.0
        cache_last = 0.0
        is_stale = False
        self._last_quote_metadata = None

        if self._cache is not None:
            try:
                qt = self._cache.quote_tick(instrument_id)
            except (AttributeError, TypeError) as exc:
                self._log.debug(
                    "ml_strategy.market_state_unavailable",
                    strategy_id=self._strategy_id,
                    instrument_id=str(instrument_id),
                    exc_info=True,
                    error=str(exc),
                )
            else:
                if qt is None:
                    self._last_quote_metadata = {
                        "available": False,
                        "ts_event": None,
                        "age_ns": None,
                        "max_age_ns": self._max_quote_age_ns,
                        "stale": None,
                    }
                else:
                    quote_ts_event = _to_int(getattr(qt, "ts_event", None))
                    quote_age_ns: int | None = None
                    stale_flag: bool | None = None
                    if (
                        self._max_quote_age_ns is not None
                        and reference_ts is not None
                        and quote_ts_event is not None
                    ):
                        quote_age_ns = reference_ts - quote_ts_event
                        stale_flag = quote_age_ns > self._max_quote_age_ns
                        if stale_flag:
                            self._log.debug(
                                "ml_strategy.quote_tick_stale",
                                strategy_id=self._strategy_id,
                                instrument_id=str(instrument_id),
                                quote_ts_event=quote_ts_event,
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

                    self._last_quote_metadata = {
                        "available": True,
                        "ts_event": quote_ts_event,
                        "age_ns": quote_age_ns,
                        "max_age_ns": self._max_quote_age_ns,
                        "stale": stale_flag,
                    }
                    if not is_stale:
                        bid = float(qt.bid_price.as_double())
                        ask = float(qt.ask_price.as_double())
                        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else 0.0
                        if mid > 0 and ask >= bid > 0:
                            spread_bps = ((ask - bid) / mid) * 10_000

            try:
                trade_tick = self._cache.trade_tick(instrument_id)
                if trade_tick is not None:
                    last_trade = float(trade_tick.price.as_double())
            except (AttributeError, TypeError) as exc:
                self._log.debug(
                    "ml_strategy.trade_tick_unavailable",
                    strategy_id=self._strategy_id,
                    instrument_id=str(instrument_id),
                    exc_info=True,
                    error=str(exc),
                )

            try:
                if hasattr(self._cache, "price"):
                    from nautilus_trader.model.enums import PriceType

                    cached_price = self._cache.price(instrument_id, PriceType.LAST)
                    if cached_price is not None:
                        cache_last = float(cached_price.as_double())
            except Exception as exc:
                self._log.debug(
                    "ml_strategy.cached_price_unavailable",
                    strategy_id=self._strategy_id,
                    instrument_id=str(instrument_id),
                    exc_info=True,
                    error=str(exc),
                )

        return {
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread_bps": spread_bps,
            "last_trade": last_trade,
            "cache_last": cache_last,
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
    "RiskHaltProviderProtocol",
]
