"""
Decision persistence component for MLTradingStrategy decomposition.

This component extracts strategy decision persistence and event publishing logic
from BaseMLStrategy following the Protocol-First Interface Design pattern.

Responsibility:
- Persist strategy decisions to StrategyStore with circuit breaker protection
- Publish decision events to message bus
- Manage fallback behavior when store is unavailable
- Track persistence metrics

"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ml.common import normalize_decision_metadata
from ml.common import resolve_decision_horizon_ms
from ml.stores.protocols import StrategyStoreProtocol
from ml.strategies.common.positions import PositionsMetadata


if TYPE_CHECKING:
    from nautilus_trader.model.objects import Quantity

    from ml.actors.base import MLSignal
    from ml.strategies.services import StrategyDecisionPublisher


logger = logging.getLogger(__name__)

_UNSET: object = object()


@runtime_checkable
class CircuitBreakerProtocol(Protocol):
    """
    Protocol for circuit breaker implementation.
    """

    def can_execute(self) -> bool:
        """
        Check if circuit breaker allows execution.
        """
        ...

    def record_success(self) -> None:
        """
        Record a successful operation.
        """
        ...

    def record_failure(self, exc: Exception | None = None) -> None:
        """
        Record a failed operation.
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


class _NoOpLogger:
    """
    No-op logger for when no logger is provided.
    """

    def debug(self, *args: object, **kwargs: object) -> None:
        """
        No-op debug.
        """
        del args, kwargs

    def info(self, *args: object, **kwargs: object) -> None:
        """
        No-op info.
        """
        del args, kwargs

    def warning(self, *args: object, **kwargs: object) -> None:
        """
        No-op warning.
        """
        del args, kwargs

    def error(self, *args: object, **kwargs: object) -> None:
        """
        No-op error.
        """
        del args, kwargs


class _SafeLogger:
    """
    Logger wrapper that tolerates extra kwargs and supports exc_info.
    """

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped
        self._fallback = logger

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        self._log("debug", message, *args, **kwargs)

    def info(self, message: str, *args: object, **kwargs: object) -> None:
        self._log("info", message, *args, **kwargs)

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        self._log("warning", message, *args, **kwargs)

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        self._log("error", message, *args, **kwargs)

    def _log(self, level: str, message: str, *args: object, **kwargs: object) -> None:
        exc_info = kwargs.pop("exc_info", None)
        try:
            log_fn = getattr(self._wrapped, level)
            try:
                if exc_info is not None:
                    log_fn(message, *args, exc_info=exc_info, **kwargs)
                else:
                    log_fn(message, *args, **kwargs)
                return
            except TypeError:
                log_fn(message, *args)
        except Exception:
            pass

        fallback_fn = getattr(self._fallback, level)
        if kwargs:
            if exc_info is not None:
                fallback_fn(message, *args, exc_info=True, extra=kwargs)
            else:
                fallback_fn(message, *args, extra=kwargs)
        else:
            if exc_info is not None:
                fallback_fn(message, *args, exc_info=True)
            else:
                fallback_fn(message, *args)


class DecisionPersistenceComponent:
    """
    Persists strategy decisions and publishes events.

    This component is extracted from BaseMLStrategy to provide focused,
    testable decision persistence functionality following the facade pattern.

    Responsibilities:
    - Write decisions to StrategyStore with circuit breaker protection
    - Publish decision events to message bus
    - Handle fallback when store is unavailable
    - Track persistence metrics (decisions persisted, write latency, batch size)

    Parameters
    ----------
    strategy_id : str
        The strategy identifier for labeling metrics and events.
    strategy_store : StrategyStoreProtocol | None, optional
        The strategy store for persisting decisions. If None, events are
        published directly without persistence.
    circuit_breaker : CircuitBreakerProtocol | None, optional
        Circuit breaker for resilience. If None, no circuit breaking is applied.
    bus_publisher : Any | None, optional
        Message bus publisher for decision events.
    persist_all_signals : bool, default False
        Whether to persist HOLD signals. If False, HOLD decisions are skipped.
    log : LoggerProtocol | None, optional
        Logger instance for debug output.
    active_positions : int, default 0
        Current number of active positions (for risk metrics).
    pending_orders : int, default 0
        Current number of pending orders (for risk metrics).
    stop_loss_pct : float, default 0.0
        Stop loss percentage (for execution params).
    take_profit_pct : float, default 0.0
        Take profit percentage (for execution params).
    max_positions : int, default 1
        Maximum allowed positions (for execution params).
    is_backtesting : bool, default False
        Whether in backtesting mode.
    model_signals : dict[str, MLSignal] | None, optional
        Current model signals buffer for aggregated predictions.
    run_id : str | None, optional
        Optional run identifier for replay/audit correlation.

    Examples
    --------
    >>> component = DecisionPersistenceComponent(
    ...     strategy_id="strategy_1",
    ...     strategy_store=strategy_store,
    ...     circuit_breaker=circuit_breaker,
    ...     persist_all_signals=False,
    ... )
    >>> result = component.persist_decision(signal, "BUY", position_size)

    """

    def __init__(
        self,
        strategy_id: str,
        strategy_store: StrategyStoreProtocol | None = None,
        circuit_breaker: CircuitBreakerProtocol | None = None,
        bus_publisher: Any = None,
        persist_all_signals: bool = False,
        log: Any = None,
        active_positions: int = 0,
        pending_orders: int = 0,
        stop_loss_pct: float = 0.0,
        take_profit_pct: float = 0.0,
        max_positions: int = 1,
        is_backtesting: bool = False,
        model_signals: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> None:
        """
        Initialize the decision persistence component.
        """
        self._strategy_id = strategy_id
        self._strategy_store = strategy_store
        self._circuit_breaker = circuit_breaker
        self._bus_publisher = bus_publisher
        self._persist_all_signals = persist_all_signals
        self._log = _SafeLogger(log if log is not None else _NoOpLogger())

        # State tracking
        self._active_positions = active_positions
        self._pending_orders = pending_orders
        self._stop_loss_pct = stop_loss_pct
        self._take_profit_pct = take_profit_pct
        self._max_positions = max_positions
        self._is_backtesting = is_backtesting
        self._model_signals = model_signals or {}
        self._run_id = run_id
        self._positions_metadata: PositionsMetadata | None = None

        # Decision publisher (lazily initialized)
        self._decision_publisher: StrategyDecisionPublisher | None = None

        # Metrics (lazily initialized)
        self._decisions_persisted_counter: Any = None
        self._write_latency_histogram: Any = None
        self._batch_size_gauge: Any = None
        self._init_metrics()

    def _init_metrics(self) -> None:
        """
        Initialize Prometheus metrics via centralized bootstrap.
        """
        try:
            from ml.common.metrics_bootstrap import get_counter
            from ml.common.metrics_bootstrap import get_gauge
            from ml.common.metrics_bootstrap import get_histogram

            self._decisions_persisted_counter = get_counter(
                "ml_strategy_decisions_persisted_total",
                "Total strategy decisions persisted",
                labelnames=("strategy_id",),
            )
            self._write_latency_histogram = get_histogram(
                "ml_strategy_store_write_latency_seconds",
                "Strategy store write latency",
                labelnames=("strategy_id",),
            )
            self._batch_size_gauge = get_gauge(
                "ml_strategy_store_batch_size",
                "Strategy store batch size",
                labelnames=("strategy_id",),
            )
        except Exception:
            # Metrics unavailable - degrade gracefully
            pass

    # -------------------------------------------------------------------------
    # Public Properties
    # -------------------------------------------------------------------------

    @property
    def strategy_id(self) -> str:
        """
        Get the strategy identifier.
        """
        return self._strategy_id

    @property
    def strategy_store(self) -> StrategyStoreProtocol | None:
        """
        Get the strategy store.
        """
        return self._strategy_store

    @property
    def circuit_breaker(self) -> CircuitBreakerProtocol | None:
        """
        Get the circuit breaker.
        """
        return self._circuit_breaker

    @property
    def persist_all_signals(self) -> bool:
        """
        Get whether all signals (including HOLD) are persisted.
        """
        return self._persist_all_signals

    @property
    def circuit_breaker_open(self) -> bool:
        """
        Check if circuit breaker is currently open (blocking writes).

        Returns
        -------
        bool
            True if circuit breaker is open and blocking writes.

        """
        if self._circuit_breaker is None:
            return False
        try:
            return not self._circuit_breaker.can_execute()
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # State Update Methods
    # -------------------------------------------------------------------------

    def update_state(
        self,
        *,
        active_positions: int | None = None,
        pending_orders: int | None = None,
        is_backtesting: bool | None = None,
        model_signals: dict[str, Any] | None = None,
        positions_metadata: PositionsMetadata | None | object = _UNSET,
    ) -> None:
        """
        Update component state.

        This method allows updating the component state to reflect
        changes in the parent strategy's state.

        Parameters
        ----------
        active_positions : int | None, optional
            Updated number of active positions.
        pending_orders : int | None, optional
            Updated number of pending orders.
        is_backtesting : bool | None, optional
            Updated backtesting mode flag.
        model_signals : dict[str, Any] | None, optional
            Updated model signals buffer.
        positions_metadata : dict[str, object] | None, optional
            Updated positions metadata payload for decision persistence.

        """
        if active_positions is not None:
            self._active_positions = active_positions
        if pending_orders is not None:
            self._pending_orders = pending_orders
        if is_backtesting is not None:
            self._is_backtesting = is_backtesting
        if model_signals is not None:
            self._model_signals = model_signals
        if positions_metadata is not _UNSET:
            self._positions_metadata = (
                positions_metadata if isinstance(positions_metadata, dict) else None
            )

    def update_dependencies(
        self,
        *,
        strategy_store: StrategyStoreProtocol | None,
        bus_publisher: Any | None,
    ) -> None:
        """
        Update external dependencies that can change after initialization.

        Parameters
        ----------
        strategy_store : StrategyStoreProtocol | None
            Updated strategy store instance (or None to disable persistence).
        bus_publisher : Any | None
            Updated message bus publisher (or None to disable publishing).

        """
        self._strategy_store = strategy_store
        if bus_publisher is not self._bus_publisher:
            self._bus_publisher = bus_publisher
            self._decision_publisher = None

    # -------------------------------------------------------------------------
    # Decision Persistence
    # -------------------------------------------------------------------------

    def persist_decision(
        self,
        signal: MLSignal,
        decision_type: str,
        position_size: Quantity | None = None,
        risk_metrics: dict[str, float] | None = None,
        execution_params: dict[str, Any] | None = None,
        model_signals: dict[str, Any] | None = None,
        persist_hold: bool = False,
    ) -> bool:
        """
        Persist a strategy decision.

        This method handles the full persistence flow including:
        - Checking for store availability
        - Circuit breaker protection
        - Building default risk metrics and execution params
        - Writing to store with timing
        - Emitting metrics
        - Publishing events on failure/unavailability

        Parameters
        ----------
        signal : MLSignal
            The ML signal that triggered the decision.
        decision_type : str
            The decision type: "BUY", "SELL", or "HOLD".
        position_size : Quantity | None, optional
            The position size for the trade.
        risk_metrics : dict[str, float] | None, optional
            Risk metrics calculated for this decision.
        execution_params : dict[str, Any] | None, optional
            Execution parameters for the trade.
        model_signals : dict[str, Any] | None, optional
            Model signals buffer for aggregated predictions.
        persist_hold : bool, optional
            Force persistence of HOLD decisions even when HOLD filtering is enabled.

        Returns
        -------
        bool
            True if persisted successfully, False otherwise.

        Examples
        --------
        >>> result = component.persist_decision(
        ...     signal=signal,
        ...     decision_type="BUY",
        ...     position_size=Quantity.from_str("10"),
        ... )
        >>> assert result is True

        """
        # Use provided model_signals or fall back to instance state
        if model_signals is not None:
            self._model_signals = model_signals

        # Build model predictions for event/store
        model_predictions = self._build_model_predictions(signal)
        decision_metadata_payload = (
            signal.metadata.get("decision_metadata") if signal.metadata else None
        )
        decision_metadata = normalize_decision_metadata(
            decision_metadata_payload,
            model_id=signal.model_id,
        )
        horizon_ms = resolve_decision_horizon_ms(decision_metadata)

        # If no store is configured, publish event directly (best-effort)
        if self._strategy_store is None:
            execution_params = self._prepare_execution_params(
                execution_params=execution_params,
                signal=signal,
                decision_type=decision_type,
                position_size=position_size,
                decision_metadata=decision_metadata,
                horizon_ms=horizon_ms,
            )
            return self._handle_no_store(
                signal=signal,
                decision_type=decision_type,
                risk_metrics=risk_metrics,
                execution_params=execution_params,
                model_predictions=model_predictions,
                decision_metadata=decision_metadata,
            )

        # Skip HOLD signals unless configured to persist them
        if decision_type == "HOLD" and not self._persist_all_signals and not persist_hold:
            return False

        # Calculate default risk metrics if not provided
        if risk_metrics is None:
            risk_metrics = self._build_risk_metrics(signal, position_size)

        execution_params = self._prepare_execution_params(
            execution_params=execution_params,
            signal=signal,
            decision_type=decision_type,
            position_size=position_size,
            decision_metadata=decision_metadata,
            horizon_ms=horizon_ms,
        )

        # Check circuit breaker before store write
        if self._check_circuit_breaker_open():
            return self._handle_circuit_breaker_open(
                signal=signal,
                decision_type=decision_type,
                risk_metrics=risk_metrics,
                execution_params=execution_params,
                model_predictions=model_predictions,
                decision_metadata=decision_metadata,
            )

        # Write to store with timing
        return self._write_to_store(
            signal=signal,
            decision_type=decision_type,
            risk_metrics=risk_metrics,
            execution_params=execution_params,
            model_predictions=model_predictions,
            decision_metadata=decision_metadata,
        )

    def _handle_no_store(
        self,
        signal: MLSignal,
        decision_type: str,
        risk_metrics: dict[str, float] | None,
        execution_params: dict[str, Any] | None,
        model_predictions: dict[str, float],
        decision_metadata: dict[str, Any],
    ) -> bool:
        """
        Handle persistence when no store is available.
        """
        try:
            from ml.config.events import EventStatus

            is_live = not self._is_backtesting
            publisher = self.get_decision_publisher()
            if publisher is None:
                return False

            publisher.publish(
                strategy_id=self._strategy_id,
                instrument_id=str(signal.instrument_id),
                signal_type=decision_type,
                strength=float(signal.confidence),
                model_predictions=model_predictions,
                risk_metrics=risk_metrics,
                execution_params=execution_params,
                decision_metadata=decision_metadata,
                ts_event=int(signal.ts_event),
                is_live=is_live,
                status=EventStatus.SUCCESS,
            )
            return True
        except Exception as exc:
            self._log.warning(
                "ml_strategy.strategy_decision_publish_failed",
                strategy_id=self._strategy_id,
                instrument_id=str(signal.instrument_id),
                decision_type=decision_type,
                exc_info=True,
                error=str(exc),
            )
            return False

    def _check_circuit_breaker_open(self) -> bool:
        """
        Check if circuit breaker is open.
        """
        try:
            if self._circuit_breaker is not None and not self._circuit_breaker.can_execute():
                # Emit fallback activation metric
                self._emit_fallback_metric("open")
                return True
        except Exception as exc:
            self._log.debug(
                "ml_strategy.breaker_check_failed",
                strategy_id=self._strategy_id,
                exc_info=True,
                error=str(exc),
            )
        return False

    def _emit_fallback_metric(self, level: str) -> None:
        """
        Emit fallback activation metric.
        """
        try:
            from ml.common.metrics_bootstrap import get_counter

            get_counter(
                "ml_fallback_activations_total",
                "Fallback activations",
                labelnames=("component", "level"),
            ).labels(component="strategy_store_write", level=level).inc()
        except Exception as exc:
            self._log.debug(
                "ml_strategy.fallback_metric_emit_failed",
                strategy_id=self._strategy_id,
                level=level,
                exc_info=True,
                error=str(exc),
            )

    def _handle_circuit_breaker_open(
        self,
        signal: MLSignal,
        decision_type: str,
        risk_metrics: dict[str, float],
        execution_params: dict[str, Any],
        model_predictions: dict[str, float],
        decision_metadata: dict[str, Any],
    ) -> bool:
        """
        Handle persistence when circuit breaker is open.
        """
        # Publish guardrail event with PARTIAL status
        return self._publish_partial_event(
            signal=signal,
            decision_type=decision_type,
            risk_metrics=risk_metrics,
            execution_params=execution_params,
            model_predictions=model_predictions,
            decision_metadata=decision_metadata,
        )

    def _write_to_store(
        self,
        signal: MLSignal,
        decision_type: str,
        risk_metrics: dict[str, float],
        execution_params: dict[str, Any],
        model_predictions: dict[str, float],
        decision_metadata: dict[str, Any],
    ) -> bool:
        """
        Write decision to strategy store with timing.
        """
        start_time = time.perf_counter()

        try:
            if self._strategy_store is not None:
                is_live = not self._is_backtesting

                self._strategy_store.write_signal(
                    strategy_id=self._strategy_id,
                    instrument_id=str(signal.instrument_id),
                    signal_type=decision_type,
                    strength=float(signal.confidence),
                    model_predictions=model_predictions,
                    risk_metrics=risk_metrics,
                    execution_params=execution_params,
                    decision_metadata=decision_metadata,
                    ts_event=signal.ts_event,
                    is_live=is_live,
                    run_id=self._run_id,
                )

                # Record circuit breaker success
                if self._circuit_breaker is not None:
                    try:
                        self._circuit_breaker.record_success()
                    except Exception as exc:
                        self._log.debug(
                            f"ml_strategy.breaker_record_success_failed strategy={self._strategy_id} error={exc!r}",
                            exc_info=True,
                        )

                # Record metrics
                self._record_write_metrics(start_time)
                return True

        except Exception as exc:
            # Record circuit breaker failure
            self._handle_store_write_failure(
                exc=exc,
                signal=signal,
                decision_type=decision_type,
                risk_metrics=risk_metrics,
                execution_params=execution_params,
                model_predictions=model_predictions,
                decision_metadata=decision_metadata,
            )
            return False

        return False

    def _handle_store_write_failure(
        self,
        exc: Exception,
        signal: MLSignal,
        decision_type: str,
        risk_metrics: dict[str, float],
        execution_params: dict[str, Any],
        model_predictions: dict[str, float],
        decision_metadata: dict[str, Any],
    ) -> None:
        """
        Handle store write failure.
        """
        # Record circuit breaker failure
        if self._circuit_breaker is not None:
            try:
                self._circuit_breaker.record_failure(exc)
            except Exception as breaker_exc:
                self._log.debug(
                    f"ml_strategy.breaker_record_failure_failed strategy={self._strategy_id} error={breaker_exc!r}",
                    exc_info=True,
                )

        # Publish PARTIAL guardrail event
        self._publish_partial_event(
            signal=signal,
            decision_type=decision_type,
            risk_metrics=risk_metrics,
            execution_params=execution_params,
            model_predictions=model_predictions,
            decision_metadata=decision_metadata,
        )

        # Log error
        self._log.error(
            "ml_strategy.strategy_store_write_failed",
            strategy_id=self._strategy_id,
            instrument_id=str(signal.instrument_id),
            decision_type=decision_type,
            exc_info=True,
            error=str(exc),
        )

    def _publish_partial_event(
        self,
        signal: MLSignal,
        decision_type: str,
        risk_metrics: dict[str, float],
        execution_params: dict[str, Any],
        model_predictions: dict[str, float],
        decision_metadata: dict[str, Any],
    ) -> bool:
        """
        Publish a PARTIAL status event.
        """
        try:
            from ml.config.events import EventStatus

            publisher = self.get_decision_publisher()
            if publisher is None:
                return False

            is_live = not self._is_backtesting

            publisher.publish(
                strategy_id=self._strategy_id,
                instrument_id=str(signal.instrument_id),
                signal_type=decision_type,
                strength=float(signal.confidence),
                model_predictions=model_predictions,
                risk_metrics=risk_metrics,
                execution_params=execution_params,
                decision_metadata=decision_metadata,
                ts_event=int(signal.ts_event),
                is_live=is_live,
                status=EventStatus.PARTIAL,
            )
            return False  # Partial means not fully persisted
        except Exception as exc:
            self._log.warning(
                "ml_strategy.partial_publish_failed",
                strategy_id=self._strategy_id,
                instrument_id=str(signal.instrument_id),
                decision_type=decision_type,
                exc_info=True,
                error=str(exc),
            )
            return False

    def _record_write_metrics(self, start_time: float) -> None:
        """
        Record write metrics after successful persistence.
        """
        write_latency = time.perf_counter() - start_time

        # Increment decisions counter
        if self._decisions_persisted_counter is not None:
            try:
                self._decisions_persisted_counter.labels(
                    strategy_id=self._strategy_id,
                ).inc()
            except Exception as exc:
                self._log.debug(
                    "ml_strategy.metric_emit_failed",
                    strategy_id=self._strategy_id,
                    metric="decisions_persisted",
                    exc_info=True,
                    error=str(exc),
                )

        # Record write latency
        if self._write_latency_histogram is not None:
            try:
                self._write_latency_histogram.labels(
                    strategy_id=self._strategy_id,
                ).observe(write_latency)
            except Exception as exc:
                self._log.debug(
                    "ml_strategy.metric_emit_failed",
                    strategy_id=self._strategy_id,
                    metric="write_latency",
                    exc_info=True,
                    error=str(exc),
                )

        # Update batch size gauge
        if self._batch_size_gauge is not None:
            try:
                store = self._strategy_store
                if store is not None and hasattr(store, "_write_buffer"):
                    buffer = getattr(store, "_write_buffer", [])
                    self._batch_size_gauge.labels(
                        strategy_id=self._strategy_id,
                    ).set(len(buffer))
            except Exception as exc:
                self._log.debug(
                    "ml_strategy.metric_emit_failed",
                    strategy_id=self._strategy_id,
                    metric="batch_size",
                    exc_info=True,
                    error=str(exc),
                )

    # -------------------------------------------------------------------------
    # Decision Publisher
    # -------------------------------------------------------------------------

    def get_decision_publisher(self) -> StrategyDecisionPublisher | None:
        """
        Get or create decision publisher (lazy initialization).

        Returns
        -------
        StrategyDecisionPublisher | None
            The decision publisher, or None if unavailable.

        Examples
        --------
        >>> publisher = component.get_decision_publisher()
        >>> if publisher:
        ...     publisher.publish(...)

        """
        if self._decision_publisher is None:
            try:
                from ml.config.bus import MessageBusConfig
                from ml.strategies.services import StrategyDecisionPublisher

                cfg = MessageBusConfig.from_env()
                self._decision_publisher = StrategyDecisionPublisher(
                    self._bus_publisher,
                    scheme=cfg.scheme,
                    prefix=cfg.topic_prefix,
                )
            except Exception as exc:
                self._log.debug(
                    "ml_strategy.decision_publisher_init_failed",
                    strategy_id=self._strategy_id,
                    exc_info=True,
                    error=str(exc),
                )
                return None

        return self._decision_publisher

    def publish_decision_event(
        self,
        signal: MLSignal,
        decision_type: str,
        position_size: Quantity | None = None,
        risk_metrics: dict[str, float] | None = None,
        execution_params: dict[str, Any] | None = None,
        model_predictions: dict[str, float] | None = None,
        decision_metadata: dict[str, Any] | None = None,
        is_live: bool = True,
    ) -> bool:
        """
        Publish decision event to message bus.

        This method publishes a strategy decision event using the configured
        message bus. Publishing is best-effort and non-blocking.

        Parameters
        ----------
        signal : MLSignal
            The ML signal that triggered the decision.
        decision_type : str
            The decision type: "BUY", "SELL", or "HOLD".
        position_size : Quantity | None, optional
            The position size for the trade.
        risk_metrics : dict[str, float] | None, optional
            Risk metrics calculated for this decision.
        execution_params : dict[str, Any] | None, optional
            Execution parameters for the trade.
        model_predictions : dict[str, float] | None, optional
            Model predictions dictionary.
        decision_metadata : dict[str, Any] | None, optional
            Decision metadata payload.
        is_live : bool, default True
            Whether this is live trading or backtesting.

        Returns
        -------
        bool
            True if published successfully, False otherwise.

        Examples
        --------
        >>> success = component.publish_decision_event(
        ...     signal=signal,
        ...     decision_type="BUY",
        ...     is_live=True,
        ... )

        """
        try:
            from ml.common.message_bus import publisher_from_config
            from ml.common.message_topics import build_topic_for_stage
            from ml.config.bus import MessageBusConfig
            from ml.config.events import EventStatus
            from ml.config.events import Source
            from ml.config.events import Stage

            bus_cfg = MessageBusConfig.from_env()
            publisher = self._bus_publisher or publisher_from_config(bus_cfg)
            if publisher is None:
                return False

            instrument_str = str(signal.instrument_id)
            topic = build_topic_for_stage(
                Stage.SIGNAL_EMITTED,
                instrument_str,
                scheme=bus_cfg.scheme,
                prefix=bus_cfg.topic_prefix,
            )

            source = Source.LIVE.value if is_live else Source.HISTORICAL.value

            # Build model predictions if not provided
            if model_predictions is None:
                model_predictions = self._build_model_predictions(signal)
            if decision_metadata is None:
                decision_metadata_payload = (
                    signal.metadata.get("decision_metadata") if signal.metadata else None
                )
                decision_metadata = normalize_decision_metadata(
                    decision_metadata_payload,
                    model_id=signal.model_id,
                )

            payload: dict[str, Any] = {
                "dataset_id": "signals",
                "stage": Stage.SIGNAL_EMITTED.value,
                "status": EventStatus.SUCCESS.value,
                "source": source,
                "strategy_id": self._strategy_id,
                "instrument_id": instrument_str,
                "signal_type": decision_type,
                "strength": float(signal.confidence),
                "model_predictions": model_predictions,
                "risk_metrics": risk_metrics or {},
                "execution_params": execution_params or {},
                "decision_metadata": decision_metadata or {},
                "ts_event": int(signal.ts_event),
            }

            try:
                publisher.publish(topic, payload)
                return True
            except Exception as exc:
                # Never affect control flow
                self._log.debug(
                    "ml_strategy.decision_event_publish_failed",
                    strategy_id=self._strategy_id,
                    instrument_id=instrument_str,
                    decision_type=decision_type,
                    exc_info=True,
                    error=str(exc),
                )
                return False
        except Exception as exc:
            # Defensive: ensure hot path is not impacted
            self._log.debug(
                "ml_strategy.decision_event_build_failed",
                strategy_id=self._strategy_id,
                instrument_id=str(signal.instrument_id),
                decision_type=decision_type,
                exc_info=True,
                error=str(exc),
            )
            return False

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _build_risk_metrics(
        self,
        signal: MLSignal,
        position_size: Quantity | None,
    ) -> dict[str, float]:
        """
        Build risk metrics from signal and position.

        Parameters
        ----------
        signal : MLSignal
            The ML signal.
        position_size : Quantity | None
            The position size.

        Returns
        -------
        dict[str, float]
            Risk metrics dictionary.

        Examples
        --------
        >>> metrics = component._build_risk_metrics(signal, position_size)
        >>> assert "confidence" in metrics
        >>> assert "prediction" in metrics

        """
        metrics: dict[str, float] = {
            "confidence": float(signal.confidence),
            "prediction": float(signal.prediction),
            "active_positions": float(self._active_positions),
            "pending_orders": float(self._pending_orders),
        }

        # Add position size if provided
        if position_size is not None:
            try:
                metrics["position_size"] = float(position_size.as_double())
            except (AttributeError, TypeError) as exc:
                self._log.debug(
                    "ml_strategy.position_size_metric_failed",
                    strategy_id=self._strategy_id,
                    exc_info=True,
                    error=str(exc),
                )

        return metrics

    def _build_execution_params(
        self,
        signal: MLSignal,
        decision_type: str,
        position_size: Quantity | None = None,
    ) -> dict[str, Any]:
        """
        Build execution parameters for decision.

        Parameters
        ----------
        signal : MLSignal
            The ML signal.
        decision_type : str
            The decision type.
        position_size : Quantity | None
            The position size.

        Returns
        -------
        dict[str, Any]
            Execution parameters dictionary.

        Examples
        --------
        >>> params = component._build_execution_params(signal, "BUY", qty)
        >>> assert "stop_loss_pct" in params
        >>> assert "take_profit_pct" in params

        """
        params: dict[str, Any] = {
            "stop_loss_pct": self._stop_loss_pct,
            "take_profit_pct": self._take_profit_pct,
            "position_size": str(position_size) if position_size else None,
            "max_positions": self._max_positions,
            "current_positions": self._active_positions,
        }
        return params

    def _merge_positions_metadata(
        self,
        execution_params: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """
        Merge positions metadata into execution params when available.
        """
        if self._positions_metadata is None:
            return execution_params
        params = dict(execution_params) if execution_params is not None else {}
        positions_payload = params.get("positions")
        if isinstance(positions_payload, dict):
            for key, value in self._positions_metadata.items():
                positions_payload.setdefault(key, value)
            params["positions"] = positions_payload
        else:
            params["positions"] = dict(self._positions_metadata)
        return params

    def _inject_horizon_metadata(
        self,
        execution_params: dict[str, Any],
        *,
        decision_metadata: dict[str, Any] | None,
        horizon_ms: int | None,
    ) -> dict[str, Any]:
        """
        Inject horizon metadata into execution params when available.
        """
        if decision_metadata is None:
            return execution_params
        horizon_payload = decision_metadata.get("horizon")
        if horizon_payload is not None and "horizon" not in execution_params:
            if isinstance(horizon_payload, dict):
                execution_params["horizon"] = dict(horizon_payload)
            else:
                execution_params["horizon"] = horizon_payload
        if horizon_ms is not None and "horizon_ms" not in execution_params:
            execution_params["horizon_ms"] = int(horizon_ms)
        return execution_params

    def _prepare_execution_params(
        self,
        *,
        execution_params: dict[str, Any] | None,
        signal: MLSignal,
        decision_type: str,
        position_size: Quantity | None,
        decision_metadata: dict[str, Any] | None,
        horizon_ms: int | None,
    ) -> dict[str, Any]:
        """
        Build and normalize execution params, including horizon metadata.
        """
        if execution_params is None:
            execution_params = self._build_execution_params(signal, decision_type, position_size)
        execution_params = self._merge_positions_metadata(execution_params)
        if execution_params is None:
            execution_params = {}
        return self._inject_horizon_metadata(
            execution_params,
            decision_metadata=decision_metadata,
            horizon_ms=horizon_ms,
        )

    def _build_model_predictions(
        self,
        signal: MLSignal,
    ) -> dict[str, float]:
        """
        Build model predictions map from signal and aggregated context.

        Parameters
        ----------
        signal : MLSignal
            The ML signal.

        Returns
        -------
        dict[str, float]
            Model predictions dictionary.

        Examples
        --------
        >>> predictions = component._build_model_predictions(signal)
        >>> assert len(predictions) >= 1

        """
        # Get model ID from signal
        model_id = getattr(signal, "model_id", None) or signal.metadata.get("model_id", "unknown")
        predictions: dict[str, float] = {str(model_id): float(signal.prediction)}

        # Add aggregated model predictions if available
        try:
            if hasattr(signal, "metadata") and "aggregated_from" in signal.metadata:
                for mid in signal.metadata["aggregated_from"]:
                    if mid in self._model_signals:
                        sig = self._model_signals[mid]
                        predictions[str(mid)] = float(sig.prediction)
        except Exception as exc:
            self._log.debug(
                "ml_strategy.aggregated_predictions_build_failed",
                strategy_id=self._strategy_id,
                exc_info=True,
                error=str(exc),
            )

        return predictions


__all__ = [
    "CircuitBreakerProtocol",
    "DecisionPersistenceComponent",
    "LoggerProtocol",
    "StrategyStoreProtocol",
]
