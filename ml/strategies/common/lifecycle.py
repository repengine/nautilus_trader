"""
Lifecycle component for MLTradingStrategy decomposition.

This component extracts lifecycle management logic from BaseMLStrategy
following the Protocol-First Interface Design pattern.

Responsibility:
- Manage strategy lifecycle: subscriptions, startup, shutdown
- Subscribe to ML signals (with/without client ID)
- Subscribe to configured instrument
- Flush strategy store on stop
- Log appropriate statistics on stop

"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    pass


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


@runtime_checkable
class StrategyStoreProtocol(Protocol):
    """
    Protocol for strategy store interface.
    """

    def flush(self) -> None:
        """
        Flush any pending writes to persistent storage.
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


class LifecycleComponent:
    """
    Manages strategy lifecycle: subscriptions, startup, shutdown.

    This component is extracted from BaseMLStrategy to provide focused,
    testable lifecycle management functionality following the facade pattern.

    Responsibilities:
    - Subscribe to ML signals on startup (with/without client_id)
    - Subscribe to configured instrument on startup
    - Log configuration details on startup
    - Flush strategy store on stop
    - Log statistics on stop (trades, win rate, PnL)

    Parameters
    ----------
    strategy_id : str
        The strategy identifier for logging.
    instrument_id : Any
        The target instrument ID for subscriptions.
    signal_client_id : str | None, optional
        Specific client ID for ML signal subscriptions. If None, subscribes to all.
    signal_source : str | None, optional
        Actor identifier used to scope ML signal subscriptions. When set, the
        DataType metadata includes ``{"source": signal_source}``.
    target_model_ids : list[str] | None, optional
        List of target model IDs for logging.
    aggregation_mode : str | None, optional
        Aggregation mode for logging.
    position_size_pct : float, default 0.02
        Position size percentage for logging.
    min_confidence : float, default 0.0
        Minimum confidence threshold for logging.
    execute_trades : bool, default True
        Whether trades are being executed (vs dry run mode).
    subscribe_quote_ticks : bool, default False
        Whether to subscribe to quote ticks for execution market state.
    quote_schema : str | None, optional
        Optional quote schema parameter passed to data client subscriptions.
    subscribe_data_callback : Callable | None, optional
        Callback function to subscribe to data. Expected signature:
        `subscribe_data(data_type, client_id=None)`.
    subscribe_instrument_callback : Callable | None, optional
        Callback function to subscribe to instrument. Expected signature:
        `subscribe_instrument(instrument_id)`.
    subscribe_quote_ticks_callback : Callable | None, optional
        Callback function to subscribe to quote ticks. Expected signature:
        `subscribe_quote_ticks(instrument_id, params=None)`.
    log : LoggerProtocol | None, optional
        Logger instance for output.

    Examples
    --------
    >>> component = LifecycleComponent(
    ...     strategy_id="strategy_1",
    ...     instrument_id=InstrumentId.from_str("EURUSD.SIM"),
    ...     signal_client_id="actor_1",
    ...     subscribe_data_callback=strategy.subscribe_data,
    ...     subscribe_instrument_callback=strategy.subscribe_instrument,
    ...     log=strategy.log,
    ... )
    >>> component.on_start()  # Subscribes and logs config

    """

    def __init__(
        self,
        strategy_id: str,
        instrument_id: Any,
        signal_client_id: str | None = None,
        signal_source: str | None = None,
        target_model_ids: list[str] | None = None,
        aggregation_mode: str | None = None,
        position_size_pct: float = 0.02,
        min_confidence: float = 0.0,
        execute_trades: bool = True,
        subscribe_quote_ticks: bool = False,
        quote_schema: str | None = None,
        subscribe_data_callback: Callable[..., None] | None = None,
        subscribe_instrument_callback: Callable[..., None] | None = None,
        subscribe_quote_ticks_callback: Callable[..., None] | None = None,
        log: Any = None,
    ) -> None:
        """
        Initialize the lifecycle component.
        """
        self._strategy_id = strategy_id
        self._instrument_id = instrument_id
        self._signal_client_id = signal_client_id
        self._signal_source = signal_source
        self._target_model_ids = target_model_ids
        self._aggregation_mode = aggregation_mode
        self._position_size_pct = position_size_pct
        self._min_confidence = min_confidence
        self._execute_trades = execute_trades
        self._subscribe_quote_ticks = subscribe_quote_ticks
        self._quote_schema = quote_schema
        self._subscribe_data_callback = subscribe_data_callback
        self._subscribe_instrument_callback = subscribe_instrument_callback
        self._subscribe_quote_ticks_callback = subscribe_quote_ticks_callback
        self._log = log if log is not None else _NoOpLogger()

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
    def instrument_id(self) -> Any:
        """
        Get the instrument ID.
        """
        return self._instrument_id

    @property
    def signal_client_id(self) -> str | None:
        """
        Get the signal client ID.
        """
        return self._signal_client_id

    @property
    def signal_source(self) -> str | None:
        """
        Get the signal source identifier.
        """
        return self._signal_source

    @property
    def target_model_ids(self) -> list[str] | None:
        """
        Get the target model IDs.
        """
        return self._target_model_ids

    @property
    def aggregation_mode(self) -> str | None:
        """
        Get the aggregation mode.
        """
        return self._aggregation_mode

    @property
    def execute_trades(self) -> bool:
        """
        Get whether trades are being executed.
        """
        return self._execute_trades

    # -------------------------------------------------------------------------
    # Lifecycle Methods
    # -------------------------------------------------------------------------

    def on_start(self) -> None:
        """
        Initialize subscriptions and log configuration.

        This method sets up the strategy by subscribing to ML signals from
        the configured source (with or without client_id) and subscribing
        to the configured instrument for market data.

        If subscription callbacks are not provided, this method logs warnings
        but does not raise exceptions.

        Examples
        --------
        >>> component.on_start()
        # Subscribes to ML signals and instrument, logs configuration

        """
        self._log.info(f"Starting {self._strategy_id}")

        # Subscribe to ML signals
        self._subscribe_to_ml_signals()

        # Subscribe to instrument for market data
        self._subscribe_to_instrument()

        # Subscribe to quote ticks for execution
        self._subscribe_to_quote_ticks()

        # Log configuration
        self._log_configuration()

    def on_stop(
        self,
        strategy_store: StrategyStoreProtocol | None = None,
        signals_received: int = 0,
        trades_executed: int = 0,
        winning_trades: int = 0,
        total_pnl: Decimal = Decimal("0"),
        dry_run_trades: int = 0,
    ) -> None:
        """
        Clean up and log final statistics.

        This method flushes the strategy store buffer (if available) and
        logs final statistics based on whether the strategy was in execute
        mode or dry run mode.

        Parameters
        ----------
        strategy_store : StrategyStoreProtocol | None, optional
            The strategy store to flush. If None, no flush is attempted.
        signals_received : int, default 0
            Total number of signals received.
        trades_executed : int, default 0
            Total number of trades executed.
        winning_trades : int, default 0
            Total number of winning trades.
        total_pnl : Decimal, default Decimal("0")
            Total profit and loss.
        dry_run_trades : int, default 0
            Total number of dry run trades (when execute_trades=False).

        Examples
        --------
        >>> component.on_stop(
        ...     strategy_store=store,
        ...     signals_received=100,
        ...     trades_executed=50,
        ...     winning_trades=30,
        ...     total_pnl=Decimal("1250.50"),
        ... )

        """
        # Flush strategy store buffer
        self._flush_store_buffer(strategy_store)

        # Log statistics
        self._log_statistics(
            signals_received=signals_received,
            trades_executed=trades_executed,
            winning_trades=winning_trades,
            total_pnl=total_pnl,
            dry_run_trades=dry_run_trades,
        )

    def get_statistics(
        self,
        signals_received: int,
        trades_executed: int,
        winning_trades: int,
        total_pnl: Decimal,
    ) -> dict[str, Any]:
        """
        Return strategy statistics dictionary.

        This method builds a statistics dictionary that can be used for
        logging, metrics, or other reporting purposes.

        Parameters
        ----------
        signals_received : int
            Total number of signals received.
        trades_executed : int
            Total number of trades executed.
        winning_trades : int
            Total number of winning trades.
        total_pnl : Decimal
            Total profit and loss.

        Returns
        -------
        dict[str, Any]
            Dictionary containing strategy statistics.

        Examples
        --------
        >>> stats = component.get_statistics(
        ...     signals_received=100,
        ...     trades_executed=50,
        ...     winning_trades=30,
        ...     total_pnl=Decimal("1250.50"),
        ... )
        >>> assert stats["signals_received"] == 100
        >>> assert stats["win_rate"] == 60.0

        """
        win_rate = (winning_trades / max(trades_executed, 1)) * 100.0

        return {
            "strategy_id": self._strategy_id,
            "instrument_id": str(self._instrument_id),
            "signals_received": signals_received,
            "trades_executed": trades_executed,
            "winning_trades": winning_trades,
            "win_rate": win_rate,
            "total_pnl": float(total_pnl),
            "execute_trades": self._execute_trades,
        }

    # -------------------------------------------------------------------------
    # Private Methods - Subscriptions
    # -------------------------------------------------------------------------

    def _subscribe_to_ml_signals(self) -> None:
        """
        Subscribe to ML signals using the configured callback.
        """
        if self._subscribe_data_callback is None:
            self._log.warning(
                "ml_strategy.subscribe_data_callback_not_configured "
                f"strategy_id={self._strategy_id}",
            )
            return

        try:
            from nautilus_trader.model.data import DataType
            from nautilus_trader.model.identifiers import ClientId

            from ml.actors.base import MLSignal

            # Build data type for ML signals (optionally scoped by source)
            metadata = {"source": self._signal_source} if self._signal_source else None
            data_type = DataType(MLSignal, metadata=metadata) if metadata else DataType(MLSignal)

            # Subscribe with or without client_id
            if self._signal_client_id is not None:
                client_id = ClientId(self._signal_client_id)
                self._subscribe_data_callback(
                    data_type=data_type,
                    client_id=client_id,
                )
            else:
                self._subscribe_data_callback(
                    data_type=data_type,
                    client_id=None,
                )
        except Exception as exc:
            self._log.error(
                "ml_strategy.subscribe_ml_signals_failed "
                f"strategy_id={self._strategy_id} error={exc!r}",
                exc_info=True,
            )

    def _subscribe_to_instrument(self) -> None:
        """
        Subscribe to instrument using the configured callback.
        """
        if self._subscribe_instrument_callback is None:
            self._log.warning(
                "ml_strategy.subscribe_instrument_callback_not_configured "
                f"strategy_id={self._strategy_id}",
            )
            return

        try:
            self._subscribe_instrument_callback(self._instrument_id)
        except Exception as exc:
            self._log.error(
                "ml_strategy.subscribe_instrument_failed "
                f"strategy_id={self._strategy_id} "
                f"instrument_id={self._instrument_id} error={exc!r}",
                exc_info=True,
            )

    def _subscribe_to_quote_ticks(self) -> None:
        """
        Subscribe to quote ticks using the configured callback.
        """
        if not self._subscribe_quote_ticks:
            return

        if self._subscribe_quote_ticks_callback is None:
            self._log.warning(
                "ml_strategy.subscribe_quote_ticks_callback_not_configured "
                f"strategy_id={self._strategy_id}",
            )
            return

        params = {"schema": self._quote_schema} if self._quote_schema else None
        try:
            self._subscribe_quote_ticks_callback(self._instrument_id, params=params)
        except Exception as exc:
            self._log.error(
                "ml_strategy.subscribe_quote_ticks_failed "
                f"strategy_id={self._strategy_id} "
                f"instrument_id={self._instrument_id} error={exc!r}",
                exc_info=True,
            )

    # -------------------------------------------------------------------------
    # Private Methods - Configuration Logging
    # -------------------------------------------------------------------------

    def _log_configuration(self) -> None:
        """
        Log strategy configuration details.
        """
        self._log.info(
            f"ML Strategy configured: instrument={self._instrument_id}, "
            f"position_size={self._position_size_pct:.1%}, "
            f"min_confidence={self._min_confidence}, "
            f"target_models={self._target_model_ids}, "
            f"aggregation={self._aggregation_mode}",
        )

    # -------------------------------------------------------------------------
    # Private Methods - Shutdown
    # -------------------------------------------------------------------------

    def _flush_store_buffer(
        self,
        strategy_store: StrategyStoreProtocol | None,
    ) -> None:
        """
        Flush the strategy store buffer.
        """
        if strategy_store is None:
            return

        try:
            strategy_store.flush()
        except Exception as exc:
            self._log.error(
                f"ml_strategy.strategy_store_flush_failed "
                f"strategy={self._strategy_id} error={exc!r}",
                exc_info=True,
            )

    def _log_statistics(
        self,
        signals_received: int,
        trades_executed: int,
        winning_trades: int,
        total_pnl: Decimal,
        dry_run_trades: int,
    ) -> None:
        """
        Log final statistics based on execution mode.
        """
        win_rate = (winning_trades / max(trades_executed, 1)) * 100.0

        if self._execute_trades:
            self._log.info(
                f"Stopping {self._strategy_id} - "
                f"Signals: {signals_received}, "
                f"Trades: {trades_executed}, "
                f"Win rate: {win_rate:.1f}%, "
                f"Total PnL: {total_pnl}",
            )
        else:
            self._log.info(
                f"Stopping {self._strategy_id} [DRY RUN MODE] - "
                f"Signals: {signals_received}, "
                f"Dry Run Trades: {dry_run_trades}, "
                f"(execute_trades=False - no actual trades executed)",
            )


__all__ = [
    "LifecycleComponent",
    "LoggerProtocol",
    "StrategyStoreProtocol",
]
