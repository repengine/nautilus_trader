"""
Strategy signal store for ML pipeline integration.

This module provides storage for strategy signals and decisions with support for batch
writes, risk tracking, and execution parameters.

"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Literal, cast

from sqlalchemy import BIGINT
from sqlalchemy import BOOLEAN
from sqlalchemy import JSON
from sqlalchemy import Column
from sqlalchemy import Float
from sqlalchemy import Index
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy.engine import Engine
from typing_extensions import override

from ml.common.message_bus import BusPublisherMixin
from ml.common.message_bus import MessagePublisherProtocol
from ml.core.db_engine import EngineManager
from ml.stores.base import BaseStore
from ml.stores.base import StrategySignal
from ml.stores.mixins import BufferedStoreMixin
from ml.stores.mixins import DataRegistryMixin
from ml.stores.mixins import EngineInitMixin
from ml.stores.mixins import HealthMixin
from ml.stores.mixins import ReadQueryMixin
from ml.stores.mixins import SQLUpsertMixin
from ml.stores.mixins import StoreInitMixin
from ml.stores.services.strategy_services import StrategySignalClearService
from ml.stores.services.strategy_services import StrategySignalEventService
from ml.stores.services.strategy_services import StrategySignalQueryService
from ml.stores.services.strategy_services import StrategySignalStatsService
from ml.stores.services.strategy_services import StrategySignalWriteService


if TYPE_CHECKING:
    import pandas as pd
    from nautilus_trader.common.clock import Clock

    from ml.registry.persistence import PersistenceConfig
    from ml.registry.protocols import RegistryProtocol


logger = logging.getLogger(__name__)

__all__ = [
    "StrategySignal",
    "StrategyStore",
]


# Prometheus metrics are optional; type as Any for strict typing compatibility
data_events_total: Any
try:
    from ml.common.metrics import data_events_total as data_events_total
except Exception:
    data_events_total = None


# Backwards-compat: expose a module-level create_engine symbol for tests to monkeypatch.
def create_engine(connection_string: str) -> Engine:
    """
    Compatibility wrapper returning a SQLAlchemy Engine.

    Parameters
    ----------
    connection_string : str
        Database URL.

    Returns
    -------
    Engine
        SQLAlchemy engine managed by EngineManager.

    """
    return EngineManager.get_engine(connection_string)


class StrategyStore(
    HealthMixin,
    BufferedStoreMixin,
    SQLUpsertMixin,
    ReadQueryMixin,
    BaseStore,
    BusPublisherMixin,
    DataRegistryMixin,
    EngineInitMixin,
    StoreInitMixin,
):
    """
    Store for strategy signals and decisions with PostgreSQL backend.

    Tracks strategy signals with model attributions, risk metrics, and execution
    parameters for both backtesting and live trading.

    """

    def __init__(
        self,
        connection_string: str | None = None,
        persistence_config: PersistenceConfig | None = None,
        batch_size: int = 1000,
        flush_interval_ms: int = 100,
        clock: Clock | None = None,
        persistence_manager: object | None = None,
        flush_interval_seconds: float | None = None,
        enable_publishing: bool = False,
        publisher: MessagePublisherProtocol | None = None,
        publish_mode: Literal["batch", "row", "both"] = "batch",
        **_: object,
    ) -> None:
        """
        Initialize strategy store.

        Parameters
        ----------
        connection_string : str | None
            PostgreSQL connection string (deprecated, prefer `persistence_config`).
        persistence_config : PersistenceConfig | None
            Persistence backend configuration.
        batch_size : int
            Maximum batch size before auto-flush.
        flush_interval_ms : int
            Maximum time between flushes in milliseconds.
        clock : Clock | None
            Nautilus clock for timestamps.
        persistence_manager : Any | None
            Optional persistence manager (used in tests).
        flush_interval_seconds : float | None
            Alternative flush interval in seconds (overrides `flush_interval_ms`).
        enable_publishing : bool, optional
            When True, publish store events to the optional message bus.
        publisher : MessagePublisherProtocol | None, optional
            Publisher implementation used when `enable_publishing` is True.
        publish_mode : {"batch", "row", "both"}, optional
            Controls whether to publish batch summaries, per-row events, or both. Defaults to "batch".

        """
        # Shared initialization (connection, persistence, bus, engine, flush settings)
        self._init_store_common(
            connection_string=connection_string,
            persistence_config=persistence_config,
            batch_size=batch_size,
            flush_interval_ms=flush_interval_ms,
            flush_interval_seconds=flush_interval_seconds,
            clock=clock,
            enable_publishing=enable_publishing,
            publisher=publisher,
            publish_mode=publish_mode,
            persistence_manager=persistence_manager,
        )

        # Write buffer for batching
        self._write_buffer: list[StrategySignal] = []
        # Back-compat: expose `_buffer` alias used by older tests
        self._buffer: list[StrategySignal] = self._write_buffer

        # DataRegistry for event emission (lazy initialization)
        self._data_registry: RegistryProtocol | None = None
        # Engine + tables already initialized by _init_store_common
        # Extracted services (internal composition; public API unchanged)
        self._write_service = StrategySignalWriteService(self, logger)
        self._query_service = StrategySignalQueryService(self)
        self._stats_service = StrategySignalStatsService(self)
        self._event_service = StrategySignalEventService(self, logger)
        self._clear_service = StrategySignalClearService(self)

        # Optional circuit breaker injected by actors/services

        from ml.stores.protocols import CircuitBreakerProtocol as _CBP

        self._circuit_breaker: _CBP | None = None

    def _get_data_registry(self) -> RegistryProtocol | None:
        # Delegate to shared mixin
        return DataRegistryMixin._get_data_registry(self)

    # -- SQL identifier safety -------------------------------------------------
    # Thin wrappers for shared helpers to preserve test compatibility
    def _safe_identifier(self, name: str, allowed: set[str]) -> str:
        return ReadQueryMixin._safe_identifier(self, name, allowed)

    def _safe_table(self, base: str, allowed: set[str] | None = None) -> str:
        if allowed is None:
            allowed = {"ml_strategy_signals", "ml_strategy_performance"}
        return ReadQueryMixin._safe_table(self, base, allowed)

    def _setup_tables(self) -> None:
        """
        Create strategy_signals table if it doesn't exist.
        """
        # Define strategy_signals table
        self.strategy_signals_table = Table(
            "ml_strategy_signals",
            self.metadata,
            Column("strategy_id", String(255), primary_key=True),
            Column("instrument_id", String(100), primary_key=True),
            Column("ts_event", BIGINT, primary_key=True),  # Nautilus convention: nanoseconds
            Column("ts_init", BIGINT),
            Column("signal_type", String(20), nullable=False),  # BUY, SELL, HOLD
            Column("strength", Float, nullable=False),
            Column("model_predictions", JSON),  # Model ID -> prediction mapping
            Column("risk_metrics", JSON),  # Risk calculations
            Column("execution_params", JSON),  # Stop loss, take profit, etc.
            Column("is_live", BOOLEAN, default=False),
            Column("created_at", BIGINT),  # Dev table; DB default used in prod
            Index("idx_ml_strategy_signals_lookup", "strategy_id", "instrument_id", "ts_event"),
            Index("idx_ml_strategy_signals_type", "signal_type"),
            Index("idx_ml_strategy_signals_live", "is_live"),
        )

        # Strategy performance tracking table
        self.strategy_performance_table = Table(
            "ml_strategy_performance",
            self.metadata,
            Column("strategy_id", String(255), primary_key=True),
            Column("period_start", BIGINT, primary_key=True),
            Column("period_end", BIGINT),
            Column("signal_count", BIGINT),
            Column("buy_count", BIGINT),
            Column("sell_count", BIGINT),
            Column("hold_count", BIGINT),
            Column("avg_strength", Float),
            Column("avg_risk_score", Float),
            Column("created_at", BIGINT),
            Index("idx_ml_strategy_performance", "strategy_id", "period_start"),
        )

        # Create tables
        self.metadata.create_all(self.engine)

        # Ensure default partition exists for partitioned deployments (idempotent)
        try:
            from sqlalchemy import text as _text

            with self.engine.begin() as _conn:
                _conn.execute(
                    _text(
                        "CREATE TABLE IF NOT EXISTS ml_strategy_signals_default "
                        "PARTITION OF ml_strategy_signals DEFAULT",
                    ),
                )
        except Exception as exc:
            # Non-fatal when running against non-partitioned dev tables
            logger.debug("Default partition ensure skipped for strategy signals: %s", exc)

    def write_signal(
        self,
        strategy_id: str,
        instrument_id: str,
        signal_type: str,
        strength: float,
        model_predictions: dict[str, float],
        risk_metrics: dict[str, float],
        execution_params: dict[str, Any],
        ts_event: int,
        is_live: bool = False,
    ) -> None:
        """
        Write single strategy signal.

        Parameters
        ----------
        strategy_id : str
            Strategy identifier
        instrument_id : str
            Instrument identifier
        signal_type : str
            Signal type (BUY, SELL, HOLD)
        strength : float
            Signal strength/confidence
        model_predictions : dict[str, float]
            Model predictions used
        risk_metrics : dict[str, float]
            Risk metrics calculated
        execution_params : dict[str, Any]
            Execution parameters
        ts_event : int
            Event timestamp in nanoseconds
        is_live : bool
            Whether this is live trading

        """
        ts_init_raw = self.clock.timestamp_ns() if self.clock else time.time_ns()

        # Normalize timestamps via centralized sanitizer
        from ml.common.timestamps import sanitize_timestamp_ns

        ts_event_norm = sanitize_timestamp_ns(
            int(ts_event),
            logger=logger,
            context="StrategyStore.write_signal:ts_event",
        )
        ts_init = sanitize_timestamp_ns(
            int(ts_init_raw),
            logger=logger,
            context="StrategyStore.write_signal:ts_init",
        )

        data = StrategySignal(
            strategy_id=strategy_id,
            instrument_id=instrument_id,
            signal_type=signal_type,
            strength=strength,
            model_predictions=model_predictions,
            risk_metrics=risk_metrics,
            execution_params=execution_params,
            _ts_event=ts_event_norm,
            _ts_init=ts_init,
        )

        self._write_buffer.append(data)

        # Auto-flush if buffer full or time elapsed
        if len(self._write_buffer) >= self.batch_size:
            self.flush()
        elif self.clock and self._should_flush_by_time():
            self.flush()

    @override
    def write_batch(self, data: list[StrategySignal]) -> None:
        """
        Write batch of strategy signals.

        Preserves the historical patch point by delegating through
        `self._execute_write(values)` so tests can monkeypatch it.

        Parameters
        ----------
        data : list[StrategySignal]
            List of signals to write

        """
        if not data:
            return

        # Track stage boundary for observability (cold path only)
        import time

        ts_stage_start = time.time_ns()

        # Delegate to write service to avoid duplication and preserve patch points
        self._write_service.write_batch(data)

        # Record observability data (off hot path - background processing only)
        ts_stage_end = time.time_ns()
        # Use the first item's instrument_id as representative
        instrument_id = data[0].instrument_id if data else "unknown"
        self._record_observability_stage_boundary(
            stage="strategy_signal_storage",
            instrument_id=instrument_id,
            ts_stage_start=ts_stage_start,
            ts_stage_end=ts_stage_end,
            row_count=len(data),
        )

    def _record_observability_stage_boundary(
        self,
        *,
        stage: str,
        instrument_id: str,
        ts_stage_start: int,
        ts_stage_end: int,
        row_count: int = 1,
    ) -> None:
        """
        Record observability data via centralized helper (cold path only).
        """
        from ml.common.observability_utils import record_stage_boundary as _rec

        obs_service = getattr(self, "_observability_service", None)
        _rec(
            obs_service,
            component="strategy_store",
            instrument_id=instrument_id,
            stage=stage,
            ts_stage_start=ts_stage_start,
            ts_stage_end=ts_stage_end,
            row_count=row_count,
        )

    def _execute_write(self, values: list[dict[str, Any]]) -> None:  # pragma: no cover
        """
        Patch point preserved; delegates to write service.
        """
        self._write_service.execute_write(values)

    # Backwards-compatible alias used in some tests
    def write_signals(self, data: list[StrategySignal]) -> None:
        self.write_batch(data)

    def read_signals(
        self,
        strategy_id: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> pd.DataFrame:
        """
        Read strategy signals for analysis.

        Parameters
        ----------
        strategy_id : str
            Strategy identifier
        instrument_id : str
            Instrument identifier
        start_ns : int
            Start timestamp in nanoseconds
        end_ns : int
            End timestamp in nanoseconds

        Returns
        -------
        pd.DataFrame
            Signals within range

        """
        from typing import cast as _cast

        import pandas as pd

        return _cast(
            pd.DataFrame,
            self._query_service.read_signals(
                strategy_id=strategy_id,
                instrument_id=instrument_id,
                start_ns=start_ns,
                end_ns=end_ns,
            ),
        )

    @override
    def read_range(
        self,
        start_ns: int,
        end_ns: int,
        instrument_id: str | None = None,
    ) -> pd.DataFrame:
        """
        Read all signals in time range.

        Parameters
        ----------
        start_ns : int
            Start timestamp in nanoseconds
        end_ns : int
            End timestamp in nanoseconds
        instrument_id : str | None
            Optional instrument filter

        Returns
        -------
        pd.DataFrame
            Signals within range

        """
        from typing import cast as _cast

        import pandas as pd

        return _cast(
            pd.DataFrame,
            self._query_service.read_range(
                start_ns=start_ns,
                end_ns=end_ns,
                instrument_id=instrument_id,
            ),
        )

    @override
    def get_latest(
        self,
        instrument_id: str,
        limit: int = 1,
    ) -> pd.DataFrame:
        """
        Get latest signals for an instrument.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        limit : int
            Maximum number of entries

        Returns
        -------
        pd.DataFrame
            Latest signals

        """
        from typing import cast as _cast

        import pandas as pd

        return _cast(
            pd.DataFrame,
            self._query_service.get_latest(instrument_id=instrument_id, limit=limit),
        )

    @override
    def get_statistics(
        self,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> dict[str, Any]:
        """
        Get signal statistics.

        Parameters
        ----------
        start_ns : int | None
            Optional start timestamp
        end_ns : int | None
            Optional end timestamp

        Returns
        -------
        dict[str, Any]
            Statistics dictionary

        """
        return self._stats_service.get_statistics(start_ns=start_ns, end_ns=end_ns)

    def flush(self) -> None:
        """
        Delegate to shared buffered flush behavior.
        """
        from ml.stores.mixins import BufferedStoreMixin as _BSM

        _BSM.flush(self)

    def _emit_signal_events(self, signals: list[StrategySignal]) -> None:
        """
        Delegate to event service (non-blocking).
        """
        try:
            self._event_service.emit_signal_events(signals)
        except Exception:
            logger.debug("Signal event emission failed", exc_info=True)

    # Time-based flush decision provided by BufferedStoreMixin

    # Health check provided by BufferedStoreMixin

    # Wrapper used by BufferedStoreMixin.flush
    def _emit_events(self, signals: list[StrategySignal]) -> None:
        self._emit_signal_events(signals)

    def clear_signals(
        self,
        strategy_id: str | None = None,
        instrument_id: str | None = None,
    ) -> None:
        """
        Clear stored signals.

        Parameters
        ----------
        strategy_id : str | None
            Clear only for specific strategy
        instrument_id : str | None
            Clear only for specific instrument

        """
        self._clear_service.clear(strategy_id=strategy_id, instrument_id=instrument_id)

    def get_strategy_performance(
        self,
        strategy_id: str,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> dict[str, Any]:
        """
        Get performance metrics for a strategy.

        Parameters
        ----------
        strategy_id : str
            Strategy identifier
        start_ns : int | None
            Optional start timestamp
        end_ns : int | None
            Optional end timestamp

        Returns
        -------
        dict[str, Any]
            Performance metrics

        """
        return self._stats_service.get_strategy_performance(
            strategy_id=strategy_id,
            start_ns=start_ns,
            end_ns=end_ns,
        )

    def get_signal_distribution(
        self,
        strategy_id: str | None = None,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> dict[str, int]:
        """
        Get signal type distribution.

        Parameters
        ----------
        strategy_id : str | None
            Optional strategy filter
        start_ns : int | None
            Optional start timestamp
        end_ns : int | None
            Optional end timestamp

        Returns
        -------
        dict[str, int]
            Signal type counts

        """
        return self._stats_service.get_signal_distribution(
            strategy_id=strategy_id,
            start_ns=start_ns,
            end_ns=end_ns,
        )

    def update_performance_metrics(
        self,
        strategy_id: str,
        period_start: int,
        period_end: int,
    ) -> None:
        """
        Update aggregated performance metrics.

        Parameters
        ----------
        strategy_id : str
            Strategy identifier
        period_start : int
            Period start timestamp in nanoseconds
        period_end : int
            Period end timestamp in nanoseconds

        """
        self._stats_service.update_performance_metrics(
            strategy_id=strategy_id,
            period_start=period_start,
            period_end=period_end,
            engine=self.engine,
            performance_table=self.strategy_performance_table,
        )

    def _get_connection(self) -> object:  # pragma: no cover (test hook for patching)
        """
        Return a connection context manager (patchable in tests).
        """
        return self.engine.connect()

    # -------------------------------------------------------------------------------------
    # Compatibility reads and aliases
    # -------------------------------------------------------------------------------------

    def read_active_signals(
        self,
        strategy_id: str | None = None,
        instrument_id: str | None = None,
        hours_back: int = 1,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Read currently active strategy signals within a recent time window.

        Returns a DataFrame with columns:
        strategy_id, instrument_id, signal_type, strength, model_predictions,
        risk_metrics, execution_params, ts_event, ts_init.

        """
        from typing import cast as _cast

        import pandas as pd

        return _cast(
            pd.DataFrame,
            self._query_service.read_active_signals(
                hours_back=hours_back,
                limit=limit,
                strategy_id=strategy_id,
                instrument_id=instrument_id,
            ),
        )

    # Backwards-compatible public API used in some tests
    def get_signals(
        self,
        strategy_id: str,
        start_ns: int,
        end_ns: int,
        instrument_id: str | None = None,
    ) -> pd.DataFrame:
        """
        Return strategy signals in a time range.

        This is a compatibility shim delegating to read_signals when instrument is
        provided, and otherwise reading across instruments.

        """
        # Accept seconds or nanoseconds; normalize to ns
        from ml.common.timestamps import sanitize_timestamp_ns

        start_ns = sanitize_timestamp_ns(
            int(start_ns),
            logger=logger,
            context="StrategyStore.get_signals:start",
        )
        end_ns = sanitize_timestamp_ns(
            int(end_ns),
            logger=logger,
            context="StrategyStore.get_signals:end",
        )

        from typing import cast as _cast

        import pandas as pd

        return _cast(
            pd.DataFrame,
            self._query_service.get_signals(
                strategy_id=strategy_id,
                start_ns=start_ns,
                end_ns=end_ns,
                instrument_id=instrument_id,
            ),
        )

    def store_decision(self, *args: Any, **kwargs: Any) -> None:
        """
        Backward-compatible alias for write_signal.

        Accepts legacy fields like action/confidence/features and maps them to the
        current write_signal signature.

        """
        if args:
            self.write_signal(*args, **kwargs)
            return
        strategy_id = kwargs.get("strategy_id")
        instrument_id = kwargs.get("instrument_id")
        ts_event = kwargs.get("ts_event")
        action = kwargs.get("action", "")
        confidence = kwargs.get("confidence", 0.0)
        features = kwargs.get("features", {})
        if None in {strategy_id, instrument_id, ts_event}:
            self.write_signal(*args, **kwargs)
            return
        signal_type = str(action).lower() if action else "neutral"
        strength = float(confidence)
        model_predictions: dict[str, float] = {}
        risk_metrics: dict[str, float] = {}
        execution_params = {"features": features}
        self.write_signal(
            strategy_id=str(strategy_id),
            instrument_id=str(instrument_id),
            signal_type=signal_type,
            strength=strength,
            model_predictions=model_predictions,
            risk_metrics=risk_metrics,
            execution_params=execution_params,
            ts_event=int(cast(int, ts_event)),
        )

    # Attributes initialized via StoreInitMixin
    batch_size: int
    flush_interval_ms: int
    clock: Clock | None
    connection_string: str | None
    persistence: object | None
    _last_flush_ns: int
