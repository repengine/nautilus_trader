"""
Strategy signal store for ML pipeline integration.

This module provides storage for strategy signals and decisions with support for batch
writes, risk tracking, and execution parameters.

"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import BIGINT
from sqlalchemy import BOOLEAN
from sqlalchemy import JSON
from sqlalchemy import Column
from sqlalchemy import Float
from sqlalchemy import Index
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

from ml.stores.base import BaseStore
from ml.stores.base import StrategySignal


if TYPE_CHECKING:
    import pandas as pd

    from ml.registry.persistence import PersistenceConfig
    from nautilus_trader.common.clock import Clock


class StrategyStore(BaseStore):
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
    ):
        """
        Initialize strategy store.

        Parameters
        ----------
        connection_string : str | None
            PostgreSQL connection string (deprecated, use persistence_config)
        persistence_config : PersistenceConfig | None
            Persistence configuration
        batch_size : int
            Maximum batch size before auto-flush
        flush_interval_ms : int
            Maximum time between flushes in milliseconds
        clock : Clock | None
            Nautilus clock for timestamps

        """
        # Handle legacy connection string parameter
        if connection_string and not persistence_config:
            from ml.registry.persistence import BackendType
            from ml.registry.persistence import PersistenceConfig

            persistence_config = PersistenceConfig(
                backend=BackendType.POSTGRES,
                connection_string=connection_string,
            )

        if persistence_config:
            from ml.registry.persistence import PersistenceManager

            self.persistence: PersistenceManager | None = PersistenceManager(persistence_config)
            self.connection_string = persistence_config.connection_string
        else:
            # Fallback for testing
            self.persistence = None
            self.connection_string = (
                connection_string or "postgresql://postgres:postgres@localhost:5432/nautilus"
            )

        self.batch_size = batch_size
        self.flush_interval_ms = flush_interval_ms
        self.clock = clock

        # Write buffer for batching
        self._write_buffer: list[StrategySignal] = []
        self._last_flush_ns = 0

        # Create engine and setup tables
        if self.connection_string:
            self.engine: Engine = create_engine(self.connection_string)
            self.metadata = MetaData()
            self._setup_tables()

    def _setup_tables(self):
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
            Column("created_at", BIGINT),  # When stored (nanoseconds)
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
        ts_init = self.clock.timestamp_ns() if self.clock else int(time.time() * 1e9)

        data = StrategySignal(
            strategy_id=strategy_id,
            instrument_id=instrument_id,
            signal_type=signal_type,
            strength=strength,
            model_predictions=model_predictions,
            risk_metrics=risk_metrics,
            execution_params=execution_params,
            _ts_event=ts_event,
            _ts_init=ts_init,
        )

        self._write_buffer.append(data)

        # Auto-flush if buffer full or time elapsed
        if len(self._write_buffer) >= self.batch_size:
            self.flush()
        elif self.clock and self._should_flush_by_time():
            self.flush()

    def write_batch(self, data: list[StrategySignal]) -> None:
        """
        Write batch of strategy signals.

        Parameters
        ----------
        data : list[StrategySignal]
            List of signals to write

        """
        if not data:
            return

        session: Any = None
        if self.persistence:
            session = self.persistence.get_session()
            if not session:
                return
        else:
            # Direct connection for testing
            session = self.engine.connect()

        try:
            # Bulk insert using VALUES for performance
            values = []
            for item in data:
                values.append(
                    {
                        "strategy_id": item.strategy_id,
                        "instrument_id": item.instrument_id,
                        "ts_event": item.ts_event,
                        "ts_init": item.ts_init,
                        "signal_type": item.signal_type,
                        "strength": item.strength,
                        "model_predictions": (
                            json.dumps(item.model_predictions) if item.model_predictions else None
                        ),
                        "risk_metrics": (
                            json.dumps(item.risk_metrics) if item.risk_metrics else None
                        ),
                        "execution_params": (
                            json.dumps(item.execution_params) if item.execution_params else None
                        ),
                        "is_live": getattr(item, "is_live", False),
                        "created_at": int(datetime.utcnow().timestamp() * 1e9),
                    },
                )

            # Use INSERT with ON CONFLICT for upsert
            stmt = insert(self.strategy_signals_table)
            stmt = stmt.on_conflict_do_update(
                index_elements=["strategy_id", "instrument_id", "ts_event"],
                set_={
                    "signal_type": stmt.excluded.signal_type,
                    "strength": stmt.excluded.strength,
                    "model_predictions": stmt.excluded.model_predictions,
                    "risk_metrics": stmt.excluded.risk_metrics,
                    "execution_params": stmt.excluded.execution_params,
                    "created_at": stmt.excluded.created_at,
                },
            )

            if session:
                session.execute(stmt, values)
                if hasattr(session, "commit"):
                    session.commit()
        finally:
            if session:
                session.close()

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
        from ml._imports import HAS_POLARS
        from ml._imports import check_ml_dependencies
        from ml._imports import pl

        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        query = f"""
        SELECT
            ts_event,
            signal_type,
            strength,
            model_predictions,
            risk_metrics,
            execution_params
        FROM ml_strategy_signals
        WHERE strategy_id = '{strategy_id}'
        AND instrument_id = '{instrument_id}'
        AND ts_event >= {start_ns}
        AND ts_event < {end_ns}
        ORDER BY ts_event
        """  # noqa: S608

        # Use Polars for efficient reading
        df = pl.read_database(query, self.connection_string or "")

        # Convert to pandas for compatibility
        return df.to_pandas()

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
        from ml._imports import HAS_POLARS
        from ml._imports import check_ml_dependencies
        from ml._imports import pl

        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        where_clause = f"WHERE ts_event >= {start_ns} AND ts_event < {end_ns}"
        if instrument_id:
            where_clause += f" AND instrument_id = '{instrument_id}'"

        query = f"""
        SELECT
            strategy_id,
            instrument_id,
            ts_event,
            signal_type,
            strength,
            model_predictions,
            risk_metrics
        FROM ml_strategy_signals
        {where_clause}
        ORDER BY ts_event
        """  # noqa: S608

        df = pl.read_database(query, self.connection_string or "")
        return df.to_pandas()

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
        from ml._imports import HAS_POLARS
        from ml._imports import check_ml_dependencies
        from ml._imports import pl

        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        query = f"""
        SELECT
            strategy_id,
            ts_event,
            signal_type,
            strength,
            risk_metrics
        FROM ml_strategy_signals
        WHERE instrument_id = '{instrument_id}'
        ORDER BY ts_event DESC
        LIMIT {limit}
        """  # noqa: S608

        df = pl.read_database(query, self.connection_string or "")
        return df.to_pandas()

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
        with self.engine.connect() as conn:
            # Build WHERE clause
            where_parts = []
            if start_ns:
                where_parts.append(f"ts_event >= {start_ns}")
            if end_ns:
                where_parts.append(f"ts_event < {end_ns}")

            where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

            # Get statistics
            query = text(
                f"""
                SELECT
                    COUNT(*) as total_signals,
                    COUNT(DISTINCT strategy_id) as unique_strategies,
                    COUNT(DISTINCT instrument_id) as unique_instruments,
                    SUM(CASE WHEN signal_type = 'BUY' THEN 1 ELSE 0 END) as buy_signals,
                    SUM(CASE WHEN signal_type = 'SELL' THEN 1 ELSE 0 END) as sell_signals,
                    SUM(CASE WHEN signal_type = 'HOLD' THEN 1 ELSE 0 END) as hold_signals,
                    AVG(strength) as avg_strength,
                    MIN(ts_event) as min_ts,
                    MAX(ts_event) as max_ts
                FROM ml_strategy_signals
                {where_clause}
            """,
            )

            result = conn.execute(query).fetchone()

            if result:
                return {
                    "total_signals": result[0] or 0,
                    "unique_strategies": result[1] or 0,
                    "unique_instruments": result[2] or 0,
                    "buy_signals": result[3] or 0,
                    "sell_signals": result[4] or 0,
                    "hold_signals": result[5] or 0,
                    "avg_strength": float(result[6]) if result[6] else 0.0,
                    "min_timestamp_ns": result[7] or 0,
                    "max_timestamp_ns": result[8] or 0,
                }

        return {
            "total_signals": 0,
            "unique_strategies": 0,
            "unique_instruments": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "hold_signals": 0,
            "avg_strength": 0.0,
            "min_timestamp_ns": 0,
            "max_timestamp_ns": 0,
        }

    def flush(self) -> None:
        """
        Flush pending signals to storage.
        """
        if self._write_buffer:
            self.write_batch(self._write_buffer)
            self._write_buffer.clear()
            if self.clock:
                self._last_flush_ns = self.clock.timestamp_ns()

    def _should_flush_by_time(self) -> bool:
        """
        Check if flush is needed based on time.
        """
        if not self.clock or not self._last_flush_ns:
            return False

        elapsed_ms = (self.clock.timestamp_ns() - self._last_flush_ns) / 1e6
        return elapsed_ms >= self.flush_interval_ms

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
        with self.engine.begin() as conn:
            delete_stmt = self.strategy_signals_table.delete()

            if strategy_id:
                delete_stmt = delete_stmt.where(
                    self.strategy_signals_table.c.strategy_id == strategy_id,
                )

            if instrument_id:
                delete_stmt = delete_stmt.where(
                    self.strategy_signals_table.c.instrument_id == instrument_id,
                )

            conn.execute(delete_stmt)

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
        where_parts = [f"strategy_id = '{strategy_id}'"]
        if start_ns:
            where_parts.append(f"ts_event >= {start_ns}")
        if end_ns:
            where_parts.append(f"ts_event < {end_ns}")

        where_clause = f"WHERE {' AND '.join(where_parts)}"

        with self.engine.connect() as conn:
            query = text(
                f"""
                SELECT
                    COUNT(*) as signal_count,
                    SUM(CASE WHEN signal_type = 'BUY' THEN 1 ELSE 0 END) as buy_count,
                    SUM(CASE WHEN signal_type = 'SELL' THEN 1 ELSE 0 END) as sell_count,
                    SUM(CASE WHEN signal_type = 'HOLD' THEN 1 ELSE 0 END) as hold_count,
                    AVG(strength) as avg_strength,
                    STDDEV(strength) as std_strength,
                    MIN(strength) as min_strength,
                    MAX(strength) as max_strength
                FROM ml_strategy_signals
                {where_clause}
            """,
            )

            result = conn.execute(query).fetchone()

            if result:
                return {
                    "signal_count": result[0] or 0,
                    "buy_count": result[1] or 0,
                    "sell_count": result[2] or 0,
                    "hold_count": result[3] or 0,
                    "avg_strength": float(result[4]) if result[4] else 0.0,
                    "std_strength": float(result[5]) if result[5] else 0.0,
                    "min_strength": float(result[6]) if result[6] else 0.0,
                    "max_strength": float(result[7]) if result[7] else 0.0,
                }

        return {
            "signal_count": 0,
            "buy_count": 0,
            "sell_count": 0,
            "hold_count": 0,
            "avg_strength": 0.0,
            "std_strength": 0.0,
            "min_strength": 0.0,
            "max_strength": 0.0,
        }

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
        where_parts = []
        if strategy_id:
            where_parts.append(f"strategy_id = '{strategy_id}'")
        if start_ns:
            where_parts.append(f"ts_event >= {start_ns}")
        if end_ns:
            where_parts.append(f"ts_event < {end_ns}")

        where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

        with self.engine.connect() as conn:
            query = text(
                f"""
                SELECT signal_type, COUNT(*) as count
                FROM ml_strategy_signals
                {where_clause}
                GROUP BY signal_type
            """,
            )

            result = conn.execute(query).fetchall()

            distribution = {}
            for signal_type, count in result:
                distribution[signal_type] = count

        return distribution

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
        with self.engine.begin() as conn:
            # Calculate metrics for period
            query = text(
                """
                SELECT
                    COUNT(*) as signal_count,
                    SUM(CASE WHEN signal_type = 'BUY' THEN 1 ELSE 0 END) as buy_count,
                    SUM(CASE WHEN signal_type = 'SELL' THEN 1 ELSE 0 END) as sell_count,
                    SUM(CASE WHEN signal_type = 'HOLD' THEN 1 ELSE 0 END) as hold_count,
                    AVG(strength) as avg_strength,
                    AVG((risk_metrics->>'risk_score')::float) as avg_risk_score
                FROM ml_strategy_signals
                WHERE strategy_id = :strategy_id
                AND ts_event >= :period_start
                AND ts_event < :period_end
            """,
            )

            result = conn.execute(
                query,
                {
                    "strategy_id": strategy_id,
                    "period_start": period_start,
                    "period_end": period_end,
                },
            ).fetchone()

            if result and result[0] > 0:  # Has signals
                # Upsert performance record
                stmt = insert(self.strategy_performance_table)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["strategy_id", "period_start"],
                    set_={
                        "period_end": period_end,
                        "signal_count": result[0],
                        "buy_count": result[1],
                        "sell_count": result[2],
                        "hold_count": result[3],
                        "avg_strength": result[4],
                        "avg_risk_score": result[5],
                        "created_at": int(datetime.utcnow().timestamp() * 1e9),
                    },
                )

                conn.execute(
                    stmt,
                    {
                        "strategy_id": strategy_id,
                        "period_start": period_start,
                        "period_end": period_end,
                        "signal_count": result[0],
                        "buy_count": result[1],
                        "sell_count": result[2],
                        "hold_count": result[3],
                        "avg_strength": result[4],
                        "avg_risk_score": result[5],
                        "created_at": int(datetime.utcnow().timestamp() * 1e9),
                    },
                )
