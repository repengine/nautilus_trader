"""
Model prediction store for ML pipeline integration.

This module provides storage for model predictions with support for batch writes, time-
partitioned queries, and performance tracking.

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
from ml.stores.base import ModelPrediction


if TYPE_CHECKING:
    import pandas as pd

    from ml.registry.persistence import PersistenceConfig
    from nautilus_trader.common.clock import Clock


class ModelStore(BaseStore):
    """
    Store for model predictions with PostgreSQL backend.

    Handles both historical and live model predictions with efficient batching,
    partitioning, and performance tracking.

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
        Initialize model store.

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
        self._write_buffer: list[ModelPrediction] = []
        self._last_flush_ns = 0

        # Create engine and setup tables
        if self.connection_string:
            self.engine: Engine = create_engine(self.connection_string)
            self.metadata = MetaData()
            self._setup_tables()

    def _setup_tables(self):
        """
        Create model_predictions table if it doesn't exist.
        """
        # Define model_predictions table
        self.model_predictions_table = Table(
            "ml_model_predictions",
            self.metadata,
            Column("model_id", String(255), primary_key=True),
            Column("instrument_id", String(100), primary_key=True),
            Column("ts_event", BIGINT, primary_key=True),  # Nautilus convention: nanoseconds
            Column("ts_init", BIGINT),
            Column("prediction", Float, nullable=False),
            Column("confidence", Float),
            Column("features_used", JSON),  # Feature values at prediction time
            Column("inference_time_ms", Float),
            Column("is_live", BOOLEAN, default=False),
            Column("created_at", BIGINT),  # When stored (nanoseconds)
            Index("idx_ml_model_predictions_lookup", "model_id", "instrument_id", "ts_event"),
            Index("idx_ml_model_predictions_live", "is_live"),
        )

        # Create tables
        self.metadata.create_all(self.engine)

    def write_prediction(
        self,
        model_id: str,
        instrument_id: str,
        prediction: float,
        confidence: float,
        features: dict[str, float],
        inference_time_ms: float,
        ts_event: int,
        is_live: bool = False,
    ) -> None:
        """
        Write single model prediction.

        Parameters
        ----------
        model_id : str
            Model identifier
        instrument_id : str
            Instrument identifier
        prediction : float
            Model prediction value
        confidence : float
            Prediction confidence
        features : dict[str, float]
            Feature values used
        inference_time_ms : float
            Inference latency
        ts_event : int
            Event timestamp in nanoseconds
        is_live : bool
            Whether this is live inference

        """
        ts_init = self.clock.timestamp_ns() if self.clock else int(time.time() * 1e9)

        data = ModelPrediction(
            model_id=model_id,
            instrument_id=instrument_id,
            prediction=prediction,
            confidence=confidence,
            features_used=features,
            inference_time_ms=inference_time_ms,
            _ts_event=ts_event,
            _ts_init=ts_init,
        )

        self._write_buffer.append(data)

        # Auto-flush if buffer full or time elapsed
        if len(self._write_buffer) >= self.batch_size:
            self.flush()
        elif self.clock and self._should_flush_by_time():
            self.flush()

    def write_batch(self, data: list[ModelPrediction]) -> None:
        """
        Write batch of model predictions.

        Parameters
        ----------
        data : list[ModelPrediction]
            List of predictions to write

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
                        "model_id": item.model_id,
                        "instrument_id": item.instrument_id,
                        "ts_event": item.ts_event,
                        "ts_init": item.ts_init,
                        "prediction": item.prediction,
                        "confidence": item.confidence,
                        "features_used": (
                            json.dumps(item.features_used) if item.features_used else None
                        ),
                        "inference_time_ms": item.inference_time_ms,
                        "is_live": getattr(item, "is_live", False),
                        "created_at": int(datetime.utcnow().timestamp() * 1e9),
                    },
                )

            # Use INSERT with ON CONFLICT for upsert
            stmt = insert(self.model_predictions_table)
            stmt = stmt.on_conflict_do_update(
                index_elements=["model_id", "instrument_id", "ts_event"],
                set_={
                    "prediction": stmt.excluded.prediction,
                    "confidence": stmt.excluded.confidence,
                    "features_used": stmt.excluded.features_used,
                    "inference_time_ms": stmt.excluded.inference_time_ms,
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

    def read_predictions(
        self,
        model_id: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> pd.DataFrame:
        """
        Read predictions for analysis.

        Parameters
        ----------
        model_id : str
            Model identifier
        instrument_id : str
            Instrument identifier
        start_ns : int
            Start timestamp in nanoseconds
        end_ns : int
            End timestamp in nanoseconds

        Returns
        -------
        pd.DataFrame
            Predictions within range

        """
        from ml._imports import HAS_POLARS
        from ml._imports import check_ml_dependencies
        from ml._imports import pl

        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        query = f"""
        SELECT
            ts_event,
            prediction,
            confidence,
            features_used,
            inference_time_ms
        FROM ml_model_predictions
        WHERE model_id = '{model_id}'
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
        Read all predictions in time range.

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
            Predictions within range

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
            model_id,
            instrument_id,
            ts_event,
            prediction,
            confidence,
            inference_time_ms
        FROM ml_model_predictions
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
        Get latest predictions for an instrument.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        limit : int
            Maximum number of entries

        Returns
        -------
        pd.DataFrame
            Latest predictions

        """
        from ml._imports import HAS_POLARS
        from ml._imports import check_ml_dependencies
        from ml._imports import pl

        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        query = f"""
        SELECT
            model_id,
            ts_event,
            prediction,
            confidence,
            inference_time_ms
        FROM ml_model_predictions
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
        Get prediction statistics.

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
                    COUNT(*) as total_predictions,
                    COUNT(DISTINCT model_id) as unique_models,
                    COUNT(DISTINCT instrument_id) as unique_instruments,
                    AVG(inference_time_ms) as avg_inference_ms,
                    MAX(inference_time_ms) as max_inference_ms,
                    MIN(ts_event) as min_ts,
                    MAX(ts_event) as max_ts
                FROM ml_model_predictions
                {where_clause}
            """,
            )

            result = conn.execute(query).fetchone()

            if result:
                return {
                    "total_predictions": result[0] or 0,
                    "unique_models": result[1] or 0,
                    "unique_instruments": result[2] or 0,
                    "avg_inference_ms": float(result[3]) if result[3] else 0.0,
                    "max_inference_ms": float(result[4]) if result[4] else 0.0,
                    "min_timestamp_ns": result[5] or 0,
                    "max_timestamp_ns": result[6] or 0,
                }

        return {
            "total_predictions": 0,
            "unique_models": 0,
            "unique_instruments": 0,
            "avg_inference_ms": 0.0,
            "max_inference_ms": 0.0,
            "min_timestamp_ns": 0,
            "max_timestamp_ns": 0,
        }

    def flush(self) -> None:
        """
        Flush pending predictions to storage.
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

    def is_healthy(self) -> bool:
        """
        Check if the model store is healthy and accessible.

        Returns
        -------
        bool
            True if store is healthy, False otherwise

        """
        try:
            # Try a simple query to verify connection
            if self.engine:
                with self.engine.connect() as conn:
                    from sqlalchemy import text
                    result = conn.execute(text("SELECT 1"))
                    return result is not None
            return True  # If no engine, assume healthy (in-memory mode)
        except Exception:
            return False

    def clear_predictions(
        self,
        model_id: str | None = None,
        instrument_id: str | None = None,
    ) -> None:
        """
        Clear stored predictions.

        Parameters
        ----------
        model_id : str | None
            Clear only for specific model
        instrument_id : str | None
            Clear only for specific instrument

        """
        with self.engine.begin() as conn:
            delete_stmt = self.model_predictions_table.delete()

            if model_id:
                delete_stmt = delete_stmt.where(self.model_predictions_table.c.model_id == model_id)

            if instrument_id:
                delete_stmt = delete_stmt.where(
                    self.model_predictions_table.c.instrument_id == instrument_id,
                )

            conn.execute(delete_stmt)

    def get_model_performance(
        self,
        model_id: str,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> dict[str, Any]:
        """
        Get performance metrics for a model.

        Parameters
        ----------
        model_id : str
            Model identifier
        start_ns : int | None
            Optional start timestamp
        end_ns : int | None
            Optional end timestamp

        Returns
        -------
        dict[str, Any]
            Performance metrics

        """
        where_parts = [f"model_id = '{model_id}'"]
        if start_ns:
            where_parts.append(f"ts_event >= {start_ns}")
        if end_ns:
            where_parts.append(f"ts_event < {end_ns}")

        where_clause = f"WHERE {' AND '.join(where_parts)}"

        with self.engine.connect() as conn:
            query = text(
                f"""
                SELECT
                    COUNT(*) as prediction_count,
                    AVG(confidence) as avg_confidence,
                    STDDEV(confidence) as std_confidence,
                    AVG(inference_time_ms) as avg_latency_ms,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY inference_time_ms) as p50_latency_ms,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY inference_time_ms) as p95_latency_ms,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY inference_time_ms) as p99_latency_ms
                FROM ml_model_predictions
                {where_clause}
            """,
            )

            result = conn.execute(query).fetchone()

            if result:
                return {
                    "prediction_count": result[0] or 0,
                    "avg_confidence": float(result[1]) if result[1] else 0.0,
                    "std_confidence": float(result[2]) if result[2] else 0.0,
                    "avg_latency_ms": float(result[3]) if result[3] else 0.0,
                    "p50_latency_ms": float(result[4]) if result[4] else 0.0,
                    "p95_latency_ms": float(result[5]) if result[5] else 0.0,
                    "p99_latency_ms": float(result[6]) if result[6] else 0.0,
                }

        return {
            "prediction_count": 0,
            "avg_confidence": 0.0,
            "std_confidence": 0.0,
            "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
        }
