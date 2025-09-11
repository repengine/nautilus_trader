"""
Model prediction store for ML pipeline integration.

This module provides storage for model predictions with support for batch writes, time-
partitioned queries, and performance tracking.

"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

from sqlalchemy import BIGINT
from sqlalchemy import BOOLEAN
from sqlalchemy import JSON
from sqlalchemy import Column
from sqlalchemy import Float
from sqlalchemy import Index
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import text
from sqlalchemy.engine import Engine
from typing_extensions import override

from ml._imports import HAS_PROMETHEUS
from ml._imports import Counter
from ml.common.message_bus import BusPublisherMixin
from ml.common.message_bus import MessagePublisherProtocol
from ml.config.events import EventStatus
from ml.config.events import Stage
from ml.core.db_engine import EngineManager
from ml.stores._buffered_store import BufferedStoreMixin
from ml.stores._engine_mixin import EngineInitMixin
from ml.stores._health_mixin import HealthMixin
from ml.stores._read_helpers import ReadQueryMixin
from ml.stores._registry_mixin import DataRegistryMixin
from ml.stores._upsert_mixin import SQLUpsertMixin
from ml.stores.base import BaseStore
from ml.stores.base import ModelPrediction


if TYPE_CHECKING:
    import pandas as pd

    from ml.registry.persistence import PersistenceConfig
    from ml.registry.protocols import RegistryProtocol
    from nautilus_trader.common.clock import Clock


logger = logging.getLogger(__name__)


# Backwards-compat: expose a module-level create_engine symbol for tests to monkeypatch.
def create_engine(connection_string: str, **kwargs: object) -> Engine:
    # mypy: allow forwarding arbitrary kwargs to EngineManager
    return EngineManager.get_engine(connection_string, **kwargs)  # type: ignore[arg-type]


# Prometheus metrics for prediction events (centralized)
data_events_total: Counter | None = None
if HAS_PROMETHEUS:
    try:
        from ml.common.metrics import data_events_total as _central_data_events_total

        data_events_total = _central_data_events_total
    except Exception:
        data_events_total = None


class ModelStore(
    HealthMixin,
    BufferedStoreMixin,
    SQLUpsertMixin,
    ReadQueryMixin,
    BaseStore,
    BusPublisherMixin,
    DataRegistryMixin,
    EngineInitMixin,
):
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
        persistence_manager: object | None = None,
        flush_interval_seconds: float | None = None,
        enable_publishing: bool = False,
        publisher: MessagePublisherProtocol | None = None,
        publish_mode: Literal["batch", "row", "both"] = "batch",
        **_: object,
    ) -> None:
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
        persistence_manager : Any | None
            Optional persistence manager (for testing)
        flush_interval_seconds : float | None
            Alternative flush interval in seconds (overrides flush_interval_ms)
        enable_publishing : bool, optional
            When True, publish store events to the optional message bus.
        publisher : MessagePublisherProtocol | None, optional
            Publisher implementation used when `enable_publishing` is True.
        publish_mode : {"batch", "row", "both"}, optional
            Controls whether to publish batch summaries, per-row events, or both. Defaults to "batch".

        """
        # Handle legacy connection string parameter
        if connection_string and not persistence_config:
            # Only create PersistenceConfig for PostgreSQL connections
            if "postgresql://" in connection_string or "postgres://" in connection_string:
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
        self.flush_interval_ms = int(flush_interval_ms)
        if flush_interval_seconds is not None:
            self.flush_interval_ms = int(flush_interval_seconds * 1000)
        self.clock = clock
        # Optional message publishing
        self._init_bus_publishing(
            enable_publishing=enable_publishing,
            publisher=publisher,
            publish_mode=publish_mode,
        )

        # Allow tests to inject a mock persistence manager directly
        if persistence_manager is not None:
            try:
                self.persistence = persistence_manager  # type: ignore[assignment]
            except Exception as exc:
                logger.debug("Ignoring persistence_manager injection error: %s", exc)

        # Write buffer for batching
        self._write_buffer: list[ModelPrediction] = []
        self._last_flush_ns = 0

        # Back-compat: expose `_buffer` alias used by older tests
        # Do not store a separate list; keep a reference to the same object.
        self._buffer: list[ModelPrediction] = self._write_buffer

        # DataRegistry for event emission (lazy initialization)
        self._data_registry: RegistryProtocol | None = None

        # Create engine, metadata, and setup tables (shared init)
        self._init_engine_and_tables()

    def _get_data_registry(self) -> RegistryProtocol | None:
        # Delegate to shared mixin
        return DataRegistryMixin._get_data_registry(self)

    def set_data_registry(self, registry: RegistryProtocol) -> None:
        """
        Set the DataRegistry instance used for event emission.

        Parameters
        ----------
        registry : RegistryProtocol
            The shared registry instance to use.

        """
        self._data_registry = registry

    def _setup_tables(self) -> None:
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
            Column("created_at", BIGINT),  # Dev table; DB default used in prod
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
        from ml.common.timestamps import sanitize_timestamp_ns

        ts_init = (
            self.clock.timestamp_ns()
            if self.clock
            else sanitize_timestamp_ns(
                time.time_ns(),
                logger=logger,
                context="ModelStore.write_prediction:ts_init",
            )
        )

        # Normalize timestamp defensively
        ts_event_norm = sanitize_timestamp_ns(
            int(ts_event),
            logger=logger,
            context="ModelStore.write_prediction",
        )

        data = ModelPrediction(
            model_id=model_id,
            instrument_id=instrument_id,
            prediction=prediction,
            confidence=confidence,
            features_used=features,
            inference_time_ms=inference_time_ms,
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
    def write_batch(self, data: list[ModelPrediction], emit_events: bool = True) -> None:
        """
        Write batch of model predictions.

        Parameters
        ----------
        data : list[ModelPrediction]
            List of predictions to write
        emit_events : bool
            Whether to emit events (default True, False when called from flush to avoid duplication)

        """
        if not data:
            return

        # Prepare values mapping
        values: list[dict[str, Any]] = []
        for item in data:
            values.append(
                {
                    "model_id": item.model_id,
                    "instrument_id": item.instrument_id,
                    "ts_event": item.ts_event,
                    "ts_init": item.ts_init,
                    "prediction": item.prediction,
                    "confidence": item.confidence,
                    "features_used": item.features_used if item.features_used else None,
                    "inference_time_ms": item.inference_time_ms,
                    "is_live": getattr(item, "is_live", False),
                },
            )

        # Allow tests to patch and short-circuit DB writes
        self._execute_write(values)

        # Emit events after successful write if this is a direct call (not from flush)
        if emit_events:
            self._emit_prediction_events(data)

    def _execute_write(
        self,
        values: list[dict[str, Any]],
    ) -> None:  # pragma: no cover
        """
        Upsert predictions and publish via shared helper (patchable in tests).
        """
        if not values:
            return
        # Optional audit logging (sampled)
        try:
            import os
            import random

            sample = int(os.getenv("ML_AUDIT", "0"))
            if sample > 0 and random.randint(1, sample) == 1:
                logger.info(
                    "AUDIT ModelStore._execute_write: n=%d keys=%s",
                    len(values),
                    list(values[0].keys()) if values else [],
                )
        except Exception as e:
            logger.debug("Audit logging skipped due to error: %s", e)

        self._execute_upsert_and_publish(
            values=values,
            ts_event_field="ts_event",
            ts_init_field="ts_init",
            context="ModelStore._execute_write",
            key_fields=("model_id", "instrument_id", "ts_event"),
            table=self.model_predictions_table,
            conflict_cols=["model_id", "instrument_id", "ts_event"],
            update_cols=[
                "prediction",
                "confidence",
                "features_used",
                "inference_time_ms",
            ],
            dataset_id="predictions",
            stage=Stage.PREDICTION_EMITTED,
            instrument_key="instrument_id",
            ts_field="ts_event",
            run_id_batch="model_store_write",
            run_id_row="model_store_row",
            source="inference",
            logger=logger,
        )

    # Backwards-compatible alias used in some tests
    def write_predictions(self, data: list[ModelPrediction]) -> None:
        self.write_batch(data)

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
        import pandas as pd
        from sqlalchemy import text as _text

        table_name = self._qualified_table("ml_model_predictions")
        sql = _text(
            f"""
            SELECT ts_event, prediction, confidence, features_used, inference_time_ms
            FROM {table_name}
            WHERE model_id = :model_id
              AND instrument_id = :instrument_id
              AND ts_event >= :start_ns
              AND ts_event < :end_ns
            ORDER BY ts_event
            """,
        )

        from typing import cast

        with self.engine.connect() as conn:
            params: dict[str, int | str] = {
                "model_id": model_id,
                "instrument_id": instrument_id,
                "start_ns": int(start_ns),
                "end_ns": int(end_ns),
            }
        return pd.read_sql_query(sql, conn, params=cast(Mapping[str, int | str], params))

    # -------------------------------------------------------------------------------------
    # Compatibility reads and aliases
    # -------------------------------------------------------------------------------------

    def read_latest_predictions(
        self,
        model_id: str,
        instrument_id: str | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Read the latest predictions for a model with optional instrument filter.

        Parameters
        ----------
        model_id : str
            Model identifier to filter by.
        instrument_id : str | None
            Optional instrument filter.
        limit : int
            Maximum number of rows to return.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns: model_id, instrument_id, prediction, confidence,
            inference_time_ms, is_live, ts_event, ts_init.

        """
        import pandas as pd
        from sqlalchemy import text as _text

        table_name = self._qualified_table("ml_model_predictions")

        where_parts: list[str] = ["model_id = :model_id"]
        params: dict[str, Any] = {"model_id": model_id, "limit": int(limit)}
        if instrument_id is not None:
            where_parts.append("instrument_id = :instrument_id")
            params["instrument_id"] = instrument_id

        sql = _text(
            f"""
            SELECT model_id,
                   instrument_id,
                   prediction,
                   confidence,
                   inference_time_ms,
                   is_live,
                   ts_event,
                   ts_init
            FROM {table_name}
            WHERE {' AND '.join(where_parts)}
            ORDER BY ts_event DESC
            LIMIT :limit
            """,
        )

        # Prefer a mock-friendly session if available; else use engine
        sess: Any | None = None
        try:
            if hasattr(self, "persistence") and self.persistence is not None:
                # Support both real manager and test doubles providing `get_session` or `session`
                if hasattr(self.persistence, "get_session"):
                    sess = self.persistence.get_session()
                elif hasattr(self.persistence, "session"):
                    sess = getattr(self.persistence, "session")
        except Exception:
            sess = None

        if sess is not None:
            # Use simple execute/fetch for MagicMock compatibility
            try:
                from sqlalchemy import text as _text2

                rows = sess.execute(_text2(str(sql)), params).fetchall()
            except Exception:
                rows = []
            data = [
                {
                    "model_id": r[0],
                    "instrument_id": r[1],
                    "prediction": r[2],
                    "confidence": r[3],
                    "inference_time_ms": r[5],
                    "is_live": r[6],
                    "ts_event": r[7],
                    "ts_init": r[8],
                }
                for r in rows
            ]
            df = pd.DataFrame(
                data,
                columns=[
                    "model_id",
                    "instrument_id",
                    "prediction",
                    "confidence",
                    "inference_time_ms",
                    "is_live",
                    "ts_event",
                    "ts_init",
                ],
            )
            if not len(df.index):
                with self.engine.connect() as conn:
                    return pd.read_sql_query(sql, conn, params=cast(Any, params))
            return df
        else:
            # External API typing variance; use explicit Any per standards
            with self.engine.connect() as conn:
                return pd.read_sql_query(sql, conn, params=cast(Any, params))

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
        import pandas as pd
        from sqlalchemy import text as _text

        if instrument_id is None:
            sql = _text(
                f"""
                SELECT model_id, instrument_id, ts_event, prediction, confidence, inference_time_ms
                FROM {self._qualified_table('ml_model_predictions')}
                WHERE ts_event >= :start_ns AND ts_event < :end_ns
                ORDER BY ts_event
                """,
            )
            params: dict[str, int | str] = {"start_ns": int(start_ns), "end_ns": int(end_ns)}
        else:
            sql = _text(
                f"""
                SELECT model_id, instrument_id, ts_event, prediction, confidence, inference_time_ms
                FROM {self._qualified_table('ml_model_predictions')}
                WHERE ts_event >= :start_ns AND ts_event < :end_ns
                  AND instrument_id = :instrument_id
                ORDER BY ts_event
                """,
            )
            params = {
                "start_ns": int(start_ns),
                "end_ns": int(end_ns),
                "instrument_id": instrument_id,
            }

        with self.engine.connect() as conn:
            return pd.read_sql_query(sql, conn, params=cast(Any, params))

    # Backwards-compatible public API used in some tests
    def get_predictions(
        self,
        model_id: str,
        start_ns: int,
        end_ns: int,
        instrument_id: str | None = None,
    ) -> pd.DataFrame:
        """
        Return predictions for a model within a time range.

        This is a compatibility shim delegating to read_predictions.

        """
        import pandas as pd

        # Accept seconds or nanoseconds; normalize to ns
        from ml.common.timestamps import sanitize_timestamp_ns

        start_ns = sanitize_timestamp_ns(
            int(start_ns),
            logger=logger,
            context="ModelStore.get_predictions:start",
        )
        end_ns = sanitize_timestamp_ns(
            int(end_ns),
            logger=logger,
            context="ModelStore.get_predictions:end",
        )

        if instrument_id is None:
            # Return across instruments by unioning results
            sql = text(
                f"""
                SELECT model_id, instrument_id, ts_event, prediction, confidence, inference_time_ms
                FROM {self._qualified_table('ml_model_predictions')}
                WHERE model_id = :model_id
                  AND ts_event >= :start_ns AND ts_event < :end_ns
                ORDER BY instrument_id, ts_event
                """,
            )
            with self.engine.connect() as conn:
                params2: dict[str, int | str] = {
                    "model_id": model_id,
                    "start_ns": int(start_ns),
                    "end_ns": int(end_ns),
                }
                return pd.read_sql_query(sql, conn, params=params2)
        else:
            return self.read_predictions(
                model_id=model_id,
                instrument_id=instrument_id,
                start_ns=start_ns,
                end_ns=end_ns,
            )

    @override
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
        import pandas as pd
        from sqlalchemy import text as _text

        sql = _text(
            f"""
            SELECT model_id, ts_event, prediction, confidence, inference_time_ms
            FROM {self._qualified_table('ml_model_predictions')}
            WHERE instrument_id = :instrument_id
            ORDER BY ts_event DESC
            LIMIT :limit
            """,
        )

        with self.engine.connect() as conn:
            params2: dict[str, int | str] = {"instrument_id": instrument_id, "limit": int(limit)}
            return pd.read_sql_query(sql, conn, params=params2)

    @override
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
        # Build WHERE clause with parameters
        conditions: list[str] = []
        params: dict[str, Any] = {}
        if start_ns is not None:
            conditions.append("ts_event >= :start_ns")
            params["start_ns"] = int(start_ns)
        if end_ns is not None:
            conditions.append("ts_event < :end_ns")
            params["end_ns"] = int(end_ns)

        base_sql = (
            "SELECT COUNT(*) as total_predictions, "
            "COUNT(DISTINCT model_id) as unique_models, "
            "COUNT(DISTINCT instrument_id) as unique_instruments, "
            "AVG(inference_time_ms) as avg_inference_ms, "
            "MAX(inference_time_ms) as max_inference_ms, "
            "MIN(ts_event) as min_ts, "
            "MAX(ts_event) as max_ts "
            "FROM public.ml_model_predictions "
        )
        if conditions:
            base_sql += "WHERE " + " AND ".join(conditions)

        # Prefer a mock-friendly session when available; else engine
        sess: Any | None = None
        try:
            if hasattr(self, "persistence") and self.persistence is not None:
                # Prefer `.session` when present (MagicMock friendly)
                sess = getattr(self.persistence, "session", None)
                if sess is None and hasattr(self.persistence, "get_session"):
                    sess = self.persistence.get_session()
        except Exception:
            sess = None

        if sess is not None:
            from sqlalchemy import text as _text2

            query = _text2(base_sql)
            try:
                result = sess.execute(query, params).fetchone()
            except Exception:
                result = None
        else:
            with self.engine.connect() as conn:
                query = text(base_sql)
                result = conn.execute(query, params).fetchone()

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
        Delegate to shared buffered flush behavior.
        """
        # Use mixin implementation to avoid duplication across stores
        from ml.stores._buffered_store import BufferedStoreMixin as _BSM

        _BSM.flush(self)

    # Wrapper used by BufferedStoreMixin.flush
    def _emit_events(self, predictions: list[ModelPrediction]) -> None:
        self._emit_prediction_events(predictions)

    def _emit_prediction_events(self, predictions: list[ModelPrediction]) -> None:
        """
        Emit PREDICTION_EMITTED events for the flushed predictions.

        Parameters
        ----------
        predictions : list[ModelPrediction]
            List of predictions that were successfully written

        """
        try:
            registry = self._get_data_registry()
            if registry is None:
                return

            # Group predictions by model_id and instrument_id for efficient event emission
            from collections import defaultdict

            grouped: dict[tuple[str, str], list[ModelPrediction]] = defaultdict(list)

            for pred in predictions:
                key = (pred.model_id, pred.instrument_id)
                grouped[key].append(pred)

            # Emit events for each group
            for (model_id, instrument_id), group_preds in grouped.items():
                if not group_preds:
                    continue

                # Generate unique run ID for this batch
                run_id = f"prediction_{model_id}_{uuid.uuid4().hex[:8]}_{int(time.time())}"

                # Get timestamp range from the group
                ts_values = [p.ts_event for p in group_preds]
                ts_min = min(ts_values)
                ts_max = max(ts_values)

                # Use canonical dataset id; model_id is conveyed via metrics/metadata
                dataset_id = "predictions"

                # Determine source based on is_live flag (if available)
                source = "realtime"
                if hasattr(group_preds[0], "is_live"):
                    source = "realtime" if group_preds[0].is_live else "historical"

                # Emit the event
                registry.emit_event(
                    dataset_id=dataset_id,
                    instrument_id=instrument_id,
                    stage=Stage.PREDICTION_EMITTED.value,
                    source=source,
                    run_id=run_id,
                    ts_min=ts_min,
                    ts_max=ts_max,
                    count=len(group_preds),
                    status=EventStatus.SUCCESS.value,
                    metadata={"model_id": model_id},
                )

                # Update watermark for tracking progress
                registry.update_watermark(
                    dataset_id=dataset_id,
                    instrument_id=instrument_id,
                    source=source,
                    last_success_ns=ts_max,
                    count=len(group_preds),
                    completeness_pct=100.0,  # Predictions are complete once written
                )

                # Update Prometheus metrics if available
                if data_events_total:
                    data_events_total.labels(
                        dataset_type="predictions",
                        component=model_id,
                        stage=Stage.PREDICTION_EMITTED.value,
                        source=source,
                        status=EventStatus.SUCCESS.value,
                    ).inc()

                logger.debug(
                    "Emitted PREDICTION_EMITTED event: dataset=%s, instrument=%s, "
                    "model=%s, count=%d, ts_range=[%d, %d], source=%s",
                    dataset_id,
                    instrument_id,
                    model_id,
                    len(group_preds),
                    ts_min,
                    ts_max,
                    source,
                )

        except Exception as e:
            # Non-blocking: log but don't fail the prediction storage
            logger.warning(f"Failed to emit prediction event: {e}")

    def _should_flush_by_time(self) -> bool:
        """
        Check if flush is needed based on time.
        """
        if not self.clock or not self._last_flush_ns:
            return False

        elapsed_ms = (self.clock.timestamp_ns() - self._last_flush_ns) / 1e6
        return bool(elapsed_ms >= float(self.flush_interval_ms))

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
        *,
        hours_back: int | None = None,
    ) -> dict[str, Any]:
        """
        Get performance metrics for a model.

        Parameters
        ----------
        model_id : str
            Model identifier.
        start_ns : int | None
            Optional start timestamp (nanoseconds).
        end_ns : int | None
            Optional end timestamp (nanoseconds).
        hours_back : int | None
            Optional lookback window in hours. When provided, overrides start/end.

        Returns
        -------
        dict[str, Any]
            Performance metrics

        """
        # Convert hours_back to concrete window when provided
        if hours_back is not None:
            import time as _time

            end_ns = int(_time.time() * 1e9)
            start_ns = int(end_ns - hours_back * 3600 * 1e9)

        conditions: list[str] = ["model_id = :model_id"]
        params: dict[str, Any] = {"model_id": model_id}
        if start_ns is not None:
            conditions.append("ts_event >= :start_ns")
            params["start_ns"] = int(start_ns)
        if end_ns is not None:
            conditions.append("ts_event < :end_ns")
            params["end_ns"] = int(end_ns)

        sql = (
            "SELECT COUNT(*) as prediction_count, "
            "AVG(confidence) as avg_confidence, "
            "STDDEV(confidence) as std_confidence, "
            "AVG(inference_time_ms) as avg_latency_ms, "
            "PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY inference_time_ms) as p50_latency_ms, "
            "PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY inference_time_ms) as p95_latency_ms, "
            "PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY inference_time_ms) as p99_latency_ms "
            "FROM public.ml_model_predictions "
            "WHERE " + " AND ".join(conditions)
        )

        with self.engine.connect() as conn:
            query = text(sql)
            result = conn.execute(query, params).fetchone()

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

    def store_prediction(self, *args: Any, **kwargs: Any) -> None:
        """
        Backward-compatible alias that accepts minimal explicit args.

        Allows calling with model_id, instrument_id, ts_event, prediction, confidence
        and fills features={} and inference_time_ms=0.0 when not provided.

        """
        if args:
            self.write_prediction(*args, **kwargs)
            return
        model_id = kwargs.get("model_id")
        instrument_id = kwargs.get("instrument_id")
        ts_event = kwargs.get("ts_event")
        prediction = kwargs.get("prediction")
        confidence = kwargs.get("confidence")
        features = kwargs.get("features", {})
        inference_time_ms = kwargs.get("inference_time_ms", 0.0)
        if None in {model_id, instrument_id, ts_event, prediction, confidence}:
            self.write_prediction(*args, **kwargs)
            return
        self.write_prediction(
            model_id=str(model_id),
            instrument_id=str(instrument_id),
            prediction=float(cast(float, prediction)),
            confidence=float(cast(float, confidence)),
            features=dict(cast(dict[str, float], features)),
            inference_time_ms=float(cast(float, inference_time_ms)),
            ts_event=int(cast(int, ts_event)),
        )
