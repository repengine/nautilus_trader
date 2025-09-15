"""
Model prediction store for ML pipeline integration.

This module provides storage for model predictions with support for batch writes, time-
partitioned queries, and performance tracking.

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

from ml._imports import HAS_PROMETHEUS
from ml._imports import Counter
from ml.common.message_bus import BusPublisherMixin
from ml.common.message_bus import MessagePublisherProtocol
from ml.core.db_engine import EngineManager
from ml.stores.base import BaseStore
from ml.stores.base import ModelPrediction
from ml.stores.mixins import BufferedStoreMixin
from ml.stores.mixins import DataRegistryMixin
from ml.stores.mixins import EngineInitMixin
from ml.stores.mixins import HealthMixin
from ml.stores.mixins import ReadQueryMixin
from ml.stores.mixins import SQLUpsertMixin
from ml.stores.mixins import StoreInitMixin
from ml.stores.services.model_services import ModelEventService
from ml.stores.services.model_services import ModelQueryService
from ml.stores.services.model_services import ModelStatsService
from ml.stores.services.model_services import ModelWriteService


if TYPE_CHECKING:
    import pandas as pd
    from nautilus_trader.common.clock import Clock

    from ml.registry.persistence import PersistenceConfig
    from ml.registry.protocols import RegistryProtocol


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
    StoreInitMixin,
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
        self._write_buffer: list[ModelPrediction] = []
        # Back-compat: expose `_buffer` alias used by older tests
        # Do not store a separate list; keep a reference to the same object.
        self._buffer: list[ModelPrediction] = self._write_buffer
        # batch_size already set by init mixin; class attribute annotated for mypy

        # DataRegistry for event emission (lazy initialization)
        self._data_registry: RegistryProtocol | None = None

        # Engine + tables already initialized by _init_store_common
        # Extracted services (internal composition; public API unchanged)
        self._write_service = ModelWriteService(self, logger)
        self._query_service = ModelQueryService(self)
        self._stats_service = ModelStatsService(self)
        self._event_service = ModelEventService(self, logger)

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

        Preserves the historical patch point by delegating through
        `self._execute_write(values)` so tests can monkeypatch it.

        Parameters
        ----------
        data : list[ModelPrediction]
            List of predictions to write
        emit_events : bool
            Whether to emit events (default True, False when called from flush to avoid duplication)

        """
        if not data:
            return

        # Track stage boundary for observability (cold path only)
        ts_stage_start = time.time_ns()

        # Delegate to write service to avoid duplication and preserve patch points
        self._write_service.write_batch(data)

        # Emit events after successful write if this is a direct call (not from flush)
        if emit_events:
            self._emit_prediction_events(data)

        # Record observability data (off hot path - background processing only)
        ts_stage_end = time.time_ns()
        # Use the first item's instrument_id as representative
        instrument_id = data[0].instrument_id if data else "unknown"
        self._record_observability_stage_boundary(
            stage="model_prediction_storage",
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
        """Record observability data via centralized helper (cold path only)."""
        from ml.common.observability_utils import record_stage_boundary as _rec

        obs_service = getattr(self, "_observability_service", None)
        _rec(
            obs_service,
            component="model_store",
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
        from typing import cast as _cast

        import pandas as pd

        return _cast(
            pd.DataFrame,
            self._query_service.read_predictions(
                model_id=model_id,
                instrument_id=instrument_id,
                start_ns=start_ns,
                end_ns=end_ns,
            ),
        )

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
        from typing import cast as _cast

        import pandas as pd

        return _cast(
            pd.DataFrame,
            self._query_service.read_latest_predictions(
                model_id=model_id,
                instrument_id=instrument_id,
                limit=limit,
            ),
        )

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

        from typing import cast as _cast

        import pandas as pd

        return _cast(
            pd.DataFrame,
            self._query_service.get_predictions(
                model_id=model_id,
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
        from typing import cast as _cast

        import pandas as pd

        return _cast(
            pd.DataFrame,
            self._query_service.get_latest_by_instrument(
                instrument_id=instrument_id,
                limit=limit,
            ),
        )

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
        return self._stats_service.get_statistics(start_ns=start_ns, end_ns=end_ns)

    def flush(self) -> None:
        """
        Delegate to shared buffered flush behavior.
        """
        # Use mixin implementation to avoid duplication across stores
        from ml.stores.mixins import BufferedStoreMixin as _BSM

        _BSM.flush(self)

    # Wrapper used by BufferedStoreMixin.flush
    def _emit_events(self, predictions: list[ModelPrediction]) -> None:
        self._emit_prediction_events(predictions)

    def _emit_prediction_events(self, predictions: list[ModelPrediction]) -> None:
        """
        Delegate to event service (non-blocking).
        """
        try:
            self._event_service.emit_prediction_events(predictions)
        except Exception:
            logger.debug("Prediction event emission failed", exc_info=True)

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
        return self._stats_service.get_model_performance(
            model_id=model_id,
            start_ns=start_ns,
            end_ns=end_ns,
            hours_back=hours_back,
        )

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

    # Attributes initialized via StoreInitMixin
    batch_size: int
    flush_interval_ms: int
    clock: Clock | None
    connection_string: str | None
    persistence: object | None
    _last_flush_ns: int
