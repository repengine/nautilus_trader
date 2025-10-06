"""
Feature store for ML pipeline integration with Nautilus Trader.

This module provides a unified interface for computing, storing, and retrieving
ML features from the same PostgreSQL instance used by Nautilus Trader.

Key principles:
- Single PostgreSQL container (Nautilus's existing one)
- FeatureEngineer provides all computation logic (training/inference parity)
- Features stored alongside Nautilus market data for unified access
- Efficient batch computation for historical data
- Real-time computation for live trading

"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import numpy as np
import numpy.typing as npt
from sqlalchemy import BIGINT
from sqlalchemy import BOOLEAN
from sqlalchemy import JSON
from sqlalchemy import Column
from sqlalchemy import Index
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

from ml.common.db_utils import get_or_create_engine
from ml.common.error_handlers import with_fallback
from ml.common.event_emitter import emit_dataset_event_and_watermark
from ml.common.message_bus import BusPublisherMixin
from ml.common.message_bus import MessagePublisherProtocol
from ml.common.message_topics import build_topic_for_stage
from ml.config.base import MLFeatureConfig
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.features.engineering import IndicatorManager
from ml.features.pipeline import PipelineRunner
from ml.features.pipeline import PipelineSpec
from ml.stores.mixins import DataRegistryMixin
from ml.stores.mixins import HealthMixin


if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd
    from polars import DataFrame as PlDataFrame

    from ml.registry.protocols import RegistryProtocol
    from nautilus_trader.model.data import Bar


logger = logging.getLogger(__name__)


# Backwards-compat: expose a module-level PersistenceManager symbol for tests to monkeypatch.
try:  # pragma: no cover - used only in tests which patch this symbol
    from ml.registry import persistence as _persistence

    _RealPM = _persistence.PersistenceManager
    PersistenceManager: type[Any] = _RealPM
except Exception:  # pragma: no cover

    class _StubPM:
        """
        Test stub for patching.
        """

    PersistenceManager = _StubPM


class _CounterLike(Protocol):
    def labels(self, **kwargs: object) -> _CounterLike: ...

    def inc(self, *args: object, **kwargs: object) -> None: ...


_data_events_total: _CounterLike | None = None


def _get_data_events_total() -> _CounterLike | None:
    global _data_events_total
    if _data_events_total is None:
        try:
            from ml.common.metrics_bootstrap import get_counter

            _data_events_total = get_counter(
                "nautilus_ml_data_events_total",
                "Total data events processed by stage",
                labelnames=["dataset_type", "component", "stage", "source", "status"],
            )
        except Exception:
            _data_events_total = None
    return _data_events_total


class FeatureStore(HealthMixin, BusPublisherMixin, DataRegistryMixin):
    """
    Unified feature computation and storage for ML pipeline.

    This class ensures training/inference parity by using the same FeatureEngineer for
    both batch (historical) and online (live) computation.

    Features are stored in the same PostgreSQL instance as Nautilus data, enabling
    efficient joins for training and avoiding data duplication.

    """

    def __init__(
        self,
        connection_string: str,
        feature_config: FeatureConfig | MLFeatureConfig | None = None,
        pipeline_spec: PipelineSpec | None = None,
        persistence_manager: object | None = None,
        enable_publishing: bool = False,
        publisher: MessagePublisherProtocol | None = None,
        publish_mode: Literal["batch", "row", "both"] = "batch",
        # Accept extra kwargs for compatibility
        **_: Any,
    ) -> None:
        """
        Initialize the feature store.

        Parameters
        ----------
        connection_string : str
            PostgreSQL connection string (same as Nautilus uses).
            Example: "postgresql://postgres:postgres@localhost:5432/nautilus"
        feature_config : FeatureConfig, optional
            Configuration for feature engineering.
        pipeline_spec : PipelineSpec, optional
            Pipeline specification for feature computation.
        persistence_manager : object | None
            Optional persistence/session provider (used by tests for mocking).
        enable_publishing : bool, optional
            When True, publish store events to the optional message bus.
        publisher : MessagePublisherProtocol | None, optional
            Publisher implementation used when `enable_publishing` is True.
        publish_mode : {"batch", "row", "both"}, optional
            Controls whether to publish batch summaries, per-row events, or both. Defaults to "batch".

        """
        self.connection_string = connection_string
        self._data_registry: RegistryProtocol | None = None
        # Optional persistence manager (mock-friendly)
        self.persistence: object | None = persistence_manager
        # Accept both FeatureConfig and MLFeatureConfig; normalize to FeatureConfig
        if isinstance(feature_config, FeatureConfig):
            self.feature_config: FeatureConfig = feature_config
        elif isinstance(feature_config, MLFeatureConfig):
            try:
                import msgspec as _msgspec

                self.feature_config = FeatureConfig(**_msgspec.to_builtins(feature_config))
            except Exception:
                self.feature_config = FeatureConfig(**getattr(feature_config, "__dict__", {}))
        else:
            self.feature_config = FeatureConfig()
        self.pipeline_spec = pipeline_spec

        # Create engine and setup tables (reflect partitioned table created by migrations)
        self.engine: Engine = get_or_create_engine(connection_string)
        self.metadata = MetaData()
        self._setup_tables()

        # Feature engineer for computation (ensures parity)
        self.feature_engineer = FeatureEngineer(self.feature_config)
        # Internal indicator managers (fallback for online computation when actor does not pass one)
        self._indicator_managers: dict[str, IndicatorManager] = {}

        # Pipeline runners for declarative features (offline vs online)
        self.pipeline_runner_offline: PipelineRunner | None
        self.pipeline_runner_online: PipelineRunner | None
        self.pipeline_hash: str
        if self.pipeline_spec:
            from ml.registry.base import DataRequirements

            # Offline (batch/teacher): allow L1_L2 to include microstructure/trade-flow
            self.pipeline_runner_offline = PipelineRunner(
                self.pipeline_spec,
                DataRequirements.L1_L2,
            )
            # Online (student/runtime): limit to L1 until actors are available
            self.pipeline_runner_online = PipelineRunner(
                self.pipeline_spec,
                DataRequirements.L1_ONLY,
            )
            self.pipeline_hash = self.pipeline_runner_offline.compute_signature()
        else:
            self.pipeline_runner_offline = None
            self.pipeline_runner_online = None
        self.pipeline_hash = self._compute_config_hash()

        # Lightweight write buffer for compatibility with older tests
        # (FeatureStore writes synchronously by default; buffer is only used
        #  when write_batch is called in tests.)
        from ml.stores.base import FeatureData  # import locally to avoid cycles in type hints

        self._write_buffer: list[FeatureData] = []
        # Back-compat alias expected by tests
        self._buffer: list[FeatureData] = self._write_buffer
        # Optional message publishing
        self._init_bus_publishing(
            enable_publishing=enable_publishing,
            publisher=publisher,
            publish_mode=publish_mode,
        )

        # Optional circuit breaker injected by actors/services

        from ml.stores.protocols import CircuitBreakerProtocol as _CBP

        self._circuit_breaker: _CBP | None = None

    def _get_data_registry(self) -> RegistryProtocol | None:
        """
        Delegate initialization to shared mixin (unified across stores).
        """
        return DataRegistryMixin._get_data_registry(self)

    # Allow external injection of a shared DataRegistry instance
    def set_data_registry(self, registry: RegistryProtocol) -> None:
        """
        Set the DataRegistry instance used for event emission and watermarks.

        Parameters
        ----------
        registry : RegistryProtocol
            The shared registry instance to use.

        """
        self._data_registry = registry

    def compute_historical_parallel(
        self,
        instrument_ids: list[str],
        start: datetime | None = None,
        end: datetime | None = None,
        *,
        force_recompute: bool = False,
        max_workers: int = 4,
    ) -> dict[str, int]:
        """
        Compute-and-store historical features for multiple instruments in parallel.

        Parameters
        ----------
        instrument_ids : list[str]
            Instruments to compute.
        start : datetime, optional
            Start time (inclusive).
        end : datetime, optional
            End time (exclusive).
        force_recompute : bool, default False
            Recompute even if features exist.
        max_workers : int, default 4
            Maximum concurrent workers (bounded to avoid pool exhaustion).

        Returns
        -------
        dict[str, int]
            Mapping instrument_id -> rows written (0 on failure).

        """
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import as_completed

        results: dict[str, int] = {}

        if not instrument_ids:
            return results

        # Cap workers to a reasonable limit to play nicely with DB pools
        workers = max(1, min(max_workers, 8))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_to_inst = {
                ex.submit(
                    self.compute_and_store_historical,
                    instrument_id=inst,
                    start=start or datetime.now(UTC) - timedelta(days=1),
                    end=end or datetime.now(UTC),
                    force_recompute=force_recompute,
                ): inst
                for inst in instrument_ids
            }
            for fut in as_completed(fut_to_inst):
                inst = fut_to_inst[fut]
                try:
                    results[inst] = int(fut.result())
                except Exception:  # pragma: no cover - environment dependent
                    logger.error(
                        "Parallel feature compute failed for %s",
                        inst,
                        exc_info=True,
                    )
                    results[inst] = 0

        return results

    def _setup_tables(self) -> None:
        """
        Reflect (preferred) or create a compatible ml_feature_values table.

        The canonical schema is created by migrations (partitioned by ts_event):
        - feature_set_id VARCHAR(255)
        - instrument_id VARCHAR(100)
        - ts_event BIGINT
        - ts_init BIGINT
        - values JSONB
        - is_live BOOLEAN
        - source VARCHAR(50)
        - created_at TIMESTAMPTZ
        Primary key (id, ts_event) where id is BIGSERIAL.

        """
        from ml.stores.table_factory import get_schema_name

        schema_name = get_schema_name(self.engine)

        try:
            # Prefer reflecting the migrated table. Avoid opportunistic DDL here to
            # prevent lock contention with concurrent writers in tests/integration;
            # migrations and test fixtures are responsible for indexes/partitions.
            self.feature_values_table = Table(
                "ml_feature_values",
                self.metadata,
                autoload_with=self.engine,
                schema=schema_name,
            )
        except Exception:
            # Fallback: create a non-partitioned compatible table for tests/dev
            from sqlalchemy import Integer

            self.feature_values_table = Table(
                "ml_feature_values",
                self.metadata,
                Column("id", Integer, primary_key=True, autoincrement=True),
                Column("feature_set_id", String(255), nullable=False),
                Column("instrument_id", String(100), nullable=False),
                Column("ts_event", BIGINT, nullable=False),
                Column("ts_init", BIGINT, nullable=False),
                Column("values", JSON, nullable=False),
                Column("is_live", BOOLEAN, default=False),
                Column("source", String(50)),
                Column("created_at", BIGINT),
                Index(
                    "idx_ml_feature_values_lookup",
                    "feature_set_id",
                    "instrument_id",
                    "ts_event",
                ),
                Index(
                    "uq_ml_feature_values_key_dev",
                    "feature_set_id",
                    "instrument_id",
                    "ts_event",
                    unique=True,
                ),
                Index("idx_ml_feature_values_live", "is_live"),
                schema=schema_name,
            )
            self.metadata.create_all(self.engine)

    @staticmethod
    def _normalize_ts_ns(ts_value: int) -> tuple[int, bool]:
        """
        Delegate to centralized timestamp normalization utility.
        """
        from ml.common.timestamps import normalize_timestamp_ns

        return normalize_timestamp_ns(ts_value)

    # Present for test monkeypatching and future extension; no-op here.
    def _store_to_postgres(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        """
        Store computed features (placeholder).

        Tests may monkeypatch this method. In production, storage is handled directly in
        compute_realtime/compute_and_store_historical.

        """
        return None

    def _compute_config_hash(self) -> str:
        """
        Compute hash of feature configuration for versioning.
        """
        # Handle both dict-like and dataclass objects
        if hasattr(self.feature_config, "__dict__"):
            config_dict = self.feature_config.__dict__
        else:
            # For frozen dataclasses, convert to dict
            import msgspec

            config_dict = msgspec.to_builtins(self.feature_config)

        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def compute_and_store_historical(
        self,
        instrument_id: str,
        start: datetime,
        end: datetime,
        force_recompute: bool = False,
    ) -> int:
        """
        Compute and store features for historical data.

        This method:
        1. Queries bars from Nautilus PostgreSQL tables
        2. Computes features using FeatureEngineer (same logic as live)
        3. Stores features in ml_feature_values table

        Parameters
        ----------
        instrument_id : str
            Instrument to compute features for.
        start : datetime
            Start time for historical computation.
        end : datetime
            End time for historical computation.
        force_recompute : bool, default False
            If True, recompute even if features exist.

        Returns
        -------
        int
            Number of feature rows computed and stored.

        """
        # Check if features already exist
        if not force_recompute and self._features_exist(instrument_id, start, end):
            return 0

        # Load bars from Nautilus tables
        bars_df = self._load_bars_from_nautilus(instrument_id, start, end)
        if bars_df.is_empty():
            return 0

        # Compute features (batch) ensuring parity with online
        features_df, _ = self.feature_engineer.calculate_features_batch(bars_df)

        feature_names = self._get_feature_names()
        feature_set_id = self._get_feature_set_id()

        # Prepare rows with JSONB values mapping
        rows: list[dict[str, Any]] = []
        timestamps = bars_df["ts_event"].to_numpy()
        # created_at is managed by DB default (TIMESTAMPTZ)

        # Convert feature rows to dicts
        # features_df is a DataFrame (polars or pandas). Use row-wise access safely.
        if hasattr(features_df, "iter_rows"):
            # Polars
            from typing import cast

            from ml.ml_types import PandasDF
            from ml.ml_types import PolarsDF

            pf = cast(PolarsDF, features_df)
            for i, row_vals in enumerate(pf.iter_rows()):
                ts_event = int(timestamps[i])
                values_map = {name: float(row_vals[idx]) for idx, name in enumerate(feature_names)}
                rows.append(
                    {
                        "feature_set_id": feature_set_id,
                        "instrument_id": instrument_id,
                        "ts_event": ts_event,
                        "ts_init": ts_event,
                        "values": values_map,
                        "is_live": False,
                        "source": "historical",
                        # created_at omitted: DB default
                    },
                )
        else:
            # Pandas path
            from typing import cast

            from ml.ml_types import PandasDF

            pdf = cast(PandasDF, features_df)
            for i in range(len(pdf)):
                ts_event = int(timestamps[i])
                row = pdf.iloc[i]
                values_map = {name: float(row[name]) for name in feature_names}
                rows.append(
                    {
                        "feature_set_id": feature_set_id,
                        "instrument_id": instrument_id,
                        "ts_event": ts_event,
                        "ts_init": ts_event,
                        "values": values_map,
                        "is_live": False,
                        "source": "historical",
                        # created_at omitted: DB default
                    },
                )

        # Bulk upsert into partitioned table
        with self.engine.begin() as conn:

            stmt: Any = insert(self.feature_values_table)
            # Upsert on (feature_set_id, instrument_id, ts_event)
            stmt = stmt.on_conflict_do_update(
                index_elements=[
                    "feature_set_id",
                    "instrument_id",
                    "ts_event",
                ],
                set_={
                    "values": stmt.excluded["values"],
                    "ts_init": stmt.excluded.ts_init,
                    "source": stmt.excluded.source,
                    # created_at left as existing/default
                },
            )
            conn.execute(stmt, rows)

        # Emit FEATURE_COMPUTED event for successful historical computation (non-blocking)
        self._emit_historical_event(instrument_id, timestamps, len(rows))

        return len(rows)

    def compute_realtime(
        self,
        bar: Bar,
        store: bool = True,
        indicator_manager: IndicatorManager | None = None,
    ) -> npt.NDArray[np.float32]:
        """
        Compute features for real-time inference.

        Uses the SAME FeatureEngineer as historical computation to ensure
        perfect parity between training and inference.

        Parameters
        ----------
        bar : Bar
            Current bar from Nautilus.
        store : bool, default True
            Whether to store computed features for future training.
        indicator_manager : IndicatorManager | None, default None
            Optional indicator manager for stateful indicator computation.

        Returns
        -------
        npt.NDArray[np.float32]
            Computed feature vector.

        """
        # Prepare indicator manager (prefer provided from actor for shared state)
        instrument_key = str(
            getattr(
                bar,
                "instrument_id",
                getattr(bar, "bar_type", getattr(bar, "instrument_id", None)),
            ),
        )
        instrument_key = (
            str(bar.bar_type.instrument_id)
            if hasattr(bar, "bar_type") and hasattr(bar.bar_type, "instrument_id")
            else str(getattr(bar, "instrument_id", "unknown"))
        )

        if indicator_manager is None:
            indicator_manager = self._indicator_managers.get(instrument_key)
            if indicator_manager is None:
                indicator_manager = IndicatorManager(self.feature_engineer.config)
                self._indicator_managers[instrument_key] = indicator_manager

        # Update indicators from bar and compute online features
        indicator_manager.update_from_bar(bar)
        if not indicator_manager.all_initialized():
            # Not enough history yet - return empty array to signal no prediction
            return np.zeros(0, dtype=np.float32)

        current_bar = {
            "close": float(bar.close),
            "volume": float(bar.volume),
            "high": float(bar.high),
            "low": float(bar.low),
        }

        features = self.feature_engineer.calculate_features_online(
            current_bar=current_bar,
            indicator_manager=indicator_manager,
            scaler=None,
        )

        # Optionally store for future training (with circuit-breaker protection)
        if store and features.size > 0:
            feature_names = self._get_feature_names_online()
            values_map = {
                name: float(features[idx])
                for idx, name in enumerate(feature_names)
                if idx < features.size
            }
            from ml.common.timestamps import sanitize_timestamp_ns

            tse_norm = sanitize_timestamp_ns(
                int(bar.ts_event),
                logger=logger,
                context="FeatureStore.realtime",
            )
            tsi_norm = sanitize_timestamp_ns(
                int(bar.ts_init),
                logger=logger,
                context="FeatureStore.realtime",
            )

            row = {
                "feature_set_id": self._get_feature_set_id(),
                "instrument_id": str(
                    (
                        bar.bar_type.instrument_id
                        if hasattr(bar, "bar_type")
                        else getattr(bar, "instrument_id", "unknown")
                    ),
                ),
                "ts_event": tse_norm,
                "ts_init": tsi_norm,
                "values": values_map,
                "is_live": True,
                "source": "live",
                # created_at omitted: DB default
            }

            cb = self._circuit_breaker
            if cb is not None and not cb.can_execute():
                return features

            try:
                with self.engine.begin() as conn:
                    stmt: Any = insert(self.feature_values_table)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["feature_set_id", "instrument_id", "ts_event"],
                        set_={
                            "values": stmt.excluded["values"],
                            "ts_init": stmt.excluded.ts_init,
                            "is_live": stmt.excluded.is_live,
                            "source": stmt.excluded.source,
                            # created_at kept as existing/default
                        },
                    )
                    conn.execute(stmt, row)
            except Exception:
                try:
                    if cb is not None:
                        cb.record_failure()
                except Exception:
                    pass
                raise
            else:
                try:
                    if cb is not None:
                        cb.record_success()
                except Exception:
                    pass

                # Emit FEATURE_COMPUTED event for successful realtime computation with storage
                try:
                    registry = self._get_data_registry()
                    if registry:
                        # Generate unique run ID for this computation
                        run_id = f"feature_realtime_{uuid.uuid4().hex[:8]}_{int(time.time())}"

                        # Context identifiers
                        feature_set_id = self._get_feature_set_id()
                        instrument_id_str = str(
                            (
                                bar.bar_type.instrument_id
                                if hasattr(bar, "bar_type")
                                else getattr(bar, "instrument_id", "unknown")
                            ),
                        )
                        dataset_id = "features"

                        # Emit via shared helper (event + watermark + metrics)
                        emit_dataset_event_and_watermark(
                            registry,
                            dataset_id=dataset_id,
                            instrument_id=instrument_id_str,
                            stage=Stage.FEATURE_COMPUTED,
                            source=Source.LIVE,
                            run_id=run_id,
                            ts_min=int(bar.ts_event),
                            ts_max=int(bar.ts_event),
                            count=1,
                            status=EventStatus.SUCCESS,
                            dataset_type="features",
                            component=feature_set_id,
                        )

                        logger.debug(
                            "Emitted FEATURE_COMPUTED event for realtime computation: "
                            "dataset=%s, instrument=%s, ts_event=%d",
                            dataset_id,
                            instrument_id_str,
                            int(bar.ts_event),
                        )
                except Exception:
                    # Non-blocking: log but don't fail the feature computation
                    logger.warning(
                        "Failed to emit realtime feature event",
                        exc_info=True,
                    )

        return features

    def get_training_data(
        self,
        instrument_id: str,
        start: datetime,
        end: datetime,
        include_bars: bool = True,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], list[str]]:
        """
        Load features for training.

        Parameters
        ----------
        instrument_id : str
            Instrument to load features for.
        start : datetime
            Start time.
        end : datetime
            End time.
        include_bars : bool, default True
            Whether to join with bar data for labels.

        Returns
        -------
        tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], list[str]]
            Features array, timestamps array, and feature names.

        """
        # Currently training data loading returns feature arrays only; bars can be joined
        # by caller if required. Consume flag to avoid unused parameter warnings.
        _ = include_bars
        from ml.common.timestamps import sanitize_timestamp_ns

        start_ns = sanitize_timestamp_ns(
            int(start.timestamp() * 1e9),
            context="feature_store.compute_and_store_historical.start",
        )
        end_ns = sanitize_timestamp_ns(
            int(end.timestamp() * 1e9),
            context="feature_store.compute_and_store_historical.end",
        )

        # Query features for feature_set_id and time range
        feature_set_id = self._get_feature_set_id()
        query = (
            select(
                self.feature_values_table.c.ts_event,
                self.feature_values_table.c["values"],
            )
            .where(
                (self.feature_values_table.c.feature_set_id == feature_set_id)
                & (self.feature_values_table.c.instrument_id == instrument_id)
                & (self.feature_values_table.c.ts_event >= start_ns)
                & (self.feature_values_table.c.ts_event <= end_ns),
            )
            .order_by(self.feature_values_table.c.ts_event)
        )

        with self.engine.connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()

        if not rows:
            return np.array([]), np.array([]), []

        # Extract data
        feature_names = self._get_feature_names()
        timestamps = np.array([row[0] for row in rows], dtype=np.int64)
        # Rows contain JSON values map; reconstruct arrays in feature_names order
        feature_arrays: list[list[float]] = []
        for _, values_json in rows:
            mapping = values_json
            if isinstance(mapping, str):
                try:
                    mapping = json.loads(mapping)
                except Exception:
                    mapping = {}
            feature_arrays.append([float(mapping.get(name, 0.0)) for name in feature_names])
        features = np.array(feature_arrays, dtype=np.float64)

        return features, timestamps, feature_names

    def _load_bars_from_nautilus(
        self,
        instrument_id: str,
        start: datetime,
        end: datetime,
    ) -> PlDataFrame:
        """
        Load bars from Nautilus PostgreSQL tables.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier.
        start : datetime
            Start time.
        end : datetime
            End time.

        Returns
        -------
        pl.DataFrame
            Bars dataframe with Nautilus schema.

        """
        from typing import Any as _Any
        from typing import cast as _cast

        import pandas as pd
        from sqlalchemy import text as _text

        from ml._imports import pl

        pl = _cast(_Any, pl)
        from ml.common.timestamps import sanitize_timestamp_ns

        start_ns = sanitize_timestamp_ns(
            int(start.timestamp() * 1e9),
            context="feature_store._load_bars_from_nautilus.start",
        )
        end_ns = sanitize_timestamp_ns(
            int(end.timestamp() * 1e9),
            context="feature_store._load_bars_from_nautilus.end",
        )
        sql = _text(
            """
            SELECT ts_event, open, high, low, close, volume
            FROM public.bar
            WHERE instrument_id = :instrument_id
              AND ts_event >= :start_ns
              AND ts_event <= :end_ns
            ORDER BY ts_event
            """,
        )
        with self.engine.connect() as conn:
            from collections.abc import Mapping
            from typing import Any as _Any
            from typing import cast as _cast

            _params = _cast(
                Mapping[str, _Any],
                {
                    "instrument_id": instrument_id,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                },
            )
            pdf = pd.read_sql_query(sql, conn, params=_params)
        from typing import cast as _cast

        return _cast("PlDataFrame", pl.from_pandas(pdf))

    def _features_exist(
        self,
        instrument_id: str,
        start: datetime,
        end: datetime,
    ) -> bool:
        """
        Check if features already exist for the given range.
        """
        from ml.common.timestamps import sanitize_timestamp_ns

        start_ns = sanitize_timestamp_ns(
            int(start.timestamp() * 1e9),
            context="feature_store._load_quotes_from_nautilus.start",
        )
        end_ns = sanitize_timestamp_ns(
            int(end.timestamp() * 1e9),
            context="feature_store._load_quotes_from_nautilus.end",
        )

        feature_set_id = self._get_feature_set_id()
        query = (
            select(self.feature_values_table.c.ts_event)
            .where(
                (self.feature_values_table.c.feature_set_id == feature_set_id)
                & (self.feature_values_table.c.instrument_id == instrument_id)
                & (self.feature_values_table.c.ts_event >= start_ns)
                & (self.feature_values_table.c.ts_event <= end_ns),
            )
            .limit(1)
        )

        with self.engine.connect() as conn:
            result = conn.execute(query)
            return result.fetchone() is not None

    def _get_feature_names(self) -> list[str]:
        """
        Get OFFLINE feature names from pipeline or config.
        """
        if self.pipeline_runner_offline:
            return self.pipeline_runner_offline.compute_feature_names()
        else:
            # Get from FeatureEngineer
            return self.feature_engineer.get_feature_names()

    def _get_feature_names_online(self) -> list[str]:
        """
        Get ONLINE (hot-path) feature names from pipeline or config with L1_ONLY gating.
        """
        if self.pipeline_runner_online:
            return self.pipeline_runner_online.compute_feature_names()
        # Derive from current FeatureEngineer configuration if no pipeline_spec provided
        from ml.features.pipeline import PipelineRunner as _PR
        from ml.registry.base import DataRequirements as _DR

        spec = self.feature_engineer.build_pipeline_spec_from_config()
        return _PR(spec, allowable=_DR.L1_ONLY).compute_feature_names()

    def _get_feature_set_id(self) -> str:
        """
        Derive a stable feature_set_id for storage.

        Prefer pipeline signature; otherwise use config hash prefix.
        """
        if self.pipeline_hash:
            return f"fs_{self.pipeline_hash[:12]}"
        return f"fs_{self._compute_config_hash()[:12]}"

    @with_fallback(fallback_value=None, log_level="warning", operation_name="emit feature computation event")
    def _emit_historical_event(
        self,
        instrument_id: str,
        timestamps: npt.NDArray[np.int64],
        row_count: int,
    ) -> None:
        """
        Emit FEATURE_COMPUTED event for historical computation.

        Non-blocking operation - failures are logged but don't affect feature computation.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier.
        timestamps : npt.NDArray[np.int64]
            Array of timestamps for the computed features.
        row_count : int
            Number of rows computed.
        """
        registry = self._get_data_registry()
        if not registry:
            return

        # Generate unique run ID for this computation
        run_id = f"feature_historical_{uuid.uuid4().hex[:8]}_{int(time.time())}"

        # Use canonical dataset id
        feature_set_id = self._get_feature_set_id()
        dataset_id = "features"

        # Get the time range from timestamps
        ts_min = int(timestamps[0]) if len(timestamps) > 0 else 0
        ts_max = int(timestamps[-1]) if len(timestamps) > 0 else 0

        # Emit via shared helper (event + watermark + metrics)
        emit_dataset_event_and_watermark(
            registry,
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            stage=Stage.FEATURE_COMPUTED,
            source=Source.HISTORICAL,
            run_id=run_id,
            ts_min=ts_min,
            ts_max=ts_max,
            count=row_count,
            status=EventStatus.SUCCESS,
            dataset_type="features",
            component=feature_set_id,
        )

        logger.debug(
            "Emitted FEATURE_COMPUTED event for historical computation: "
            "dataset=%s, instrument=%s, count=%d, ts_range=[%d, %d]",
            dataset_id,
            instrument_id,
            row_count,
            ts_min,
            ts_max,
        )

    def get_latest_at_or_before(self, instrument_id: str, ts_event: int) -> dict[str, float] | None:
        """
        Return the latest feature row at or before the given timestamp.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier.
        ts_event : int
            Event timestamp in nanoseconds.

        Returns
        -------
        dict[str, float] | None
            Mapping of feature name to value, or None when not found.
        """
        from typing import Any, cast

        from sqlalchemy import desc

        from ml.common.timestamps import sanitize_timestamp_ns

        ts_norm = sanitize_timestamp_ns(int(ts_event), context="feature_store.get_latest_at_or_before")
        feature_set_id = self._get_feature_set_id()
        query = (
            select(self.feature_values_table.c["values"])
            .where(
                (self.feature_values_table.c.feature_set_id == feature_set_id)
                & (self.feature_values_table.c.instrument_id == instrument_id)
                & (self.feature_values_table.c.ts_event <= ts_norm),
            )
            .order_by(desc(self.feature_values_table.c.ts_event))
            .limit(1)
        )
        with self.engine.connect() as conn:
            row = conn.execute(query).fetchone()
        if not row:
            return None
        mapping: Any = row[0]
        if isinstance(mapping, str):
            import json as _json
            try:
                mapping = _json.loads(mapping)
            except Exception:
                mapping = {}
        try:
            return {str(k): float(v) for k, v in cast(dict[str, Any], mapping or {}).items()}
        except Exception:
            return {}

    def clear_features(
        self,
        instrument_id: str | None = None,
        feature_version: str | None = None,
    ) -> None:
        """
        Clear stored features.

        Parameters
        ----------
        instrument_id : str, optional
            Clear only for specific instrument.
        feature_version : str, optional
            Clear only specific version.

        """
        with self.engine.begin() as conn:
            delete_stmt = self.feature_values_table.delete()

            if instrument_id:
                delete_stmt = delete_stmt.where(
                    self.feature_values_table.c.instrument_id == instrument_id,
                )

            if feature_version:
                delete_stmt = delete_stmt.where(
                    self.feature_values_table.c.feature_version == feature_version,
                )

            conn.execute(delete_stmt)

    def write_features(
        self,
        feature_set_id: str | None = None,
        instrument_id: str | None = None,
        features: Mapping[str, float] | None = None,
        ts_event: int | None = None,
        ts_init: int | None = None,
        data: Any | None = None,
        *,
        publish_bus: bool = True,
    ) -> None:
        """
        Write computed features to storage.

        Supports both the explicit-args signature and a backwards-compatible
        form where callers pass a FeatureData or list[FeatureData]. This helps
        legacy tests which call `write_features([FeatureData])`.

        Parameters
        ----------
        feature_set_id : str | None
            Feature set identifier (explicit mode)
        instrument_id : str | None
            Instrument identifier (explicit mode)
        features : dict[str, float] | None
            Feature name to value mapping (explicit mode)
        ts_event : int | None
            Event timestamp in nanoseconds (explicit mode)
        ts_init : int | None
            Initialization timestamp in nanoseconds (explicit mode)
        data : Any | None
            Backwards-compat: a FeatureData or list[FeatureData]

        Parameters
        ----------
        publish_bus : bool, keyword-only, default True
            When True and publishing is enabled, publish a summary payload to the
            configured message bus. Set to False to suppress publishing when
            orchestrated by a higher-level facade (e.g., DataStore).

        """
        # Backwards compatibility: support write_features([FeatureData]) / (batch)
        batch_data: list[Any] | None = None
        if data is None and feature_set_id is not None and isinstance(feature_set_id, list):
            # Called as write_features([FeatureData])
            batch_data = feature_set_id
            feature_set_id = None
        elif data is not None:
            if isinstance(data, list):
                batch_data = data
            elif hasattr(data, "feature_values") and hasattr(data, "feature_set_id"):
                batch_data = [data]
            else:
                msg = "Unsupported data type for write_features"
                raise TypeError(msg)

        if batch_data is not None:
            batch: list[Any] = batch_data

            # Perform upserts per item
            for item in batch:
                fs_id = getattr(item, "feature_set_id")
                inst = getattr(item, "instrument_id")
                # Use safe accessor to avoid collisions with base class methods
                try:
                    vals: dict[str, float] = item.feature_values
                except Exception:
                    vals = {}
                tse = int(getattr(item, "ts_event"))
                tsi = int(getattr(item, "ts_init", tse))

                row = {
                    "feature_set_id": fs_id,
                    "instrument_id": inst,
                    "ts_event": tse,
                    "ts_init": tsi,
                    "values": vals,
                    "is_live": False,
                    "source": "computed",
                }
                self._execute_write(row)
            # Optional publish per-batch summary
            if (
                self._enable_publishing
                and self.publisher is not None
                and batch
                and self._publish_mode in ("batch", "both")
                and publish_bus
            ):
                try:
                    stage = Stage.FEATURE_COMPUTED
                    inst_any = getattr(batch[0], "instrument_id", "UNKNOWN")
                    topic = build_topic_for_stage(
                        stage,
                        str(inst_any),
                        scheme=self._topic_scheme,
                        prefix=self._topic_prefix,
                    )
                    ts_min = min(int(getattr(b, "ts_event", 0)) for b in batch)
                    ts_max = max(int(getattr(b, "ts_event", 0)) for b in batch)
                    payload: dict[str, Any] = {
                        "dataset_id": "features",
                        "instrument_id": str(inst_any),
                        "stage": stage.value,
                        "source": "computed",
                        "run_id": "feature_store_write",
                        "ts_min": ts_min,
                        "ts_max": ts_max,
                        "count": len(batch),
                        "status": EventStatus.SUCCESS.value,
                    }
                    self.publisher.publish(topic, payload)
                except Exception:
                    logger.debug("FeatureStore publish failed", exc_info=True)
            return

        # Explicit-args mode
        if feature_set_id is None or instrument_id is None or features is None or ts_event is None:
            raise TypeError(
                "write_features requires explicit arguments or a FeatureData batch",
            )

        ts_init_val = int(ts_init) if ts_init is not None else int(ts_event)

        # Normalize features mapping defensively

        features_payload: dict[str, float] = {
            str(k): float(v) for k, v in dict(features or {}).items()
        }

        # Insert with ON CONFLICT for idempotency
        row = {
            "feature_set_id": feature_set_id,
            "instrument_id": instrument_id,
            "ts_event": int(ts_event),
            "ts_init": ts_init_val,
            "values": features_payload,
            "is_live": False,
            "source": "computed",
        }
        self._execute_write(row)
        # Optional publish single-row event
        if (
            self._enable_publishing
            and self.publisher is not None
            and instrument_id is not None
            and ts_event is not None
            and self._publish_mode in ("batch", "both")
            and publish_bus
        ):
            try:
                stage = Stage.FEATURE_COMPUTED
                topic = build_topic_for_stage(
                    stage,
                    instrument_id,
                    scheme=self._topic_scheme,
                    prefix=self._topic_prefix,
                )
                payload2: dict[str, Any] = {
                    "dataset_id": "features",
                    "instrument_id": instrument_id,
                    "stage": stage.value,
                    "source": "computed",
                    "run_id": "feature_store_write",
                    "ts_min": int(ts_event),
                    "ts_max": int(ts_event),
                    "count": 1,
                    "status": EventStatus.SUCCESS.value,
                }
                self.publisher.publish(topic, payload2)
            except Exception:
                logger.debug("FeatureStore publish failed", exc_info=True)

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
            component="feature_store",
            instrument_id=instrument_id,
            stage=stage,
            ts_stage_start=ts_stage_start,
            ts_stage_end=ts_stage_end,
            row_count=row_count,
        )

    def _execute_write(
        self,
        row: dict[str, Any],
    ) -> None:  # pragma: no cover (exercised in integration)
        """
        Upsert a single feature row (patchable in tests).
        """
        # Track stage boundary for observability (cold path only)
        ts_stage_start = time.time_ns()

        # Optional audit logging (sampled)
        try:
            import os
            import random

            sample = int(os.getenv("ML_AUDIT", "0"))
            if sample > 0 and random.randint(1, sample) == 1:
                logger.info("AUDIT FeatureStore._execute_write: keys=%s", list(row.keys()))
        except Exception:
            logger.debug(
                "Audit logging skipped due to error",
                exc_info=True,
            )
        # Final guard: normalize any incoming timestamps
        from ml.common.timestamps import sanitize_timestamp_ns

        if "ts_event" in row:
            row["ts_event"] = sanitize_timestamp_ns(
                int(row["ts_event"]),
                logger=logger,
                context="FeatureStore._execute_write",
            )
        if "ts_init" in row:
            row["ts_init"] = sanitize_timestamp_ns(
                int(row["ts_init"]),
                logger=logger,
                context="FeatureStore._execute_write",
            )

        stmt = insert(self.feature_values_table).values(row)
        stmt = stmt.on_conflict_do_update(
            index_elements=["feature_set_id", "instrument_id", "ts_event"],
            set_={
                "values": stmt.excluded["values"],
                "ts_init": stmt.excluded.ts_init,
                "source": stmt.excluded.source,
            },
        )
        cb = None
        try:
            from typing import cast as _cast

            from ml.stores.protocols import CircuitBreakerProtocol as _CBP

            cb = _cast("_CBP | None", getattr(self, "_circuit_breaker", None))
        except Exception:  # pragma: no cover - defensive
            cb = None

        if cb is not None and not cb.can_execute():
            return

        try:
            with self.engine.begin() as conn:
                conn.execute(stmt)
        except Exception:
            try:
                if cb is not None:
                    cb.record_failure()
            except Exception:
                pass
            raise
        else:
            try:
                if cb is not None:
                    cb.record_success()
            except Exception:
                pass

        # Optional per-row publish when enabled
        if (
            self._enable_publishing
            and self.publisher is not None
            and self._publish_mode in ("row", "both")
        ):
            try:
                stage = Stage.FEATURE_COMPUTED
                inst = str(row.get("instrument_id", "UNKNOWN"))
                topic = build_topic_for_stage(
                    stage,
                    inst,
                    scheme=self._topic_scheme,
                    prefix=self._topic_prefix,
                )
                ts_e = int(row.get("ts_event", 0))
                payload: dict[str, Any] = {
                    "dataset_id": "features",
                    "instrument_id": inst,
                    "stage": stage.value,
                    "source": str(row.get("source", "computed")),
                    "run_id": "feature_store_row",
                    "ts_min": ts_e,
                    "ts_max": ts_e,
                    "count": 1,
                    "status": EventStatus.SUCCESS.value,
                }
                self.publisher.publish(topic, payload)
            except Exception:
                logger.debug("FeatureStore per-row publish failed", exc_info=True)

        # Record observability data (off hot path - background processing only)
        ts_stage_end = time.time_ns()
        self._record_observability_stage_boundary(
            stage="feature_storage",
            instrument_id=str(row.get("instrument_id", "unknown")),
            ts_stage_start=ts_stage_start,
            ts_stage_end=ts_stage_end,
            row_count=1,
        )

    def _execute_query(self, sql: str) -> list[Any]:  # pragma: no cover (test hook)
        """
        Execute a SQL query and return rows (patchable).
        """
        from sqlalchemy import text as _text

        with self.engine.connect() as conn:
            result = conn.execute(_text(sql))
            try:
                return [dict(row) for row in result.mappings().all()]
            except Exception:
                return list(result.fetchall())

    def flush(self) -> None:
        """
        Flush any pending writes to storage.

        Note: FeatureStore currently writes synchronously, so this is a no-op.
        Future versions may implement write buffering for performance.

        """
        # Currently a no-op as writes are synchronous
        # Future: implement write buffering similar to ModelStore

    # Backwards-compatible batch API expected by integration tests
    def write_batch(self, data: list[object]) -> None:
        """
        Write a batch of FeatureData rows (compat shim).

        Parameters
        ----------
        data : list[FeatureData]
            Rows to upsert. Accepts objects with attributes
            feature_set_id, instrument_id, ts_event, ts_init, feature_values.

        """
        if not data:
            return

        # Append to buffer for visibility during the call (tests assert
        # the buffer is cleared after write_batch returns)
        from ml.stores.base import FeatureData as _FeatureData  # local to avoid cycles

        self._write_buffer.extend(cast(list[_FeatureData], data))

        for item in list(data):
            fs_id = getattr(item, "feature_set_id", None)
            inst = getattr(item, "instrument_id", None)
            tse = int(getattr(item, "ts_event", 0))
            tsi = int(getattr(item, "ts_init", tse))
            # Use feature_values to avoid colliding with mapping API on objects
            try:
                vals = getattr(item, "feature_values")
            except Exception:
                vals = {}
            row = {
                "feature_set_id": fs_id,
                "instrument_id": inst,
                "ts_event": tse,
                "ts_init": tsi,
                "values": dict(vals or {}),
                "is_live": False,
                "source": "computed",
            }
            self._execute_write(row)

        # Clear buffer after successful write
        self._write_buffer.clear()
        if self._enable_publishing and self.publisher is not None and data:
            try:
                stage = Stage.FEATURE_COMPUTED
                inst_any = getattr(data[0], "instrument_id", "UNKNOWN")
                topic = build_topic_for_stage(
                    stage,
                    str(inst_any),
                    scheme=self._topic_scheme,
                    prefix=self._topic_prefix,
                )
                ts_min = min(int(getattr(b, "ts_event", 0)) for b in data)
                ts_max = max(int(getattr(b, "ts_event", 0)) for b in data)
                payload: dict[str, Any] = {
                    "dataset_id": "features",
                    "instrument_id": str(inst_any),
                    "stage": stage.value,
                    "source": "computed",
                    "run_id": "feature_store_write",
                    "ts_min": ts_min,
                    "ts_max": ts_max,
                    "count": len(data),
                    "status": EventStatus.SUCCESS.value,
                }
                self.publisher.publish(topic, payload)
            except Exception:
                logger.debug("FeatureStore publish failed", exc_info=True)

    def is_healthy(self) -> bool:
        """
        Check if the feature store is healthy and accessible.

        Returns
        -------
        bool
            True if store is healthy, False otherwise

        """
        try:
            # Try a simple query to verify connection
            with self.engine.connect() as conn:
                from sqlalchemy import text

                result = conn.execute(text("SELECT 1"))
                return result is not None
        except Exception:
            return False

    def _get_connection(self) -> Any:  # pragma: no cover (test hook for patching)
        """
        Return a connection context manager (patchable in tests).
        """
        return self.engine.connect()

    # -------------------------------------------------------------------------------------
    # Compatibility reads and aliases
    # -------------------------------------------------------------------------------------

    def read_range(
        self,
        start_ns: int,
        end_ns: int,
        instrument_id: str | None = None,
    ) -> pd.DataFrame:
        """
        Read features in a time range (inclusive start, exclusive end).

        Parameters
        ----------
        start_ns : int
            Start timestamp in nanoseconds (inclusive).
        end_ns : int
            End timestamp in nanoseconds (exclusive).
        instrument_id : str | None
            Optional instrument filter.

        Returns
        -------
        pd.DataFrame
            A DataFrame of rows with columns: feature_set_id, instrument_id,
            values, ts_event, ts_init.

        """
        # Local import to avoid importing pandas at module import time
        import pandas as pd
        from sqlalchemy import text as _text

        where_parts: list[str] = ["ts_event >= :start_ns", "ts_event < :end_ns"]
        params: dict[str, Any] = {"start_ns": int(start_ns), "end_ns": int(end_ns)}
        if instrument_id is not None:
            where_parts.append("instrument_id = :instrument_id")
            params["instrument_id"] = instrument_id

        table_name = (
            "ml_feature_values"
            if self.engine.dialect.name == "sqlite"
            else "public.ml_feature_values"
        )
        sql = _text(
            f"""
                SELECT feature_set_id,
                       instrument_id,
                       values,
                       ts_event,
                       ts_init
                FROM {table_name}
                WHERE {' AND '.join(where_parts)}
                ORDER BY ts_event
                """,
        )
        # Prefer a mock-friendly session when available; else engine
        sess: Any | None = getattr(self, "persistence", None)
        session_obj: Any | None = None
        if sess is not None:
            # Prefer `.session` when present (MagicMock friendly), else try `get_session()`
            try:
                session_obj = getattr(sess, "session", None)
                if session_obj is None and hasattr(sess, "get_session"):
                    session_obj = sess.get_session()
            except Exception:
                session_obj = getattr(sess, "session", None)

        if session_obj is not None:
            # Use simple execute/fetch with manual DataFrame construction for MagicMock compatibility
            try:
                from sqlalchemy import text as _text2

                rows_obj = session_obj.execute(_text2(str(sql)), params).fetchall()
            except Exception:
                rows_obj = []

            rows_list: list[tuple[Any, ...]]
            try:
                rows_list = list(rows_obj)
            except TypeError:
                rows_list = []
            else:
                try:
                    from unittest.mock import MagicMock as _MagicMock

                    if any(isinstance(row, _MagicMock) for row in rows_list):
                        rows_list = []
                except Exception:
                    rows_list = []

            if rows_list:
                data = [
                    {
                        "feature_set_id": row[0],
                        "instrument_id": row[1],
                        "values": row[2],
                        "ts_event": row[3],
                        "ts_init": row[4],
                    }
                    for row in rows_list
                ]
                return pd.DataFrame(
                    data,
                    columns=[
                        "feature_set_id",
                        "instrument_id",
                        "values",
                        "ts_event",
                        "ts_init",
                    ],
                )

        with self.engine.connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def store_features(self, *args: Any, **kwargs: Any) -> None:
        """
        Backward-compatible alias for write_features with relaxed argument requirements.

        Accepts minimal explicit args used in integration tests: instrument_id,
        ts_event, and features. Fills feature_set_id from current pipeline/config and
        ts_init with ts_event when not provided.

        """
        if args or set(kwargs.keys()) & {"feature_set_id", "data"}:
            # Delegate when full signature or batch data is supplied
            self.write_features(*args, **kwargs)
            return

        instrument_id = kwargs.get("instrument_id")
        ts_event = kwargs.get("ts_event")
        features = kwargs.get("features")
        ts_init = kwargs.get("ts_init", ts_event)
        if instrument_id is None or ts_event is None or features is None:
            # Fallback to strict path
            self.write_features(*args, **kwargs)
            return

        self.write_features(
            feature_set_id=self._get_feature_set_id(),
            instrument_id=str(instrument_id),
            features=features,
            ts_event=int(cast(int, ts_event)),
            ts_init=int(cast(int, ts_init)),
        )
