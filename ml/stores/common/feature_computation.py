#!/usr/bin/env python3

"""
Feature computation component for FeatureStore.

Extracted from FeatureStore (Phase 3.7.3). Provides real-time and historical
feature computation with indicator manager integration and training/inference parity.

CRITICAL: compute_realtime() is HOT PATH - P99 < 5ms requirement.
- NO DataFrame creation
- NO file I/O
- NO network calls
- Pre-allocate arrays
- Reuse indicator managers

"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ml.common.error_handlers import with_fallback
from ml.common.event_emitter import emit_dataset_event_and_watermark
from ml.common.timestamps import sanitize_timestamp_ns
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage


if TYPE_CHECKING:
    from nautilus_trader.model.data import Bar
    from polars import DataFrame as PlDataFrame
    from sqlalchemy import Table
    from sqlalchemy.engine import Engine

    from ml.features.engineering import FeatureConfig
    from ml.features.engineering import FeatureEngineer
    from ml.features.engineering import IndicatorManager
    from ml.registry.protocols import RegistryProtocol
    from ml.stores.common.feature_reader import FeatureReaderComponent
    from ml.stores.common.feature_writer import FeatureWriterComponent
    from ml.stores.protocols import CircuitBreakerProtocol


logger = logging.getLogger(__name__)


# =========================================================================
# Protocols
# =========================================================================


@runtime_checkable
class FeatureSchemaProtocol(Protocol):
    """
    Protocol for feature schema operations.

    Defines the interface for retrieving feature names and feature set IDs.

    """

    def get_feature_names(self) -> list[str]:
        """
        Get OFFLINE feature names from pipeline or config.

        Returns:
            List of feature name strings

        """
        ...

    def get_feature_names_online(self) -> list[str]:
        """
        Get ONLINE (hot-path) feature names with L1_ONLY gating.

        Returns:
            List of online feature name strings

        """
        ...

    def get_feature_set_id(self) -> str:
        """
        Derive a stable feature_set_id for storage.

        Returns:
            Feature set identifier string

        """
        ...


@runtime_checkable
class FeatureComputationProtocol(Protocol):
    """
    Protocol for feature computation operations.

    Defines the interface for computing features in real-time (hot path) and historical
    batch mode (cold path).

    """

    def compute_realtime(
        self,
        bar: Bar,
        store: bool = True,
        indicator_manager: IndicatorManager | None = None,
    ) -> npt.NDArray[np.float32]:
        """
        Compute features for real-time inference.

        HOT PATH: P99 < 5ms requirement.
        - NO DataFrame creation
        - NO file I/O
        - NO network calls

        Args:
            bar: Current bar from Nautilus
            store: Whether to store computed features for future training
            indicator_manager: Optional indicator manager for stateful computation

        Returns:
            Computed feature vector as float32 array

        """
        ...

    def compute_and_store_historical(
        self,
        instrument_id: str,
        start: datetime,
        end: datetime,
        force_recompute: bool = False,
    ) -> int:
        """
        Compute and store features for historical data.

        COLD PATH: Batch processing for training data.

        Args:
            instrument_id: Instrument to compute features for
            start: Start time for historical computation
            end: End time for historical computation
            force_recompute: If True, recompute even if features exist

        Returns:
            Number of feature rows computed and stored

        """
        ...

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

        COLD PATH: Parallel batch processing.

        Args:
            instrument_ids: Instruments to compute
            start: Start time (inclusive)
            end: End time (exclusive)
            force_recompute: Recompute even if features exist
            max_workers: Maximum concurrent workers

        Returns:
            Mapping instrument_id -> rows written (0 on failure)

        """
        ...


# =========================================================================
# Configuration
# =========================================================================


@dataclass(frozen=True)
class FeatureComputationConfig:
    """
    Configuration for FeatureComputationComponent.

    Attributes
    ----------
    max_parallel_workers : int
        Maximum number of parallel workers for batch computation (default: 4)
    default_lookback_days : int
        Default lookback period in days when start is not specified (default: 1)

    """

    max_parallel_workers: int = 4
    default_lookback_days: int = 1

    def __post_init__(self) -> None:
        """
        Validate configuration values.
        """
        if self.max_parallel_workers < 1:
            raise ValueError(
                f"max_parallel_workers must be >= 1, got {self.max_parallel_workers}",
            )
        if self.max_parallel_workers > 8:
            raise ValueError(
                f"max_parallel_workers capped at 8 to avoid DB pool exhaustion, got {self.max_parallel_workers}",
            )
        if self.default_lookback_days < 1:
            raise ValueError(
                f"default_lookback_days must be >= 1, got {self.default_lookback_days}",
            )


# =========================================================================
# Component Implementation
# =========================================================================


@dataclass
class FeatureComputationComponent:
    """
    Feature computation operations for FeatureStore.

    Extracted from FeatureStore (Phase 3.7.3).

    Provides:
    - compute_realtime() - Real-time feature computation (HOT PATH)
    - compute_and_store_historical() - Historical batch computation
    - compute_historical_parallel() - Parallel batch computation

    CRITICAL: compute_realtime() is HOT PATH - P99 < 5ms requirement.
    NO DataFrame creation, NO file I/O, NO network calls in hot path.

    Example
    -------
    >>> from ml.stores.common.feature_computation import FeatureComputationComponent
    >>> computation = FeatureComputationComponent(
    ...     engine=engine,
    ...     table=feature_values_table,
    ...     feature_engineer=feature_engineer,
    ...     feature_writer=writer,
    ...     feature_reader=reader,
    ...     get_feature_set_id=lambda: "fs_001",
    ...     get_feature_names=lambda: ["close_return", "volume_ratio"],
    ...     get_feature_names_online=lambda: ["close_return"],
    ... )
    >>> features = computation.compute_realtime(bar, store=True)

    """

    engine: Engine
    table: Table
    feature_engineer: FeatureEngineer
    feature_writer: FeatureWriterComponent
    feature_reader: FeatureReaderComponent
    get_feature_set_id: Callable[[], str]
    get_feature_names: Callable[[], list[str]]
    get_feature_names_online: Callable[[], list[str]]
    feature_config: FeatureConfig | None = None
    circuit_breaker: CircuitBreakerProtocol | None = None
    data_registry: RegistryProtocol | None = None
    config: FeatureComputationConfig = field(default_factory=FeatureComputationConfig)
    _indicator_managers: dict[str, Any] = field(default_factory=dict)

    def compute_realtime(
        self,
        bar: Bar,
        store: bool = True,
        indicator_manager: IndicatorManager | None = None,
    ) -> npt.NDArray[np.float32]:
        """
        Compute features for real-time inference.

        HOT PATH: P99 < 5ms requirement.
        - NO DataFrame creation
        - NO file I/O (except optional storage)
        - NO network calls
        - Pre-allocate arrays
        - Reuse indicator managers

        Uses the SAME FeatureEngineer as historical computation to ensure
        perfect parity between training and inference.

        Args:
            bar: Current bar from Nautilus
            store: Whether to store computed features for future training
            indicator_manager: Optional indicator manager for stateful computation.
                If not provided, uses internal indicator manager per instrument.

        Returns:
            Computed feature vector as float32 array.
            Returns empty array (size 0) if indicators not warmed up yet.

        Example
        -------
        >>> features = computation.compute_realtime(bar, store=True)
        >>> if features.size > 0:
        ...     # Indicators warmed up, can run inference
        ...     prediction = model.predict(features)

        """
        # Extract instrument key (avoid method calls in hot path)
        instrument_key = self._get_instrument_key(bar)

        # Get or create indicator manager (prefer provided from actor)
        if indicator_manager is None:
            indicator_manager = self._get_or_create_indicator_manager(instrument_key)

        # Update indicators from bar and compute online features
        indicator_manager.update_from_bar(bar)
        if not indicator_manager.all_initialized():
            # Not enough history yet - return empty array to signal no prediction
            return np.zeros(0, dtype=np.float32)

        # Compute features using FeatureEngineer (ensures parity)
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
            self._store_realtime_features(bar, features)

        return features

    def _get_instrument_key(self, bar: Bar) -> str:
        """
        Extract instrument key from bar object.

        Args:
            bar: Nautilus Bar object

        Returns:
            Instrument key string

        """
        if hasattr(bar, "bar_type") and hasattr(bar.bar_type, "instrument_id"):
            return str(bar.bar_type.instrument_id)
        return str(getattr(bar, "instrument_id", "unknown"))

    def _get_or_create_indicator_manager(self, instrument_key: str) -> Any:
        """
        Get existing or create new indicator manager for instrument.

        Args:
            instrument_key: Instrument identifier string

        Returns:
            IndicatorManager instance

        """
        indicator_manager = self._indicator_managers.get(instrument_key)
        if indicator_manager is None:
            # Import here to avoid circular dependency at module level
            from ml.features.engineering import IndicatorManager as IM

            if self.feature_config is not None:
                indicator_manager = IM(self.feature_config)
            else:
                indicator_manager = IM(self.feature_engineer.config)
            self._indicator_managers[instrument_key] = indicator_manager
        return indicator_manager

    def _store_realtime_features(
        self,
        bar: Bar,
        features: npt.NDArray[np.float32],
    ) -> None:
        """
        Store computed features from real-time computation.

        Args:
            bar: Source bar for timestamps and instrument ID
            features: Computed feature array

        """
        # Check circuit breaker before attempting storage
        cb = self.circuit_breaker
        if cb is not None and not cb.can_execute():
            return

        # Build feature values map
        feature_names = self.get_feature_names_online()
        values_map = {
            name: float(features[idx])
            for idx, name in enumerate(feature_names)
            if idx < features.size
        }

        # Normalize timestamps
        tse_norm = sanitize_timestamp_ns(
            int(bar.ts_event),
            logger=logger,
            context="FeatureComputation.realtime",
        )
        tsi_norm = sanitize_timestamp_ns(
            int(bar.ts_init),
            logger=logger,
            context="FeatureComputation.realtime",
        )

        # Build row for storage
        instrument_id = self._get_instrument_key(bar)
        row = {
            "feature_set_id": self.get_feature_set_id(),
            "instrument_id": instrument_id,
            "ts_event": tse_norm,
            "ts_init": tsi_norm,
            "values": values_map,
            "is_live": True,
            "source": "live",
        }

        try:
            self._execute_realtime_upsert(row)
        except Exception:
            if cb is not None:
                try:
                    cb.record_failure()
                except Exception:
                    logger.debug(
                        "Failed to record circuit breaker failure for realtime feature storage",
                        exc_info=True,
                    )
            raise
        else:
            if cb is not None:
                try:
                    cb.record_success()
                except Exception:
                    logger.debug(
                        "Failed to record circuit breaker success for realtime feature storage",
                        exc_info=True,
                    )

            # Emit FEATURE_COMPUTED event for successful realtime computation
            self._emit_realtime_event(bar, instrument_id)

    def _execute_realtime_upsert(self, row: dict[str, Any]) -> None:
        """
        Execute upsert for real-time feature row.

        Args:
            row: Feature row dictionary

        """
        from sqlalchemy.dialects.postgresql import insert

        with self.engine.begin() as conn:
            stmt: Any = insert(self.table)
            stmt = stmt.on_conflict_do_update(
                index_elements=["feature_set_id", "instrument_id", "ts_event"],
                set_={
                    "values": stmt.excluded["values"],
                    "ts_init": stmt.excluded.ts_init,
                    "is_live": stmt.excluded.is_live,
                    "source": stmt.excluded.source,
                },
            )
            conn.execute(stmt, row)

    def _emit_realtime_event(self, bar: Bar, instrument_id: str) -> None:
        """
        Emit FEATURE_COMPUTED event for real-time computation.

        Non-blocking operation - failures are logged but don't affect feature computation.

        Args:
            bar: Source bar for timestamp
            instrument_id: Instrument identifier

        """
        try:
            registry = self.data_registry
            if not registry:
                return

            run_id = f"feature_realtime_{uuid.uuid4().hex[:8]}_{int(time.time())}"
            feature_set_id = self.get_feature_set_id()
            dataset_id = "features"

            emit_dataset_event_and_watermark(
                registry,
                dataset_id=dataset_id,
                instrument_id=instrument_id,
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
                instrument_id,
                int(bar.ts_event),
            )
        except Exception:
            # Non-blocking: log but don't fail the feature computation
            logger.warning(
                "Failed to emit realtime feature event",
                exc_info=True,
            )

    def compute_and_store_historical(
        self,
        instrument_id: str,
        start: datetime,
        end: datetime,
        force_recompute: bool = False,
    ) -> int:
        """
        Compute and store features for historical data.

        COLD PATH: Batch processing for training data.

        This method:
        1. Checks if features already exist (skip if not force_recompute)
        2. Loads bars from Nautilus PostgreSQL tables
        3. Computes features using FeatureEngineer (same logic as live)
        4. Stores features in ml_feature_values table

        Args:
            instrument_id: Instrument to compute features for
            start: Start time for historical computation
            end: End time for historical computation
            force_recompute: If True, recompute even if features exist

        Returns:
            Number of feature rows computed and stored

        Example
        -------
        >>> rows = computation.compute_and_store_historical(
        ...     instrument_id="SPY.DATABENTO",
        ...     start=datetime(2024, 1, 1),
        ...     end=datetime(2024, 1, 2),
        ...     force_recompute=False,
        ... )
        >>> print(f"Stored {rows} feature rows")

        """
        # Check if features already exist
        if not force_recompute and self.feature_reader.features_exist(
            instrument_id,
            start,
            end,
        ):
            return 0

        # Load bars from Nautilus tables
        bars_df = self._load_bars_from_nautilus(instrument_id, start, end)
        if bars_df.is_empty():
            return 0

        # Compute features (batch) ensuring parity with online
        features_df, _ = self.feature_engineer.calculate_features_batch(bars_df)

        feature_names = self.get_feature_names()
        feature_set_id = self.get_feature_set_id()

        # Prepare rows with JSONB values mapping
        rows = self._prepare_historical_rows(
            features_df,
            bars_df,
            feature_names,
            feature_set_id,
            instrument_id,
        )

        if not rows:
            return 0

        # Bulk upsert into partitioned table
        self._execute_historical_bulk_upsert(rows)

        # Emit FEATURE_COMPUTED event for successful historical computation
        timestamps = bars_df["ts_event"].to_numpy()
        self._emit_historical_event(instrument_id, timestamps, len(rows))

        return len(rows)

    def _load_bars_from_nautilus(
        self,
        instrument_id: str,
        start: datetime,
        end: datetime,
    ) -> PlDataFrame:
        """
        Load bars from Nautilus PostgreSQL tables.

        Args:
            instrument_id: Instrument identifier
            start: Start time
            end: End time

        Returns:
            Polars DataFrame with Nautilus bar schema

        """
        from typing import Any as _Any
        from typing import cast as _cast

        import pandas as pd
        from sqlalchemy import text as _text

        from ml._imports import pl

        pl = _cast(_Any, pl)

        start_ns = sanitize_timestamp_ns(
            int(start.timestamp() * 1e9),
            context="feature_computation._load_bars.start",
        )
        end_ns = sanitize_timestamp_ns(
            int(end.timestamp() * 1e9),
            context="feature_computation._load_bars.end",
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

            _params = _cast(
                Mapping[str, _Any],
                {
                    "instrument_id": instrument_id,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                },
            )
            pdf = pd.read_sql_query(sql, conn, params=_params)

        return _cast("PlDataFrame", pl.from_pandas(pdf))

    def _prepare_historical_rows(
        self,
        features_df: Any,
        bars_df: Any,
        feature_names: list[str],
        feature_set_id: str,
        instrument_id: str,
    ) -> list[dict[str, Any]]:
        """
        Prepare rows for historical bulk upsert.

        Args:
            features_df: DataFrame with computed features
            bars_df: DataFrame with bar data (for timestamps)
            feature_names: List of feature names
            feature_set_id: Feature set identifier
            instrument_id: Instrument identifier

        Returns:
            List of row dictionaries for upsert

        """
        from typing import cast

        from ml.ml_types import PandasDF
        from ml.ml_types import PolarsDF

        rows: list[dict[str, Any]] = []
        timestamps = bars_df["ts_event"].to_numpy()

        # Convert feature rows to dicts based on DataFrame type
        if hasattr(features_df, "iter_rows"):
            # Polars path
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
                    },
                )
        else:
            # Pandas path
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
                    },
                )

        return rows

    def _execute_historical_bulk_upsert(self, rows: list[dict[str, Any]]) -> None:
        """
        Execute bulk upsert for historical feature rows.

        Args:
            rows: List of feature row dictionaries

        """
        from sqlalchemy.dialects.postgresql import insert

        with self.engine.begin() as conn:
            stmt: Any = insert(self.table)
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
                },
            )
            conn.execute(stmt, rows)

    @with_fallback(
        fallback_value=None,
        log_level="warning",
        operation_name="emit historical event",
    )
    def _emit_historical_event(
        self,
        instrument_id: str,
        timestamps: npt.NDArray[np.int64],
        row_count: int,
    ) -> None:
        """
        Emit FEATURE_COMPUTED event for historical computation.

        Non-blocking operation - failures are logged but don't affect feature computation.

        Args:
            instrument_id: Instrument identifier
            timestamps: Array of timestamps for the computed features
            row_count: Number of rows computed

        """
        registry = self.data_registry
        if not registry:
            return

        run_id = f"feature_historical_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        feature_set_id = self.get_feature_set_id()
        dataset_id = "features"

        ts_min = int(timestamps[0]) if len(timestamps) > 0 else 0
        ts_max = int(timestamps[-1]) if len(timestamps) > 0 else 0

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

        COLD PATH: Parallel batch processing for multiple instruments.

        Args:
            instrument_ids: Instruments to compute
            start: Start time (inclusive). Defaults to now - default_lookback_days.
            end: End time (exclusive). Defaults to now.
            force_recompute: Recompute even if features exist
            max_workers: Maximum concurrent workers (bounded to avoid pool exhaustion)

        Returns:
            Mapping instrument_id -> rows written (0 on failure)

        Example
        -------
        >>> results = computation.compute_historical_parallel(
        ...     instrument_ids=["SPY.DATABENTO", "AAPL.DATABENTO"],
        ...     start=datetime(2024, 1, 1),
        ...     end=datetime(2024, 1, 2),
        ...     max_workers=4,
        ... )
        >>> for inst, rows in results.items():
        ...     print(f"{inst}: {rows} rows")

        """
        results: dict[str, int] = {}

        if not instrument_ids:
            return results

        # Apply defaults for start/end
        effective_end = end or datetime.now(UTC)
        effective_start = start or (
            effective_end - timedelta(days=self.config.default_lookback_days)
        )

        # Cap workers to a reasonable limit to play nicely with DB pools
        workers = max(1, min(max_workers, self.config.max_parallel_workers))

        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_to_inst = {
                ex.submit(
                    self.compute_and_store_historical,
                    instrument_id=inst,
                    start=effective_start,
                    end=effective_end,
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

    def get_indicator_manager(self, instrument_id: str) -> Any | None:
        """
        Get indicator manager for an instrument if it exists.

        Args:
            instrument_id: Instrument identifier

        Returns:
            IndicatorManager instance or None if not found

        """
        return self._indicator_managers.get(instrument_id)

    def set_data_registry(self, registry: RegistryProtocol) -> None:
        """
        Set the DataRegistry instance used for event emission.

        Args:
            registry: The shared registry instance to use

        """
        self.data_registry = registry


__all__ = [
    "FeatureComputationComponent",
    "FeatureComputationConfig",
    "FeatureComputationProtocol",
    "FeatureSchemaProtocol",
]
