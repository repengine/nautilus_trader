"""
Feature computation operations for FeatureStore.

This module handles feature calculation operations for both batch (historical) and
realtime modes.

"""

from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt

from ml.common.error_handlers import with_fallback
from ml.common.event_emitter import emit_dataset_event_and_watermark
from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage


if TYPE_CHECKING:
    from nautilus_trader.model.data import Bar

    from ml.features.engineering import FeatureEngineer as LegacyFeatureEngineer
    from ml.features.engineering import IndicatorManager
    from ml.features.facade import FeatureEngineer as FacadeFeatureEngineer
    from ml.registry.protocols import RegistryProtocol

    FeatureEngineerLike = LegacyFeatureEngineer | FacadeFeatureEngineer
else:
    FeatureEngineerLike = Any


logger = logging.getLogger(__name__)


class FeatureComputationProtocol(Protocol):
    """
    Protocol for feature computation.
    """

    def compute_and_store_historical(
        self,
        instrument_id: str,
        start_ts: int,
        end_ts: int,
        feature_config: Any,
    ) -> int:
        """
        Compute and store historical features.
        """
        ...

    def compute_realtime(
        self,
        instrument_id: str,
        bar: Any,
        feature_config: Any,
    ) -> dict[str, float] | None:
        """
        Compute features in real-time.
        """
        ...


class FeatureComputation:
    """
    Handles feature computation operations.

    IMPORTANT: This is COLD PATH only (Pattern 3).
    Real-time inference uses pre-computed features.

    """

    # Pattern 5: Metrics
    _COMPUTE_COUNTER = get_counter(
        "ml_feature_computations_total",
        "Total feature computations",
    )
    _COMPUTE_LATENCY = get_histogram(
        "ml_feature_compute_duration_seconds",
        "Feature computation duration",
    )

    def __init__(
        self,
        feature_engineer: FeatureEngineerLike,
        feature_versioning: Any,
        persistence: Any | None = None,
        retrieval: Any | None = None,
        indicator_manager: IndicatorManager | None = None,
        data_registry: RegistryProtocol | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize feature computation.

        Parameters
        ----------
        feature_engineer : FeatureEngineerLike
            Feature engineering pipeline
        feature_versioning : Any
            Feature versioning component for feature set IDs
        persistence : Any | None
            Feature persistence component for storing results
        retrieval : Any | None
            Feature retrieval component for loading bars
        indicator_manager : IndicatorManager | None
            Optional indicator manager for stateful indicators
        data_registry : RegistryProtocol | None
            Data registry for event emission
        logger : logging.Logger | None
            Logger

        """
        self._feature_engineer = feature_engineer
        self._feature_versioning = feature_versioning
        self._persistence = persistence
        self._retrieval = retrieval
        self._indicator_manager = indicator_manager
        self._data_registry = data_registry
        self._logger = logger or logging.getLogger(__name__)

        # Internal indicator managers for online computation
        self._indicator_managers: dict[str, IndicatorManager] = {}

    def compute_and_store_historical(
        self,
        instrument_id: str,
        start: datetime,
        end: datetime,
        feature_set_id: str,
        feature_names: list[str],
        force_recompute: bool = False,
    ) -> int:
        """
        Compute and store historical features.

        COLD PATH operation for training data preparation.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        start : datetime
            Start time
        end : datetime
            End time
        feature_set_id : str
            Feature set identifier for storage
        feature_names : list[str]
            List of feature names to compute
        force_recompute : bool, default False
            Whether to recompute even if features exist

        Returns
        -------
        int
            Number of features computed and stored

        """
        with self._COMPUTE_LATENCY.time():
            try:
                # Check if features already exist
                if not force_recompute and self._retrieval is not None:
                    if self._retrieval._features_exist(instrument_id, start, end):
                        self._logger.info(
                            "Features already exist for %s [%s, %s], skipping",
                            instrument_id,
                            start,
                            end,
                        )
                        return 0

                # Load bars from Nautilus tables
                if self._retrieval is None:
                    self._logger.error("Retrieval component not available")
                    return 0

                from ml.common.timestamps import sanitize_timestamp_ns

                start_ts = sanitize_timestamp_ns(
                    int(start.timestamp() * 1e9),
                    context="feature_computation.compute_historical.start",
                )
                end_ts = sanitize_timestamp_ns(
                    int(end.timestamp() * 1e9),
                    context="feature_computation.compute_historical.end",
                )

                bars_df = self._retrieval._load_bars_from_nautilus(
                    instrument_id,
                    start_ts,
                    end_ts,
                )

                # Check if bars were loaded
                if hasattr(bars_df, "is_empty"):
                    if bars_df.is_empty():
                        self._logger.warning(
                            "No bars found for %s in range [%s, %s]",
                            instrument_id,
                            start,
                            end,
                        )
                        return 0
                elif len(bars_df) == 0:
                    self._logger.warning(
                        "No bars found for %s in range [%s, %s]",
                        instrument_id,
                        start,
                        end,
                    )
                    return 0

                # Compute features (batch)
                features_df, _ = self._feature_engineer.calculate_features_batch(
                    bars_df,
                )

                # Use passed feature_names and feature_set_id
                # (Already provided by facade from versioning component)

                # Prepare rows with JSONB values mapping
                rows: list[dict[str, Any]] = []
                timestamps = bars_df["ts_event"].to_numpy()

                # Convert feature rows to dicts
                if hasattr(features_df, "iter_rows"):
                    # Polars path
                    from typing import cast

                    from ml.ml_types import PolarsDF

                    pf = cast(PolarsDF, features_df)
                    for i, row_vals in enumerate(pf.iter_rows()):
                        ts_event = int(timestamps[i])
                        values_map = {
                            name: float(row_vals[idx]) for idx, name in enumerate(feature_names)
                        }
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
                    from typing import cast

                    from ml.ml_types import PandasDF

                    pdf = cast(PandasDF, features_df)
                    for i in range(len(pdf)):
                        ts_event = int(timestamps[i])
                        row_series: Any = pdf.iloc[i]
                        values_map = {name: float(row_series[name]) for name in feature_names}
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

                # Store features
                if self._persistence is not None:
                    for row in rows:
                        self._persistence._execute_write(row)

                # Emit event
                self._emit_historical_event(instrument_id, timestamps, len(rows))

                # Update metrics
                self._COMPUTE_COUNTER.inc(amount=len(rows))

                self._logger.info(
                    "Computed and stored %d features for %s",
                    len(rows),
                    instrument_id,
                )

                return len(rows)

            except Exception as e:
                self._logger.error(
                    "Failed to compute historical features for %s: %s",
                    instrument_id,
                    e,
                    exc_info=True,
                )
                return 0

    def compute_realtime(
        self,
        bar: Bar,
        indicator_manager: IndicatorManager,
        feature_set_id: str,
        feature_names_online: list[str],
        store: bool = True,
    ) -> npt.NDArray[np.float32]:
        """
        Compute features in real-time from single bar.

        HOT PATH: Must be <5ms P99 latency.

        Parameters
        ----------
        bar : Bar
            Bar data
        indicator_manager : IndicatorManager
            Indicator manager for stateful indicators
        feature_set_id : str
            Feature set identifier for storage
        feature_names_online : list[str]
            List of online feature names
        store : bool, default True
            Whether to store computed features

        Returns
        -------
        npt.NDArray[np.float32]
            Computed feature vector

        """
        # Update indicators from bar and compute online features (indicator_manager is required)
        indicator_manager.update_from_bar(bar)
        if not indicator_manager.all_initialized():
            # Not enough history yet - return empty array
            return np.zeros(0, dtype=np.float32)

        current_bar = {
            "close": float(bar.close),
            "volume": float(bar.volume),
            "high": float(bar.high),
            "low": float(bar.low),
        }

        features = self._feature_engineer.calculate_features_online(
            current_bar=current_bar,
            indicator_manager=indicator_manager,
            scaler=None,
        )

        # Optionally store for future training
        if store and features.size > 0 and self._persistence is not None:
            # Use passed feature_names_online
            values_map = {
                name: float(features[idx])
                for idx, name in enumerate(feature_names_online)
                if idx < features.size
            }

            from ml.common.timestamps import sanitize_timestamp_ns

            tse_norm = sanitize_timestamp_ns(
                int(bar.ts_event),
                logger=self._logger,
                context="FeatureComputation.realtime",
            )
            tsi_norm = sanitize_timestamp_ns(
                int(bar.ts_init),
                logger=self._logger,
                context="FeatureComputation.realtime",
            )

            row = {
                "feature_set_id": feature_set_id,  # Use passed feature_set_id
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
            }

            self._persistence._execute_write(row)

        return features

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
        Compute historical features in parallel.

        COLD PATH operation for bulk training data preparation.

        Parameters
        ----------
        instrument_ids : list[str]
            List of instrument identifiers
        start : datetime | None
            Start time
        end : datetime | None
            End time
        force_recompute : bool, default False
            Whether to recompute even if features exist
        max_workers : int, default 4
            Maximum parallel workers

        Returns
        -------
        dict[str, int]
            Mapping of instrument to number of features computed

        """
        results: dict[str, int] = {}

        if not instrument_ids:
            return results

        # Cap workers to reasonable limit
        workers = max(1, min(max_workers, 8))

        # Get feature_set_id and feature_names once from versioning
        feature_set_id = self._feature_versioning.get_feature_set_id()
        feature_names = self._feature_versioning.get_feature_names(online=False)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_to_inst = {
                ex.submit(
                    self.compute_and_store_historical,
                    instrument_id=inst,
                    start=start or datetime.now(UTC) - timedelta(days=1),
                    end=end or datetime.now(UTC),
                    feature_set_id=feature_set_id,
                    feature_names=feature_names,
                    force_recompute=force_recompute,
                ): inst
                for inst in instrument_ids
            }

            for fut in as_completed(fut_to_inst):
                inst = fut_to_inst[fut]
                try:
                    results[inst] = int(fut.result())
                except Exception:
                    self._logger.error(
                        "Parallel feature compute failed for %s",
                        inst,
                        exc_info=True,
                    )
                    results[inst] = 0

        return results

    @with_fallback(
        fallback_value=None,
        log_level="warning",
        operation_name="emit feature computation event",
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

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        timestamps : npt.NDArray[np.int64]
            Array of timestamps for the computed features
        row_count : int
            Number of rows computed

        """
        if self._data_registry is None:
            return

        # Generate unique run ID
        run_id = f"feature_historical_{uuid.uuid4().hex[:8]}_{int(time.time())}"

        # Use canonical dataset id
        feature_set_id = self._feature_versioning.get_feature_set_id()
        dataset_id = "features"

        # Get the time range from timestamps
        ts_min = int(timestamps[0]) if len(timestamps) > 0 else 0
        ts_max = int(timestamps[-1]) if len(timestamps) > 0 else 0

        # Emit via shared helper
        emit_dataset_event_and_watermark(
            self._data_registry,
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

        self._logger.debug(
            "Emitted FEATURE_COMPUTED event: dataset=%s, instrument=%s, count=%d",
            dataset_id,
            instrument_id,
            row_count,
        )
