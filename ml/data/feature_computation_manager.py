"""
Feature computation management for DataScheduler.

Manages feature computation for newly collected data, coordinating between
the feature engineer, catalog, and feature store.

This component implements Pattern 3 (Hot/Cold Path Separation) - this is COLD PATH only.

"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Protocol

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_gauge
from ml.common.metrics_bootstrap import get_histogram


if TYPE_CHECKING:

    from ml.config.scheduler_config import SchedulerConfig
    from ml.data.trading_day_calculator import TradingDayCalculator
    from ml.features.engineering import FeatureEngineer as LegacyFeatureEngineer
    from ml.features.facade import FeatureEngineer as FacadeFeatureEngineer
    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

    FeatureEngineerLike = LegacyFeatureEngineer | FacadeFeatureEngineer
else:
    FeatureEngineerLike = Any


# =============================================================================
# PROTOCOL DEFINITION
# =============================================================================


class FeatureComputationManagerProtocol(Protocol):
    """
    Protocol for feature computation management.

    Implements Pattern 2 (Protocol-First Interface Design) for structural typing.

    """

    def compute_features(
        self,
    ) -> tuple[int, list[str]]:
        """
        Compute features for newly collected data.

        Returns
        -------
        tuple[int, list[str]]
            (total_features_computed, failed_instruments)

        """
        ...


# =============================================================================
# METRICS DEFINITIONS (Pattern 5: Centralized Metrics Bootstrap)
# =============================================================================

active_feature_tasks = get_gauge(
    "nautilus_ml_active_feature_tasks",
    "Number of active feature computation tasks",
)

feature_computation_errors_total = get_counter(
    "nautilus_ml_feature_computation_errors_total",
    "Total errors during feature computation",
    ["instrument", "error_type"],
)

feature_store_latency = get_histogram(
    "nautilus_ml_feature_store_latency_seconds",
    "Feature store operation latency",
    ["operation"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# Load centralized metrics if available
try:
    from ml.common.metrics import feature_computation_duration as _feature_comp_latency
    from ml.common.metrics import feature_store_operations_total as _feature_store_ops

    feature_computation_latency = _feature_comp_latency
    feature_store_operations_total = _feature_store_ops
except Exception:
    # Fallback to no-op metrics
    class _NoOpMetric:
        def labels(self, **_: object) -> _NoOpMetric:
            return self

        def inc(self, *_: object, **__: object) -> None:
            pass

        def observe(self, *_: object, **__: object) -> None:
            pass

    feature_computation_latency = _NoOpMetric()
    feature_store_operations_total = _NoOpMetric()


# =============================================================================
# FEATURE COMPUTATION MANAGER IMPLEMENTATION
# =============================================================================


class FeatureComputationManager:
    """
    Manages feature computation for scheduled data.

    Integrates with FeatureStore to compute and persist features for ML training.
    Implements Pattern 3 (Hot/Cold Path): This is COLD PATH only - batch processing
    with no hot path performance requirements.

    """

    def __init__(
        self,
        catalog: ParquetDataCatalog,
        config: SchedulerConfig,
        feature_engineer: FeatureEngineerLike | None,
        feature_store: Any | None,
        trading_day_calc: TradingDayCalculator,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize feature computation manager.

        Parameters
        ----------
        catalog : ParquetDataCatalog
            Nautilus data catalog for querying bars
        config : SchedulerConfig
            Scheduler configuration (symbols, feature store settings)
        feature_engineer : FeatureEngineerLike | None
            Feature engineer for computing features
        feature_store : Any | None
            Feature store instance for persisting features
        trading_day_calc : TradingDayCalculator
            Trading day calculator for date range computation
        logger : logging.Logger | None
            Logger for operations

        """
        self._catalog = catalog
        self._config = config
        self._feature_engineer = feature_engineer
        self._feature_store = feature_store
        self._trading_day_calc = trading_day_calc
        self._logger = logger or logging.getLogger(__name__)

    def compute_features(
        self,
    ) -> tuple[int, list[str]]:
        """
        Compute features for newly collected data.

        This method:
        1. Queries the catalog for recent bars data
        2. Computes features using FeatureEngineer (batch mode)
        3. Stores features in FeatureStore for training/inference parity

        Returns
        -------
        tuple[int, list[str]]
            (total_features_computed, failed_instruments)

        """
        # Check if feature computation is enabled and configured
        if not self._config.feature_store_enabled:
            self._logger.debug(
                "Feature store disabled in configuration, skipping feature computation"
            )
            return (0, [])

        if self._feature_engineer is None:
            self._logger.debug("No feature engineer configured, skipping feature computation")
            return (0, [])

        if self._feature_store is None:
            self._logger.warning("Feature store not initialized, skipping feature computation")
            return (0, [])

        self._logger.info("Starting feature computation for new data...")

        # Import required modules
        from nautilus_trader.model.data import Bar
        from nautilus_trader.model.identifiers import InstrumentId

        from ml._imports import HAS_POLARS
        from ml._imports import check_ml_dependencies

        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        # Track metrics
        total_features_computed = 0
        failed_instruments: list[str] = []
        start_time = time.perf_counter()

        # Update active feature tasks gauge
        active_feature_tasks.set(len(self._config.symbols))

        try:
            # Get date range for feature computation (delegate to TradingDayCalculator)
            target_date = self._trading_day_calc.get_previous_trading_day()
            start_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)

            self._logger.info(
                f"Computing features for date range: {start_date.date()} to {end_date.date()}",
            )

            # Process each configured symbol
            for idx, symbol in enumerate(self._config.symbols):
                # Update active tasks
                active_feature_tasks.set(len(self._config.symbols) - idx)
                try:
                    # Parse symbol to get instrument_id
                    symbol_parts = symbol.split(".")
                    if len(symbol_parts) != 2:
                        self._logger.warning(f"Invalid symbol format: {symbol}, skipping")
                        failed_instruments.append(symbol)
                        continue

                    symbol_code, venue = symbol_parts

                    # Map common venue codes to Nautilus format
                    venue_map = {
                        "XNAS": "NASDAQ",
                        "XNYS": "NYSE",
                        "ARCX": "ARCA",
                        "BATS": "BATS",
                        "GLBX": "GLBX",
                    }
                    nautilus_venue = venue_map.get(venue, venue)
                    instrument_id = InstrumentId.from_str(f"{symbol_code}.{nautilus_venue}")

                    self._logger.debug(f"Processing features for {instrument_id}")

                    # Query bars from catalog
                    bars_data = self._catalog.query(
                        data_cls=Bar,
                        identifiers=[str(instrument_id)],
                        start=int(start_date.timestamp() * 1e9),
                        end=int(end_date.timestamp() * 1e9),
                    )

                    if not bars_data:
                        self._logger.warning(
                            f"No bars found for {instrument_id} on {target_date.date()}"
                        )
                        continue

                    self._logger.info(f"Found {len(bars_data)} bars for {instrument_id}")

                    # Store features in FeatureStore for future training
                    store_start_time = time.perf_counter()
                    try:
                        stored_count = self._feature_store.compute_and_store_historical(
                            instrument_id=str(instrument_id),
                            start=start_date,
                            end=end_date,
                            force_recompute=True,  # Force recompute for fresh data
                        )

                        # Record feature store metrics
                        store_duration = time.perf_counter() - store_start_time
                        feature_store_operations_total.labels(
                            operation="store_historical",
                            status="success",
                        ).inc()
                        feature_store_latency.labels(
                            operation="store_historical",
                        ).observe(store_duration)
                        feature_computation_latency.labels(
                            instrument=str(instrument_id),
                            stage="store",
                        ).observe(store_duration)

                        total_features_computed += stored_count
                        self._logger.info(
                            f"Stored {stored_count} feature rows for {instrument_id} in FeatureStore",
                        )
                        # Note: FEATURE_COMPUTED events are emitted by FeatureStore itself
                        # to avoid double-counting in metrics
                    except Exception:
                        feature_store_operations_total.labels(
                            operation="store_historical",
                            status="failure",
                        ).inc()
                        self._logger.error(
                            "Failed to store features for %s",
                            instrument_id,
                            exc_info=True,
                        )
                        failed_instruments.append(str(instrument_id))
                        continue

                except Exception:
                    self._logger.error(
                        "Error processing symbol %s",
                        symbol,
                        exc_info=True,
                    )
                    feature_computation_errors_total.labels(
                        instrument=symbol,
                        error_type="processing_error",
                    ).inc()
                    failed_instruments.append(symbol)
                    continue

            # Calculate elapsed time
            elapsed_time = time.perf_counter() - start_time

            # Log summary statistics
            self._logger.info(
                f"Feature computation completed in {elapsed_time:.2f}s: "
                f"{total_features_computed} features computed, "
                f"{len(failed_instruments)} failures",
            )

            if failed_instruments:
                self._logger.warning(f"Failed instruments: {', '.join(failed_instruments)}")

            # Log performance metrics for monitoring
            if total_features_computed > 0:
                avg_time_per_feature = elapsed_time / total_features_computed
                self._logger.info(
                    f"Average computation time: {avg_time_per_feature*1000:.2f}ms per feature row",
                )

            return (total_features_computed, failed_instruments)

        except Exception:
            self._logger.error(
                "Critical error in feature computation",
                exc_info=True,
            )
            feature_computation_errors_total.labels(
                instrument="all",
                error_type="critical",
            ).inc()
            raise
        finally:
            # Reset active tasks
            active_feature_tasks.set(0)
