"""
Feature computation component extracted from DataScheduler.

This component handles feature computation logic including:
- Computing features for newly collected data
- Lazy initialization of FeatureStore
- Venue code mapping (e.g., XNAS -> NASDAQ)
- Prometheus metrics tracking for feature computation

Extracted from legacy DataScheduler (lines 1196-1385):
- _compute_features() method

"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_gauge
from ml.common.metrics_bootstrap import get_histogram


if TYPE_CHECKING:
    from ml.config.scheduler_config import SchedulerConfig
    from ml.features.engineering import FeatureEngineer
    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


logger = logging.getLogger(__name__)


# =============================================================================
# VENUE MAPPING
# =============================================================================

VENUE_MAP: dict[str, str] = {
    "XNAS": "NASDAQ",
    "XNYS": "NYSE",
    "ARCX": "ARCA",
    "BATS": "BATS",
    "GLBX": "GLBX",
}


# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

# Import module-level metrics from scheduler module for compatibility
# These are created at module load and must be the same instances
try:
    from ml.data.scheduler import active_feature_tasks
    from ml.data.scheduler import feature_computation_errors_total
    from ml.data.scheduler import feature_store_latency
    from ml.data.scheduler import feature_store_operations_total
except ImportError:
    # Fallback for isolated testing - create local metrics
    active_feature_tasks = get_gauge(
        "nautilus_ml_active_feature_tasks",
        "Number of active feature computation tasks",
    )
    feature_store_operations_total = get_counter(
        "nautilus_ml_feature_store_operations_total",
        "Total feature store operations",
        ["operation", "status"],
    )
    feature_store_latency = get_histogram(
        "nautilus_ml_feature_store_latency_seconds",
        "Feature store operation latency",
        ["operation"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    feature_computation_errors_total = get_counter(
        "nautilus_ml_feature_computation_errors_total",
        "Total errors during feature computation",
        ["instrument", "error_type"],
    )

# Create component-specific latency metric with the labels we need
# This uses a unique name to avoid conflicts with the common.metrics version
# which has different labels
feature_computation_store_latency = get_histogram(
    "nautilus_ml_feature_computation_store_latency_seconds",
    "Feature computation store latency (per instrument)",
    ["instrument", "stage"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)


class FeatureComputationProtocol(Protocol):
    """
    Protocol for feature computation operations.

    This protocol defines the contract for feature computation components,
    enabling duck typing for testing and alternative implementations.

    Methods
    -------
    compute_features
        Compute and store features for newly collected data.

    """

    def compute_features(
        self,
        config: SchedulerConfig,
        catalog: ParquetDataCatalog,
        feature_engineer: FeatureEngineer | None,
        feature_store: Any | None,
        init_feature_store_fn: Callable[[], Any | None],
        get_previous_day_fn: Callable[[], datetime],
    ) -> tuple[int, list[str]]:
        """
        Compute features for newly collected data.

        Args:
            config: Scheduler configuration with feature store settings.
            catalog: Parquet catalog for querying bars data.
            feature_engineer: Feature engineer for computing features.
            feature_store: Existing feature store instance (may be None).
            init_feature_store_fn: Callable to lazily initialize feature store.
            get_previous_day_fn: Callable to get the previous trading day.

        Returns:
            Tuple of (total_features_computed, failed_instruments).

        """
        ...


class FeatureComputationComponent:
    """
    Component for feature computation logic extracted from DataScheduler.

    This component handles computing features for newly collected data:
    - Validates configuration and dependencies before computing
    - Lazy initializes FeatureStore if not already available
    - Maps venue codes to Nautilus format (e.g., XNAS -> NASDAQ)
    - Queries catalog for bars data within the target date range
    - Calls FeatureStore.compute_and_store_historical() for each instrument
    - Records Prometheus metrics for observability

    All operations record Prometheus metrics for monitoring and emit
    appropriate warnings/errors without raising exceptions that would
    prevent processing of remaining instruments.

    Example:
        >>> from ml.data.common.feature_computation import FeatureComputationComponent
        >>> component = FeatureComputationComponent()
        >>> total, failed = component.compute_features(
        ...     config=scheduler_config,
        ...     catalog=catalog,
        ...     feature_engineer=engineer,
        ...     feature_store=None,
        ...     init_feature_store_fn=init_fn,
        ...     get_previous_day_fn=get_prev_day,
        ... )
        >>> print(f"Computed {total} features, {len(failed)} failures")

    """

    def compute_features(
        self,
        config: SchedulerConfig,
        catalog: ParquetDataCatalog,
        feature_engineer: FeatureEngineer | None,
        feature_store: Any | None,
        init_feature_store_fn: Callable[[], Any | None],
        get_previous_day_fn: Callable[[], datetime],
    ) -> tuple[int, list[str]]:
        """
        Compute features for newly collected data.

        Uses the configured feature engineer to compute features for ML models.
        This method:
        1. Validates config and dependencies
        2. Lazily initializes FeatureStore if needed
        3. Queries the catalog for recent bars data
        4. Computes and stores features using FeatureStore.compute_and_store_historical()

        Args:
            config: Scheduler configuration with feature_store_enabled flag
                and symbols list.
            catalog: Parquet catalog for querying bars data.
            feature_engineer: Feature engineer for computing features.
                If None, feature computation is skipped.
            feature_store: Existing feature store instance. If None,
                init_feature_store_fn will be called to initialize.
            init_feature_store_fn: Callable to lazily initialize feature store
                when feature_store is None.
            get_previous_day_fn: Callable to get the previous trading day
                for date range calculation.

        Returns:
            Tuple of (total_features_computed, failed_instruments).
            total_features_computed: Number of feature rows stored.
            failed_instruments: List of instrument strings that failed.

        Example:
            >>> component = FeatureComputationComponent()
            >>> total, failed = component.compute_features(
            ...     config=SchedulerConfig(feature_store_enabled=True),
            ...     catalog=catalog,
            ...     feature_engineer=engineer,
            ...     feature_store=store,
            ...     init_feature_store_fn=lambda: None,
            ...     get_previous_day_fn=lambda: datetime.now(),
            ... )
            >>> assert isinstance(total, int)
            >>> assert isinstance(failed, list)

        """
        # Check if feature computation is enabled and configured
        if not config.feature_store_enabled:
            logger.debug(
                "Feature store disabled in configuration, skipping feature computation"
            )
            return 0, []

        if feature_engineer is None:
            logger.debug("No feature engineer configured, skipping feature computation")
            return 0, []

        # Lazy initialize feature store if needed
        current_store = feature_store
        if current_store is None:
            logger.warning("Feature store not initialized, attempting to initialize now")
            current_store = init_feature_store_fn()
            if current_store is None:
                logger.error(
                    "Failed to initialize feature store, skipping feature computation"
                )
                return 0, []

        logger.info("Starting feature computation for new data...")

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
        active_feature_tasks.set(len(config.symbols))

        try:
            # Get date range for feature computation
            # Process previous trading day's data
            target_date = get_previous_day_fn()
            start_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = target_date.replace(
                hour=23, minute=59, second=59, microsecond=999999
            )

            logger.info(
                f"Computing features for date range: {start_date.date()} to {end_date.date()}",
            )

            # Process each configured symbol
            for idx, symbol in enumerate(config.symbols):
                # Update active tasks
                active_feature_tasks.set(len(config.symbols) - idx)
                try:
                    # Parse symbol to get instrument_id
                    symbol_parts = symbol.split(".")
                    if len(symbol_parts) != 2:
                        logger.warning(f"Invalid symbol format: {symbol}, skipping")
                        failed_instruments.append(symbol)
                        continue

                    symbol_code, venue = symbol_parts

                    # Map common venue codes to Nautilus format
                    nautilus_venue = VENUE_MAP.get(venue, venue)
                    instrument_id = InstrumentId.from_str(
                        f"{symbol_code}.{nautilus_venue}"
                    )

                    logger.debug(f"Processing features for {instrument_id}")

                    # Query bars from catalog
                    # Using the catalog's query method with proper parameters
                    # Convert datetime to timestamp (nanoseconds since epoch) for catalog
                    bars_data = catalog.query(
                        data_cls=Bar,
                        identifiers=[str(instrument_id)],
                        start=int(start_date.timestamp() * 1e9),
                        end=int(end_date.timestamp() * 1e9),
                    )

                    if not bars_data:
                        logger.warning(
                            f"No bars found for {instrument_id} on {target_date.date()}"
                        )
                        continue

                    logger.info(f"Found {len(bars_data)} bars for {instrument_id}")

                    # Store features in FeatureStore for future training
                    # Using the FeatureStore's compute_and_store_historical method
                    store_start_time = time.perf_counter()
                    try:
                        stored_count = current_store.compute_and_store_historical(
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
                        feature_computation_store_latency.labels(
                            instrument=str(instrument_id),
                            stage="store",
                        ).observe(store_duration)

                        total_features_computed += stored_count
                        logger.info(
                            f"Stored {stored_count} feature rows for "
                            f"{instrument_id} in FeatureStore",
                        )
                        # Note: FEATURE_COMPUTED events are emitted by FeatureStore
                        # itself to avoid double-counting in metrics
                    except Exception:
                        feature_store_operations_total.labels(
                            operation="store_historical",
                            status="failure",
                        ).inc()
                        logger.error(
                            "Failed to store features for %s",
                            instrument_id,
                            exc_info=True,
                        )
                        failed_instruments.append(str(instrument_id))
                        continue

                except Exception:
                    logger.error(
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
            logger.info(
                f"Feature computation completed in {elapsed_time:.2f}s: "
                f"{total_features_computed} features computed, "
                f"{len(failed_instruments)} failures",
            )

            if failed_instruments:
                logger.warning(f"Failed instruments: {', '.join(failed_instruments)}")

            # Log performance metrics for monitoring
            if total_features_computed > 0:
                avg_time_per_feature = elapsed_time / total_features_computed
                logger.info(
                    f"Average computation time: {avg_time_per_feature*1000:.2f}ms "
                    "per feature row",
                )

            return total_features_computed, failed_instruments

        except Exception:
            logger.error(
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


__all__ = [
    "VENUE_MAP",
    "FeatureComputationComponent",
    "FeatureComputationProtocol",
]
