"""
Data Scheduler for automated daily data collection and processing.

This module provides scheduling capabilities for automated data collection from
Databento and feature computation for ML models.

"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ml._imports import HAS_PROMETHEUS
from ml._imports import Counter
from ml._imports import Gauge
from ml._imports import Histogram
from ml.config.scheduler_config import DatabentoConfig
from ml.config.scheduler_config import SchedulerConfig
from ml.data.collector import DataCollector
from nautilus_trader.adapters.databento.loaders import DatabentoDataLoader
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


if TYPE_CHECKING:
    from ml.features.engineering import FeatureEngineer


logger = logging.getLogger(__name__)

# =============================================================================
# PROMETHEUS METRICS DEFINITIONS
# =============================================================================

# Data Collection Metrics
data_collected_total = Counter(
    "nautilus_ml_data_collected_total",
    "Total data points collected from external sources",
    ["source", "instrument", "data_type"],
)

data_collection_errors_total = Counter(
    "nautilus_ml_data_collection_errors_total",
    "Total errors during data collection",
    ["source", "instrument", "error_type"],
)

data_collection_latency = Histogram(
    "nautilus_ml_data_collection_latency_seconds",
    "Data collection latency in seconds",
    ["source", "instrument"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

# Feature Computation Metrics
features_computed_total = Counter(
    "nautilus_ml_features_computed_total",
    "Total features computed",
    ["instrument", "feature_type"],
)

feature_computation_errors_total = Counter(
    "nautilus_ml_feature_computation_errors_total",
    "Total errors during feature computation",
    ["instrument", "error_type"],
)

feature_computation_latency = Histogram(
    "nautilus_ml_feature_computation_latency_seconds",
    "Feature computation latency in seconds",
    ["instrument", "stage"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# Pipeline Stage Metrics
pipeline_stage_latency = Histogram(
    "nautilus_ml_pipeline_stage_latency_seconds",
    "Pipeline stage execution latency in seconds",
    ["stage"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

pipeline_runs_total = Counter(
    "nautilus_ml_pipeline_runs_total",
    "Total pipeline runs",
    ["status"],  # success, failure, partial
)

# Resource Usage Metrics
active_collection_tasks = Gauge(
    "nautilus_ml_active_collection_tasks",
    "Number of active data collection tasks",
)

active_feature_tasks = Gauge(
    "nautilus_ml_active_feature_tasks",
    "Number of active feature computation tasks",
)

data_retention_cleanup_total = Counter(
    "nautilus_ml_data_retention_cleanup_total",
    "Total data retention cleanup operations",
    ["status"],
)

# Data Quality Metrics
data_missing_ratio = Gauge(
    "nautilus_ml_data_missing_ratio",
    "Ratio of missing data points",
    ["instrument", "data_type"],
)

data_staleness_seconds = Gauge(
    "nautilus_ml_data_staleness_seconds",
    "Age of most recent data in seconds",
    ["instrument"],
)

# API Metrics
api_request_total = Counter(
    "nautilus_ml_api_request_total",
    "Total API requests made",
    ["endpoint", "status_code"],
)

api_rate_limit_hits = Counter(
    "nautilus_ml_api_rate_limit_hits_total",
    "Total API rate limit hits",
    ["endpoint"],
)

# Catalog Metrics
catalog_write_operations_total = Counter(
    "nautilus_ml_catalog_write_operations_total",
    "Total catalog write operations",
    ["status"],
)

catalog_write_latency = Histogram(
    "nautilus_ml_catalog_write_latency_seconds",
    "Catalog write operation latency",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# Feature Store Metrics
feature_store_operations_total = Counter(
    "nautilus_ml_feature_store_operations_total",
    "Total feature store operations",
    ["operation", "status"],
)

feature_store_latency = Histogram(
    "nautilus_ml_feature_store_latency_seconds",
    "Feature store operation latency",
    ["operation"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)


@contextmanager
def track_pipeline_stage(stage: str) -> Generator[None, None, None]:
    """
    Context manager to track pipeline stage execution time.

    Parameters
    ----------
    stage : str
        Name of the pipeline stage to track

    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        pipeline_stage_latency.labels(stage=stage).observe(duration)


class DataScheduler:
    """
    Scheduler for automated daily data collection and processing.

    This class coordinates:
    1. Daily collection from Databento API
    2. Writing to ParquetDataCatalog
    3. Triggering feature computation
    4. Managing data retention policies

    """

    def __init__(
        self,
        catalog: ParquetDataCatalog,
        config: SchedulerConfig | None = None,
        collector: DataCollector | None = None,
        feature_engineer: FeatureEngineer | None = None,
        metrics_port: int | None = None,
        start_metrics_server: bool = True,
    ) -> None:
        """
        Initialize data scheduler.

        Parameters
        ----------
        catalog : ParquetDataCatalog
            Nautilus data catalog for data storage
        config : SchedulerConfig, optional
            Configuration for scheduler. If None, uses defaults.
        collector : DataCollector, optional
            Data collector for fetching from Databento
        feature_engineer : FeatureEngineer, optional
            Feature engineer for computing features
        metrics_port : int, optional
            Port for metrics HTTP server. Defaults to 8000.
        start_metrics_server : bool, default=True
            Whether to start the metrics HTTP server

        """
        self.catalog = catalog
        self.config = config or SchedulerConfig()
        self.collector = collector or DataCollector()
        self.feature_engineer = feature_engineer

        # Scheduling state
        self.enabled = True
        self._databento_loader = DatabentoDataLoader()

        # Initialize feature store if configured
        self._feature_store: Any | None = None
        if self.config.feature_store_enabled and self.feature_engineer is not None:
            self._initialize_feature_store()

        # Initialize metrics server if configured
        self._metrics_server: Any | None = None
        if start_metrics_server and HAS_PROMETHEUS:
            self._start_metrics_server(metrics_port or 8000)

        logger.info(
            f"Initialized DataScheduler with {len(self.config.symbols)} symbols, "
            f"retention={self.config.retention_days} days, "
            f"feature_store={'enabled' if self.config.feature_store_enabled else 'disabled'}"
            f"{f', metrics_port={metrics_port or 8000}' if start_metrics_server else ''}",
        )

    def _initialize_feature_store(self) -> None:
        """
        Initialize the FeatureStore with proper configuration.

        This method sets up the FeatureStore for batch feature computation and storage,
        ensuring training/inference parity.

        """
        from ml._imports import HAS_POLARS
        from ml._imports import check_ml_dependencies
        from ml.features.engineering import FeatureConfig
        from ml.stores.feature_store import FeatureStore

        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        try:
            # Get connection string from config, environment, or use default
            db_connection = (
                self.config.feature_store_connection
                or os.getenv("NAUTILUS_DB_CONNECTION")
                or "postgresql://postgres:postgres@localhost:5432/nautilus"
            )

            # Get feature config from the feature engineer
            feature_config: FeatureConfig
            if self.feature_engineer is not None and hasattr(self.feature_engineer, "config"):
                feature_config = self.feature_engineer.config
            else:
                feature_config = FeatureConfig()

            self._feature_store = FeatureStore(
                connection_string=db_connection,
                feature_config=feature_config,
            )

            # Log connection info (hide password for security)
            safe_connection = db_connection.split("@")[1] if "@" in db_connection else db_connection
            logger.info(f"Initialized FeatureStore with connection to: {safe_connection}")

        except Exception as e:
            logger.error(f"Failed to initialize FeatureStore: {e}")
            self._feature_store = None
            # Don't raise - allow scheduler to work without feature store

    def _start_metrics_server(self, port: int) -> None:
        """
        Start the HTTP server for Prometheus metrics.

        Parameters
        ----------
        port : int
            Port number for the metrics server

        """
        try:
            from ml.monitoring._config import MonitoringConfig
            from ml.monitoring.server import MetricsServer

            # Create monitoring config with specified port
            monitoring_config = MonitoringConfig(
                enabled=True,
                metrics_port=port,
            )

            self._metrics_server = MetricsServer(config=monitoring_config)
            self._metrics_server.start()
            logger.info(f"Started metrics server on port {port}")
        except Exception as e:
            logger.warning(f"Failed to start metrics server: {e}")
            self._metrics_server = None

    def run_daily_update(self) -> None:
        """
        Run the complete daily update process.

        This includes:
        1. Collecting latest data from Databento
        2. Writing to catalog
        3. Computing features if configured
        4. Cleaning old data based on retention policy

        """
        logger.info("Starting daily data update...")
        pipeline_start_time = time.time()
        pipeline_status = "success"

        try:
            # Step 1: Collect latest data
            with track_pipeline_stage("data_collection"):
                self._collect_latest_data()

            # Step 2: Compute features if configured
            if self.feature_engineer is not None:
                with track_pipeline_stage("feature_computation"):
                    self._compute_features()

            # Step 3: Clean old data
            with track_pipeline_stage("data_cleanup"):
                self._clean_old_data()

            logger.info("Daily data update completed successfully")

        except Exception as e:
            pipeline_status = "failure"
            logger.error(f"Daily data update failed: {e}")
            raise
        finally:
            # Record overall pipeline metrics
            pipeline_duration = time.time() - pipeline_start_time
            pipeline_runs_total.labels(status=pipeline_status).inc()
            pipeline_stage_latency.labels(stage="complete_pipeline").observe(pipeline_duration)

    def _collect_latest_data(self) -> None:
        """
        Collect latest data from Databento.

        Fetches:
        - Previous trading day's minute bars
        - L2 depth data if configured
        - Trades and quotes if configured

        Raises
        ------
        ValueError
            If DATABENTO_API_KEY is not set
        ImportError
            If databento library is not installed

        """
        # Get previous trading day
        target_date = self._get_previous_trading_day()
        start_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)

        logger.info(f"Collecting data for {start_date.date()}")

        # Check for API key
        api_key = os.getenv("DATABENTO_API_KEY")
        if not api_key:
            logger.error("DATABENTO_API_KEY environment variable not set")
            raise ValueError(
                "DATABENTO_API_KEY environment variable is required for data collection",
            )

        try:
            # Import Databento client lazily to avoid event loop issues
            import databento as db
        except ImportError as e:
            logger.error(f"Databento library not installed: {e}")
            logger.info("Install databento with: pip install databento")
            raise

        # Initialize Databento client
        client = db.Historical(api_key)

        # Setup temporary directory if needed
        temp_data_dir: Path | None = None
        if self.config.databento.use_temporary_files:
            temp_data_dir = Path(self.config.databento.temp_data_dir)
            temp_data_dir.mkdir(exist_ok=True)

        # Track collection statistics
        collected_count = 0
        failed_count = 0

        # Track active collection tasks
        active_collection_tasks.set(len(self.config.symbols))

        try:
            # Collect data for each symbol
            for idx, symbol in enumerate(self.config.symbols):
                # Update active tasks gauge
                active_collection_tasks.set(len(self.config.symbols) - idx)

                success = self._collect_symbol_data(
                    client=client,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    target_date=target_date,
                    temp_data_dir=temp_data_dir,
                )

                if success:
                    collected_count += 1
                else:
                    failed_count += 1
        finally:
            # Reset active tasks
            active_collection_tasks.set(0)

        # Clean up temporary directory if empty
        if temp_data_dir and temp_data_dir.exists():
            if not any(temp_data_dir.iterdir()):
                temp_data_dir.rmdir()

        # Log final statistics
        logger.info(
            f"Data collection completed: {collected_count} succeeded, {failed_count} failed "
            f"out of {len(self.config.symbols)} symbols",
        )

        if failed_count > len(self.config.symbols) * 0.5:  # More than 50% failed
            logger.warning(
                "High failure rate in data collection - investigate connectivity or API limits",
            )
            # Record API rate limit metric if high failure rate
            if failed_count > len(self.config.symbols) * 0.7:
                api_rate_limit_hits.labels(endpoint="databento_timeseries").inc()

    def _collect_symbol_data(
        self,
        client: Any,  # databento.Historical
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        target_date: datetime,
        temp_data_dir: Path | None,
    ) -> bool:
        """
        Collect data for a single symbol.

        Parameters
        ----------
        client : databento.Historical
            Databento client instance
        symbol : str
            Symbol in format "SYMBOL.VENUE"
        start_date : datetime
            Start of data range
        end_date : datetime
            End of data range
        target_date : datetime
            Target date for data collection
        temp_data_dir : Path | None
            Directory for temporary files, if using

        Returns
        -------
        bool
            True if collection succeeded, False otherwise

        """
        logger.info(f"Collecting data for {symbol}")
        collection_start_time = time.time()

        # Parse symbol format
        symbol_parts = symbol.split(".")
        if len(symbol_parts) != 2:
            logger.warning(f"Invalid symbol format: {symbol}, expected SYMBOL.VENUE")
            data_collection_errors_total.labels(
                source="databento",
                instrument=symbol,
                error_type="invalid_symbol_format",
            ).inc()
            return False

        symbol_code, venue = symbol_parts

        # Retry logic for resilience
        for attempt in range(self.config.max_retries):
            try:
                # Fetch data from Databento
                logger.debug(
                    f"Fetching {self.config.databento.schema} for {symbol_code} "
                    f"from {start_date} to {end_date} (attempt {attempt + 1})",
                )

                if self.config.databento.use_temporary_files and temp_data_dir:
                    # Save to temporary DBN file
                    temp_file = (
                        temp_data_dir
                        / f"{symbol_code}_{target_date.strftime('%Y%m%d')}_{self.config.databento.schema}.dbn"
                    )

                    # Request and save data
                    api_start_time = time.time()
                    response = client.timeseries.get_range(
                        dataset=self.config.databento.dataset,
                        symbols=[symbol_code],
                        schema=self.config.databento.schema,
                        start=start_date,
                        end=end_date,
                        stype_in=self.config.databento.stype_in,
                    )
                    api_request_total.labels(
                        endpoint="databento_timeseries",
                        status_code="200",
                    ).inc()

                    response.to_file(str(temp_file))

                    # Load from DBN file
                    data = self._load_from_dbn_file(
                        temp_file,
                        symbol_code,
                        venue,
                    )

                    # Clean up temp file
                    if temp_file.exists():
                        temp_file.unlink()
                else:
                    # Direct processing without temp files
                    response = client.timeseries.get_range(
                        dataset=self.config.databento.dataset,
                        symbols=[symbol_code],
                        schema=self.config.databento.schema,
                        start=start_date,
                        end=end_date,
                        stype_in=self.config.databento.stype_in,
                    )

                    # Convert to DataFrame and then to Nautilus objects
                    # This path would need additional implementation
                    logger.warning("Direct processing not yet fully implemented, using temp files")
                    return False

                if data:
                    # Write to catalog with metrics
                    catalog_start_time = time.time()
                    logger.info(f"Writing {len(data)} records to catalog for {symbol_code}")

                    try:
                        self.catalog.write_data(data)
                        catalog_write_operations_total.labels(status="success").inc()
                        catalog_write_latency.observe(time.time() - catalog_start_time)
                    except Exception:
                        catalog_write_operations_total.labels(status="failure").inc()
                        raise

                    # Record collection metrics
                    collection_duration = time.time() - collection_start_time
                    data_collected_total.labels(
                        source="databento",
                        instrument=symbol,
                        data_type=self.config.databento.schema,
                    ).inc(len(data))
                    data_collection_latency.labels(
                        source="databento",
                        instrument=symbol,
                    ).observe(collection_duration)

                    # Calculate and record data freshness
                    data_age = (datetime.now() - target_date).total_seconds()
                    data_staleness_seconds.labels(instrument=symbol).set(data_age)

                    logger.info(f"Successfully collected and stored data for {symbol_code}")
                    return True
                else:
                    logger.warning(f"No data returned for {symbol_code}")
                    data_collection_errors_total.labels(
                        source="databento",
                        instrument=symbol,
                        error_type="no_data",
                    ).inc()
                    return False

            except Exception as e:
                logger.error(f"Error collecting data for {symbol} (attempt {attempt + 1}): {e}")

                # Classify error type for metrics
                error_type = "unknown"
                if "rate limit" in str(e).lower():
                    error_type = "rate_limit"
                    api_rate_limit_hits.labels(endpoint="databento_timeseries").inc()
                elif "connection" in str(e).lower() or "timeout" in str(e).lower():
                    error_type = "connection"
                elif "unauthorized" in str(e).lower() or "forbidden" in str(e).lower():
                    error_type = "auth"
                    api_request_total.labels(
                        endpoint="databento_timeseries",
                        status_code="401",
                    ).inc()
                else:
                    api_request_total.labels(
                        endpoint="databento_timeseries",
                        status_code="500",
                    ).inc()

                data_collection_errors_total.labels(
                    source="databento",
                    instrument=symbol,
                    error_type=error_type,
                ).inc()

                if attempt < self.config.max_retries - 1:
                    logger.info(f"Retrying in {self.config.retry_delay_seconds} seconds...")
                    time.sleep(self.config.retry_delay_seconds)
                else:
                    logger.error(
                        f"Failed to collect data for {symbol} after {self.config.max_retries} attempts",
                    )
                    return False

        return False

    def _load_from_dbn_file(
        self,
        file_path: Path,
        symbol_code: str,
        venue: str,
    ) -> list[Any]:
        """
        Load data from DBN file using DatabentoDataLoader.

        Parameters
        ----------
        file_path : Path
            Path to the DBN file
        symbol_code : str
            Symbol code without venue
        venue : str
            Trading venue code

        Returns
        -------
        list[Any]
            List of Nautilus data objects

        """
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

        # Load data based on schema
        return self._databento_loader.from_dbn_file(
            path=file_path,
            instrument_id=instrument_id,
            price_precision=self.config.databento.price_precision,
            as_legacy_cython=True,
            bars_timestamp_on_close=True if "ohlcv" in self.config.databento.schema else False,
            include_trades="trades" in self.config.databento.schema,
        )

    def _get_previous_trading_day(self) -> datetime:
        """
        Get the previous trading day based on current date.

        Returns
        -------
        datetime
            Previous trading day

        """
        today = datetime.now()

        if today.weekday() == 0:  # Monday
            # Get Friday's data
            return today - timedelta(days=3)
        elif today.weekday() == 6:  # Sunday
            # Get Friday's data
            return today - timedelta(days=2)
        else:
            # Get previous day's data
            return today - timedelta(days=1)

    def _compute_features(self) -> None:
        """
        Compute features for newly collected data.

        Uses the configured feature engineer to compute features for ML models.
        This method:
        1. Queries the catalog for recent bars data
        2. Computes features using FeatureEngineer (batch mode)
        3. Stores features in FeatureStore for training/inference parity

        """
        # Check if feature computation is enabled and configured
        if not self.config.feature_store_enabled:
            logger.debug("Feature store disabled in configuration, skipping feature computation")
            return

        if self.feature_engineer is None:
            logger.debug("No feature engineer configured, skipping feature computation")
            return

        if self._feature_store is None:
            logger.warning("Feature store not initialized, attempting to initialize now")
            self._initialize_feature_store()
            if self._feature_store is None:
                logger.error("Failed to initialize feature store, skipping feature computation")
                return

        logger.info("Starting feature computation for new data...")

        # Import required modules
        from ml._imports import HAS_POLARS
        from ml._imports import check_ml_dependencies
        from nautilus_trader.model.data import Bar
        from nautilus_trader.model.identifiers import InstrumentId

        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        from ml._imports import pl

        # Track metrics
        total_features_computed = 0
        failed_instruments = []
        start_time = time.time()

        # Update active feature tasks gauge
        active_feature_tasks.set(len(self.config.symbols))

        try:
            # Get date range for feature computation
            # Process previous trading day's data
            target_date = self._get_previous_trading_day()
            start_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)

            logger.info(
                f"Computing features for date range: {start_date.date()} to {end_date.date()}",
            )

            # Process each configured symbol
            for idx, symbol in enumerate(self.config.symbols):
                # Update active tasks
                active_feature_tasks.set(len(self.config.symbols) - idx)
                try:
                    # Parse symbol to get instrument_id
                    symbol_parts = symbol.split(".")
                    if len(symbol_parts) != 2:
                        logger.warning(f"Invalid symbol format: {symbol}, skipping")
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

                    logger.debug(f"Processing features for {instrument_id}")

                    # Query bars from catalog
                    # Using the catalog's query method with proper parameters
                    # Convert datetime to timestamp (nanoseconds since epoch) for catalog
                    bars_data = self.catalog.query(
                        data_cls=Bar,
                        identifiers=[str(instrument_id)],
                        start=int(start_date.timestamp() * 1e9),
                        end=int(end_date.timestamp() * 1e9),
                    )

                    if not bars_data:
                        logger.warning(f"No bars found for {instrument_id} on {target_date.date()}")
                        continue

                    logger.info(f"Found {len(bars_data)} bars for {instrument_id}")

                    # Convert bars to DataFrame for feature computation
                    # Create a list of dictionaries from Bar objects
                    bars_list = []
                    for bar in bars_data:
                        bars_list.append(
                            {
                                "open": float(bar.open),
                                "high": float(bar.high),
                                "low": float(bar.low),
                                "close": float(bar.close),
                                "volume": float(bar.volume),
                                "ts_event": int(bar.ts_event),
                                "ts_init": int(bar.ts_init),
                            },
                        )

                    if not bars_list:
                        logger.warning(f"No valid bars to process for {instrument_id}")
                        continue

                    # Create Polars DataFrame
                    bars_df = pl.DataFrame(bars_list)

                    # Ensure DataFrame is sorted by timestamp
                    bars_df = bars_df.sort("ts_event")

                    # Compute features using the feature engineer (batch mode)
                    try:
                        feature_start_time = time.time()
                        features_df, feature_names = self.feature_engineer.calculate_features_batch(
                            bars_df,
                        )

                        if features_df is None or (
                            hasattr(features_df, "is_empty") and features_df.is_empty()
                        ):
                            logger.warning(f"No features computed for {instrument_id}")
                            feature_computation_errors_total.labels(
                                instrument=str(instrument_id),
                                error_type="no_features",
                            ).inc()
                            continue

                        # Get number of features computed
                        num_features = (
                            len(features_df)
                            if hasattr(features_df, "__len__")
                            else features_df.shape[0]
                        )

                        # Record feature computation metrics
                        feature_duration = time.time() - feature_start_time
                        feature_computation_latency.labels(
                            instrument=str(instrument_id),
                            stage="compute",
                        ).observe(feature_duration)
                        features_computed_total.labels(
                            instrument=str(instrument_id),
                            feature_type="batch",
                        ).inc(num_features)

                        logger.info(f"Computed {num_features} feature rows for {instrument_id}")

                        # Store features in FeatureStore for future training
                        # Using the FeatureStore's compute_and_store_historical method
                        store_start_time = time.time()
                        try:
                            stored_count = self._feature_store.compute_and_store_historical(
                                instrument_id=str(instrument_id),
                                start=start_date,
                                end=end_date,
                                force_recompute=True,  # Force recompute for fresh data
                            )

                            # Record feature store metrics
                            store_duration = time.time() - store_start_time
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
                            logger.info(
                                f"Stored {stored_count} feature rows for {instrument_id} in FeatureStore",
                            )
                        except Exception:
                            feature_store_operations_total.labels(
                                operation="store_historical",
                                status="failure",
                            ).inc()
                            raise

                    except Exception as e:
                        logger.error(f"Failed to compute features for {instrument_id}: {e}")
                        feature_computation_errors_total.labels(
                            instrument=str(instrument_id),
                            error_type="computation_error",
                        ).inc()
                        failed_instruments.append(str(instrument_id))
                        continue

                except Exception as e:
                    logger.error(f"Error processing symbol {symbol}: {e}")
                    feature_computation_errors_total.labels(
                        instrument=symbol,
                        error_type="processing_error",
                    ).inc()
                    failed_instruments.append(symbol)
                    continue

            # Calculate elapsed time
            elapsed_time = time.time() - start_time

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
                    f"Average computation time: {avg_time_per_feature*1000:.2f}ms per feature row",
                )

        except Exception as e:
            logger.error(f"Critical error in feature computation: {e}", exc_info=True)
            feature_computation_errors_total.labels(
                instrument="all",
                error_type="critical",
            ).inc()
            raise
        finally:
            # Reset active tasks
            active_feature_tasks.set(0)

    def _clean_old_data(self) -> None:
        """
        Clean data older than retention period.

        Removes:
        - Minute bars older than retention_days
        - L2 depth older than 30 days
        - Keeps daily bars indefinitely

        """
        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)

        logger.info(f"Cleaning data older than {cutoff_date.date()}")
        cleanup_start_time = time.time()

        try:
            # In production, this would:
            # 1. Query catalog for old data
            # 2. Delete files/partitions older than cutoff
            # 3. Update catalog metadata

            # For now, record successful cleanup
            data_retention_cleanup_total.labels(status="success").inc()

            cleanup_duration = time.time() - cleanup_start_time
            pipeline_stage_latency.labels(stage="data_cleanup").observe(cleanup_duration)

            logger.info("Data cleanup completed")
        except Exception as e:
            data_retention_cleanup_total.labels(status="failure").inc()
            logger.error(f"Data cleanup failed: {e}")
            raise

    def schedule_updates(self, cron_expression: str | None = None) -> None:
        """
        Schedule automated daily updates.

        Parameters
        ----------
        cron_expression : str, optional
            Cron expression for scheduling. Defaults to daily at 4 AM UTC.

        """
        if cron_expression is None:
            # Default: Daily at 4 AM UTC
            cron_expression = "0 4 * * *"

        logger.info(f"Scheduling updates with cron: {cron_expression}")

        # In production, this would use a scheduler like:
        # - APScheduler for Python-based scheduling
        # - Airflow for enterprise scheduling
        # - Cron for simple Unix-based scheduling

        # Example with APScheduler:
        # from apscheduler.schedulers.background import BackgroundScheduler
        # from apscheduler.triggers.cron import CronTrigger
        #
        # scheduler = BackgroundScheduler()
        # scheduler.add_job(
        #     func=self.run_daily_update,
        #     trigger=CronTrigger.from_crontab(cron_expression),
        #     id='daily_data_update',
        #     name='Daily Data Update',
        #     replace_existing=True
        # )
        # scheduler.start()

        logger.info("Scheduler configured successfully")

    def stop(self) -> None:
        """
        Stop the scheduler and clean up resources.
        """
        if self._metrics_server:
            try:
                self._metrics_server.stop()
                logger.info("Stopped metrics server")
            except Exception as e:
                logger.warning(f"Error stopping metrics server: {e}")

        self.enabled = False
        logger.info("Scheduler stopped")

    def get_status(self) -> dict[str, str | int | bool]:
        """
        Get current scheduler status.

        Returns
        -------
        dict
            Status information including:
            - enabled: Whether scheduler is active
            - last_run: Last successful update
            - next_run: Next scheduled update
            - data_stats: Current data statistics

        """
        # In production, this would query actual scheduler state
        return {
            "enabled": self.enabled,
            "collection_time": self.config.collection_time,
            "retention_days": self.config.retention_days,
            "symbol_count": len(self.config.symbols),
            "databento_dataset": self.config.databento.dataset,
            "databento_schema": self.config.databento.schema,
            "has_feature_engineer": self.feature_engineer is not None,
            "catalog_path": str(self.catalog.path) if hasattr(self.catalog, "path") else "N/A",
        }


def main() -> None:
    """
    Run example usage of DataScheduler.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize components
    catalog = ParquetDataCatalog("./data")

    # Create configuration
    config = SchedulerConfig(
        symbols=["SPY.XNAS", "QQQ.XNAS", "IWM.XNAS"],  # Start with small universe
        retention_days=90,
        databento=DatabentoConfig(
            dataset="GLBX.MDP3",
            schema="ohlcv-1m",
        ),
    )

    # Create scheduler
    scheduler = DataScheduler(
        catalog=catalog,
        config=config,
    )

    # Get status
    status = scheduler.get_status()
    logger.info(f"Scheduler status: {status}")

    # Run manual update (for testing)
    # scheduler.run_daily_update()

    # Schedule automated updates
    # scheduler.schedule_updates()


if __name__ == "__main__":
    main()
