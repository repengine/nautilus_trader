"""
DataSchedulerFacade that delegates to extracted components.

This facade provides the EXACT same public API as the legacy DataScheduler
while internally delegating to focused, single-responsibility components.

The facade pattern enables:
- Gradual migration from monolithic to component-based architecture
- Feature flag switching between legacy and facade implementations
- Testing of components in isolation while preserving integration behavior

"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from ml._imports import HAS_PROMETHEUS
from ml.config.scheduler_config import SchedulerConfig
from ml.data.collector import DataCollector
from ml.data.common import DailyUpdateOrchestratorComponent
from ml.data.common import DataCleanupComponent
from ml.data.common import DataCollectionComponent
from ml.data.common import DatasetRegistrationComponent
from ml.data.common import FeatureComputationComponent
from ml.data.common import MetricsServerComponent
from ml.data.common import OrchestratorCollectionComponent
from ml.data.common import SchedulerInitComponent
from nautilus_trader.adapters.databento.loaders import DatabentoDataLoader
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


if TYPE_CHECKING:
    from ml.features.engineering import FeatureEngineer
    from ml.registry.protocols import RegistryProtocol


logger = logging.getLogger(__name__)


def use_legacy_scheduler() -> bool:
    """
    Return True if ML_USE_LEGACY_SCHEDULER=1.

    This feature flag controls whether the legacy DataScheduler or the
    component-based DataSchedulerFacade is used.

    Returns:
        True if the environment variable ML_USE_LEGACY_SCHEDULER is set to "1",
        False otherwise.

    Example:
        >>> import os
        >>> os.environ["ML_USE_LEGACY_SCHEDULER"] = "1"
        >>> use_legacy_scheduler()
        True
        >>> os.environ["ML_USE_LEGACY_SCHEDULER"] = "0"
        >>> use_legacy_scheduler()
        False

    """
    return os.getenv("ML_USE_LEGACY_SCHEDULER", "0") == "1"


def create_data_scheduler(
    catalog: ParquetDataCatalog,
    config: SchedulerConfig | None = None,
    collector: DataCollector | None = None,
    feature_engineer: FeatureEngineer | None = None,
    metrics_port: int | None = None,
    start_metrics_server: bool = True,
    connection: str | None = None,
    use_orchestrator: bool = False,
    dual_write: bool = False,
) -> DataScheduler | DataSchedulerFacade:
    """
    Create scheduler based on feature flag.

    Factory function that returns either the legacy DataScheduler or the
    component-based DataSchedulerFacade depending on the ML_USE_LEGACY_SCHEDULER
    environment variable.

    Args:
        catalog: Nautilus data catalog for data storage.
        config: Configuration for scheduler. If None, uses defaults.
        collector: Data collector for fetching from Databento.
        feature_engineer: Feature engineer for computing features.
        metrics_port: Port for metrics HTTP server. Defaults to 8000.
        start_metrics_server: Whether to start the metrics HTTP server.
        connection: Database connection string for feature store.
        use_orchestrator: Whether to use orchestrator-based collection.
        dual_write: Whether to dual-write to both SQL and catalog.

    Returns:
        DataScheduler if ML_USE_LEGACY_SCHEDULER=1, otherwise DataSchedulerFacade.

    Example:
        >>> from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
        >>> catalog = ParquetDataCatalog("./data")
        >>> scheduler = create_data_scheduler(catalog)
        >>> isinstance(scheduler, (DataScheduler, DataSchedulerFacade))
        True

    """
    if use_legacy_scheduler():
        from ml.data.scheduler import DataScheduler

        return DataScheduler(
            catalog=catalog,
            config=config,
            collector=collector,
            feature_engineer=feature_engineer,
            metrics_port=metrics_port,
            start_metrics_server=start_metrics_server,
            connection=connection,
            use_orchestrator=use_orchestrator,
            dual_write=dual_write,
        )

    return DataSchedulerFacade(
        catalog=catalog,
        config=config,
        collector=collector,
        feature_engineer=feature_engineer,
        metrics_port=metrics_port,
        start_metrics_server=start_metrics_server,
        connection=connection,
        use_orchestrator=use_orchestrator,
        dual_write=dual_write,
    )


# Import at module level for type hints in factory function
# This is a forward reference that gets resolved later
from ml.data.scheduler import DataScheduler  # noqa: E402


class DataSchedulerFacade:
    """
    Facade for automated daily data collection and processing using components.

    This class provides the EXACT same public API as DataScheduler but internally
    delegates to focused, single-responsibility components. It coordinates:
    1. Daily collection from Databento API
    2. Writing to ParquetDataCatalog
    3. Triggering feature computation
    4. Managing data retention policies

    The facade enables gradual migration from monolithic to component-based
    architecture while maintaining backward compatibility.

    Attributes:
        catalog: ParquetDataCatalog instance for data storage.
        config: SchedulerConfig instance with configuration.
        collector: DataCollector for Databento data fetching.
        feature_engineer: Optional FeatureEngineer for feature computation.
        enabled: Whether the scheduler is currently enabled.
        _databento_loader: DatabentoDataLoader for DBN file loading.
        _current_run_id: Unique ID for current collection run.
        _data_registry: DataRegistry instance for event tracking.
        _feature_store: FeatureStore instance for feature persistence.
        _metrics_server: MetricsServer instance for Prometheus metrics.
        _feature_store_connection: Resolved database connection string.
        _use_orchestrator: Whether to use orchestrator-based collection.
        _dual_write: Whether to dual-write to both SQL and catalog.

    Example:
        >>> from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
        >>> from ml.config.scheduler_config import SchedulerConfig
        >>> catalog = ParquetDataCatalog("./data")
        >>> config = SchedulerConfig(symbols=["SPY.XNAS"])
        >>> scheduler = DataSchedulerFacade(catalog, config)
        >>> status = scheduler.get_status()
        >>> assert status["enabled"] is True

    """

    def __init__(
        self,
        catalog: ParquetDataCatalog,
        config: SchedulerConfig | None = None,
        collector: DataCollector | None = None,
        feature_engineer: FeatureEngineer | None = None,
        metrics_port: int | None = None,
        start_metrics_server: bool = True,
        connection: str | None = None,
        use_orchestrator: bool = False,
        dual_write: bool = False,
    ) -> None:
        """
        Initialize data scheduler facade.

        Args:
            catalog: Nautilus data catalog for data storage.
            config: Configuration for scheduler. If None, uses defaults.
            collector: Data collector for fetching from Databento.
            feature_engineer: Feature engineer for computing features.
            metrics_port: Port for metrics HTTP server. Defaults to 8000.
            start_metrics_server: Whether to start the metrics HTTP server.
            connection: Database connection string for feature store.
            use_orchestrator: Whether to use orchestrator-based collection.
            dual_write: Whether to dual-write to both SQL and catalog.

        """
        # Initialize components
        self._init_component = SchedulerInitComponent()
        self._cleanup_component = DataCleanupComponent()
        self._metrics_component = MetricsServerComponent()
        self._daily_update_component = DailyUpdateOrchestratorComponent()
        self._registration_component = DatasetRegistrationComponent()
        self._feature_computation_component = FeatureComputationComponent()
        self._orchestrator_component = OrchestratorCollectionComponent()
        self._data_collection_component = DataCollectionComponent()

        # Store public attributes (same as legacy)
        self.catalog = catalog
        self.config = config or SchedulerConfig()
        self.collector = collector or DataCollector()
        self.feature_engineer = feature_engineer

        # Unified ingestion flags
        self._use_orchestrator: bool = bool(use_orchestrator)
        self._dual_write: bool = bool(dual_write)

        # Scheduling state
        self.enabled = True
        self._databento_loader = DatabentoDataLoader()
        self._current_run_id: str = ""  # Will be set during collection runs

        # Resolve connection string using component
        self._feature_store_connection: str | None = self._init_component.resolve_connection(
            config=self.config,
            connection=connection,
        )

        # Initialize DataRegistry using component
        self._data_registry: RegistryProtocol | None = self._init_component.init_data_registry(
            connection=self._feature_store_connection,
        )

        # Initialize feature store using component
        self._feature_store: Any | None = None
        if self.config.feature_store_enabled and self.feature_engineer is not None:
            self._feature_store = self._init_component.init_feature_store(
                config=self.config,
                connection=self._feature_store_connection,
                feature_engineer=self.feature_engineer,
            )

        # Initialize metrics server using component
        self._metrics_server: Any | None = None
        if start_metrics_server and HAS_PROMETHEUS:
            self._metrics_server = self._metrics_component.start_metrics_server(
                port=metrics_port or 8000,
            )

        logger.info(
            f"Initialized DataSchedulerFacade with {len(self.config.symbols)} symbols, "
            f"retention={self.config.retention_days} days, "
            f"feature_store={'enabled' if self.config.feature_store_enabled else 'disabled'}"
            f"{f', metrics_port={metrics_port or 8000}' if start_metrics_server else ''}",
        )

    def run_daily_update(self) -> None:
        """
        Run the complete daily update process.

        This includes:
        1. Collecting latest data from Databento
        2. Writing to catalog
        3. Computing features if configured
        4. Cleaning old data based on retention policy

        Delegates to DailyUpdateOrchestratorComponent for orchestration.

        Raises:
            Exception: Re-raised after recording failure metric.

        """
        self._daily_update_component.run_daily_update(
            use_orchestrator=self._use_orchestrator,
            feature_engineer=self.feature_engineer,
            collect_latest_data_fn=self._collect_latest_data,
            collect_via_orchestrator_fn=self._collect_via_orchestrator,
            compute_features_fn=self._compute_features,
            clean_old_data_fn=self._clean_old_data,
        )

    def schedule_updates(self, cron_expression: str | None = None) -> None:
        """
        Schedule automated daily updates.

        Delegates to DataCleanupComponent for scheduling configuration.

        Args:
            cron_expression: Cron expression for scheduling.
                Defaults to "0 4 * * *" (daily at 4 AM UTC).

        """
        self._cleanup_component.schedule_updates(cron_expression=cron_expression)

    def stop(self) -> None:
        """
        Stop the scheduler and clean up resources.

        Delegates to DataCleanupComponent for graceful shutdown.

        """
        self._cleanup_component.stop(metrics_server=self._metrics_server)
        self.enabled = False

    def get_status(self) -> dict[str, str | int | bool]:
        """
        Get current scheduler status.

        Delegates to DataCleanupComponent for status retrieval.

        Returns:
            Status information including:
            - enabled: Whether scheduler is active
            - collection_time: Configured collection time
            - retention_days: Data retention period
            - symbol_count: Number of symbols configured
            - databento_dataset: Databento dataset name
            - databento_schema: Databento schema name
            - has_feature_engineer: Whether feature engineer is configured
            - catalog_path: Path to data catalog or "N/A"

        """
        return self._cleanup_component.get_status(
            config=self.config,
            catalog=self.catalog,
            feature_engineer=self.feature_engineer,
            enabled=self.enabled,
        )

    def _get_previous_trading_day(self) -> Any:
        """
        Get the previous trading day based on current date.

        Delegates to DataCleanupComponent.

        Returns:
            Previous trading day datetime.

        """
        return self._cleanup_component.get_previous_trading_day()

    def _ensure_dataset_registered(
        self,
        dataset_id: str,
        dataset_type_label: str,
        location: str,
    ) -> None:
        """
        Ensure a dataset manifest exists in the registry.

        Delegates to DatasetRegistrationComponent.

        Args:
            dataset_id: The dataset identifier (e.g., "ohlcv_spy_xnas").
            dataset_type_label: High-level dataset type label
                ("bars", "trades", "tbbo", "mbp1").
            location: Storage location for the dataset.

        """
        self._registration_component.ensure_dataset_registered(
            registry=self._data_registry,
            dataset_id=dataset_id,
            dataset_type_label=dataset_type_label,
            location=location,
            retention_days=int(getattr(self.config, "retention_days", 90)),
        )

    def _initialize_feature_store(self) -> None:
        """
        Initialize the FeatureStore with proper configuration.

        Delegates to SchedulerInitComponent for lazy initialization.

        """
        self._feature_store = self._init_component.init_feature_store(
            config=self.config,
            connection=self._feature_store_connection,
            feature_engineer=self.feature_engineer,
        )

    def _collect_latest_data(self) -> None:
        """
        Collect latest data from Databento.

        Delegates to DataCollectionComponent for data fetching.

        Raises:
            ValueError: If DATABENTO_API_KEY is not set.
            ImportError: If databento library is not installed.

        """
        # Create wrapper function for ensure_dataset_registered to match expected signature
        def ensure_registered_wrapper(
            registry: RegistryProtocol | None,
            dataset_id: str,
            dataset_type_label: str,
            location: str,
            retention_days: int,
        ) -> None:
            self._registration_component.ensure_dataset_registered(
                registry=registry,
                dataset_id=dataset_id,
                dataset_type_label=dataset_type_label,
                location=location,
                retention_days=retention_days,
            )

        _collected, _failed = self._data_collection_component.collect_latest_data(
            config=self.config,
            catalog=self.catalog,
            registry=self._data_registry,
            ensure_registered_fn=ensure_registered_wrapper,
            get_previous_day_fn=self._get_previous_trading_day,
        )

        # Update current run ID from component
        self._current_run_id = self._data_collection_component._current_run_id

    def _collect_via_orchestrator(self) -> None:
        """
        Collect data via IngestionOrchestrator with optional dual-write.

        Delegates to OrchestratorCollectionComponent.

        Raises:
            ValueError: If API key or DB connection is missing.
            RuntimeError: If DataRegistry is not initialized.

        """
        self._orchestrator_component.collect_via_orchestrator(
            config=self.config,
            connection=self._feature_store_connection,
            registry=self._data_registry,
            catalog=self.catalog,
            dual_write=self._dual_write,
        )

    def _compute_features(self) -> None:
        """
        Compute features for newly collected data.

        Delegates to FeatureComputationComponent.

        """
        self._feature_computation_component.compute_features(
            config=self.config,
            catalog=self.catalog,
            feature_engineer=self.feature_engineer,
            feature_store=self._feature_store,
            init_feature_store_fn=self._initialize_feature_store,
            get_previous_day_fn=self._get_previous_trading_day,
        )

    def _clean_old_data(self) -> None:
        """
        Clean data older than retention period.

        Delegates to DataCleanupComponent.

        """
        self._cleanup_component.clean_old_data(retention_days=self.config.retention_days)


__all__ = [
    "DataSchedulerFacade",
    "create_data_scheduler",
    "use_legacy_scheduler",
]
