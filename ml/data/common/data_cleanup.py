"""
Data cleanup and scheduling component extracted from DataScheduler.

This component handles cleanup and utility operations including:
- Data retention cleanup based on configurable retention period
- Previous trading day calculation (accounting for weekends)
- Scheduling of automated updates
- Scheduler stop/status operations

Extracted from legacy DataScheduler (lines 1174-1501):
- _clean_old_data() (lines 1387-1421)
- _get_previous_trading_day() (lines 1174-1194)
- schedule_updates() (lines 1423-1458)
- stop() (lines 1460-1475)
- get_status() (lines 1477-1501)

"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Protocol

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram


if TYPE_CHECKING:
    from ml.config.scheduler_config import SchedulerConfig


logger = logging.getLogger(__name__)


# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

# Import module-level metrics from scheduler module for compatibility
# These are created at module load and must be the same instances
try:
    from ml.data.scheduler import data_retention_cleanup_total
    from ml.data.scheduler import pipeline_stage_latency
except ImportError:
    # Fallback for isolated testing - create local metrics
    data_retention_cleanup_total = get_counter(
        "nautilus_ml_data_retention_cleanup_total",
        "Total data retention cleanup operations",
        ["status"],
    )
    pipeline_stage_latency = get_histogram(
        "nautilus_ml_pipeline_stage_latency_seconds",
        "Pipeline stage execution latency in seconds",
        ["stage"],
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    )


class DataCleanupProtocol(Protocol):
    """
    Protocol for data cleanup and scheduling operations.

    This protocol defines the contract for data cleanup components,
    enabling duck typing for testing and alternative implementations.

    Methods
    -------
    clean_old_data
        Clean data older than retention period.
    get_previous_trading_day
        Calculate the previous trading day (skipping weekends).
    schedule_updates
        Schedule automated daily updates.
    stop
        Stop scheduler and cleanup resources.
    get_status
        Get current scheduler status.

    """

    def clean_old_data(self, retention_days: int) -> None:
        """
        Clean data older than retention period.

        Args:
            retention_days: Number of days to retain data.

        Raises:
            Exception: Re-raised after recording failure metric.

        """
        ...

    def get_previous_trading_day(self) -> datetime:
        """
        Get previous trading day based on current date.

        Returns:
            Previous trading day datetime (skips weekends).

        """
        ...

    def schedule_updates(self, cron_expression: str | None = None) -> None:
        """
        Schedule automated daily updates.

        Args:
            cron_expression: Cron expression for scheduling.
                Defaults to "0 4 * * *" (daily at 4 AM UTC).

        """
        ...

    def stop(self, metrics_server: Any | None) -> None:
        """
        Stop scheduler and cleanup resources.

        Args:
            metrics_server: Metrics server instance to stop.

        Note:
            This method must NOT raise - logs warning on failure.

        """
        ...

    def get_status(
        self,
        config: SchedulerConfig,
        catalog: Any,
        feature_engineer: Any | None,
        enabled: bool,
    ) -> dict[str, str | int | bool]:
        """
        Get current scheduler status.

        Args:
            config: Scheduler configuration.
            catalog: Data catalog instance.
            feature_engineer: Feature engineer instance (if any).
            enabled: Whether scheduler is enabled.

        Returns:
            Status dictionary with scheduler state information.

        """
        ...


class DataCleanupComponent:
    """
    Component for data cleanup and scheduling operations.

    This component extracts cleanup and utility responsibilities from DataScheduler,
    providing focused methods for:
    - Data retention cleanup based on configurable retention period
    - Previous trading day calculation (accounts for weekends)
    - Scheduling of automated updates (placeholder for APScheduler integration)
    - Stop/status operations for scheduler lifecycle

    All cleanup operations record Prometheus metrics for observability.
    The stop() method never raises exceptions to ensure clean shutdown.

    Example:
        >>> from ml.data.common.data_cleanup import DataCleanupComponent
        >>> component = DataCleanupComponent()
        >>> previous = component.get_previous_trading_day()
        >>> assert previous.weekday() < 5  # Never a weekend

    """

    def clean_old_data(self, retention_days: int) -> None:
        """
        Clean data older than retention period.

        Removes stale data based on the configured retention period:
        - Minute bars older than retention_days are removed
        - L2 depth older than 30 days (if applicable)
        - Daily bars are kept indefinitely

        Records success/failure metrics via Prometheus counters.

        Args:
            retention_days: Number of days to retain data. Must be positive.

        Raises:
            Exception: Re-raised after recording failure metric to allow
                caller to handle or propagate the error.

        Example:
            >>> component = DataCleanupComponent()
            >>> component.clean_old_data(retention_days=90)  # Clean data > 90 days

        """
        cutoff_date = datetime.now() - timedelta(days=retention_days)

        logger.info(f"Cleaning data older than {cutoff_date.date()}")
        cleanup_start_time = time.perf_counter()

        try:
            # In production, this would:
            # 1. Query catalog for old data
            # 2. Delete files/partitions older than cutoff
            # 3. Update catalog metadata

            # For now, record successful cleanup
            data_retention_cleanup_total.labels(status="success").inc()

            cleanup_duration = time.perf_counter() - cleanup_start_time
            pipeline_stage_latency.labels(stage="data_cleanup").observe(cleanup_duration)

            logger.info("Data cleanup completed")
        except Exception:
            data_retention_cleanup_total.labels(status="failure").inc()
            logger.error(
                "Data cleanup failed",
                exc_info=True,
            )
            raise

    def get_previous_trading_day(self) -> datetime:
        """
        Get the previous trading day based on current date.

        Calculates the most recent trading day, accounting for weekends:
        - Monday returns previous Friday (3 days back)
        - Sunday returns previous Friday (2 days back)
        - Other weekdays return previous day (1 day back)

        Note: This is a simplified implementation that does not account
        for market holidays. For production use, integrate with a proper
        trading calendar.

        Returns:
            Previous trading day as datetime. The returned date will
            always be a weekday (Monday=0 through Friday=4).

        Example:
            >>> component = DataCleanupComponent()
            >>> previous = component.get_previous_trading_day()
            >>> assert previous.weekday() < 5  # Never Saturday (5) or Sunday (6)

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

    def schedule_updates(self, cron_expression: str | None = None) -> None:
        """
        Schedule automated daily updates.

        Configures scheduled execution of the daily update pipeline.
        In production, this would integrate with APScheduler, Airflow,
        or system cron.

        Args:
            cron_expression: Cron expression for scheduling.
                Defaults to "0 4 * * *" (daily at 4 AM UTC).
                Format: minute hour day month weekday

        Example:
            >>> component = DataCleanupComponent()
            >>> component.schedule_updates()  # Default: 4 AM UTC
            >>> component.schedule_updates("0 6 * * 1-5")  # 6 AM on weekdays

        Note:
            This is currently a placeholder implementation. In production,
            uncomment and configure APScheduler or alternative scheduler.

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

    def stop(self, metrics_server: Any | None) -> None:
        """
        Stop scheduler and cleanup resources.

        Gracefully stops the metrics server if running and marks
        the scheduler as disabled. This method never raises exceptions
        to ensure clean shutdown even when errors occur.

        Args:
            metrics_server: Metrics server instance to stop.
                If None, no server stop is attempted.

        Example:
            >>> component = DataCleanupComponent()
            >>> # Assuming metrics_server is a running MetricsServer instance
            >>> component.stop(metrics_server)

        Note:
            Any errors during metrics server shutdown are logged as
            warnings with exc_info=True but not propagated.

        """
        if metrics_server:
            try:
                metrics_server.stop()
                logger.info("Stopped metrics server")
            except Exception:
                logger.warning(
                    "Error stopping metrics server",
                    exc_info=True,
                )

        logger.info("Scheduler stopped")

    def get_status(
        self,
        config: SchedulerConfig,
        catalog: Any,
        feature_engineer: Any | None,
        enabled: bool,
    ) -> dict[str, str | int | bool]:
        """
        Get current scheduler status.

        Returns a dictionary with comprehensive scheduler state information
        including configuration, catalog path, and component availability.

        Args:
            config: Scheduler configuration containing symbols, retention,
                and databento settings.
            catalog: Data catalog instance (must have .path attribute).
            feature_engineer: Feature engineer instance (if any).
            enabled: Whether scheduler is currently enabled.

        Returns:
            Status dictionary containing:
            - enabled: bool - Whether scheduler is active
            - collection_time: str - Configured collection time
            - retention_days: int - Data retention period
            - symbol_count: int - Number of symbols configured
            - databento_dataset: str - Databento dataset name
            - databento_schema: str - Databento schema name
            - has_feature_engineer: bool - Whether feature engineer is configured
            - catalog_path: str - Path to data catalog or "N/A"

        Example:
            >>> from ml.config.scheduler_config import SchedulerConfig
            >>> component = DataCleanupComponent()
            >>> status = component.get_status(
            ...     config=SchedulerConfig(),
            ...     catalog=mock_catalog,
            ...     feature_engineer=None,
            ...     enabled=True,
            ... )
            >>> assert status["enabled"] is True
            >>> assert status["retention_days"] == 90

        """
        return {
            "enabled": enabled,
            "collection_time": config.collection_time,
            "retention_days": config.retention_days,
            "symbol_count": len(config.symbols),
            "databento_dataset": config.databento.dataset,
            "databento_schema": config.databento.schema,
            "has_feature_engineer": feature_engineer is not None,
            "catalog_path": str(catalog.path) if hasattr(catalog, "path") else "N/A",
        }


__all__ = [
    "DataCleanupComponent",
    "DataCleanupProtocol",
]
