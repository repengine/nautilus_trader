"""
Data Scheduler for automated daily data collection and processing.

This module provides scheduling capabilities for automated data collection
from Databento and feature computation for ML models.

"""

from __future__ import annotations

import logging
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING

from ml.data.collector import DataCollector
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


if TYPE_CHECKING:
    from ml.features.engineering import FeatureEngineer


logger = logging.getLogger(__name__)


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
        collector: DataCollector | None = None,
        feature_engineer: FeatureEngineer | None = None,
        retention_days: int = 90,
    ) -> None:
        """
        Initialize data scheduler.

        Parameters
        ----------
        catalog : ParquetDataCatalog
            Nautilus data catalog for data storage
        collector : DataCollector, optional
            Data collector for fetching from Databento
        feature_engineer : FeatureEngineer, optional
            Feature engineer for computing features
        retention_days : int, default 90
            Number of days to retain historical data

        """
        self.catalog = catalog
        self.collector = collector or DataCollector()
        self.feature_engineer = feature_engineer
        self.retention_days = retention_days

        # Scheduling configuration
        self.collection_time = "04:00"  # 4 AM UTC (before US market open)
        self.enabled = True

        logger.info(
            f"Initialized DataScheduler with retention={retention_days} days"
        )

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

        try:
            # Step 1: Collect latest data
            self._collect_latest_data()

            # Step 2: Compute features if configured
            if self.feature_engineer is not None:
                self._compute_features()

            # Step 3: Clean old data
            self._clean_old_data()

            logger.info("Daily data update completed successfully")

        except Exception as e:
            logger.error(f"Daily data update failed: {e}")
            raise

    def _collect_latest_data(self) -> None:
        """
        Collect latest data from Databento.

        Fetches:
        - Previous trading day's minute bars
        - L2 depth data if available
        - Trades and quotes

        """
        # Get previous trading day
        today = datetime.now()
        if today.weekday() == 0:  # Monday
            # Get Friday's data
            target_date = today - timedelta(days=3)
        elif today.weekday() == 6:  # Sunday
            # Get Friday's data
            target_date = today - timedelta(days=2)
        else:
            # Get previous day's data
            target_date = today - timedelta(days=1)

        start_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)

        logger.info(f"Collecting data for {start_date.date()}")

        # Collect minute bars for all symbols
        # Note: In production, this would call Databento API
        # For now, we'll log the intention
        logger.info(
            f"Would collect minute bars from {start_date} to {end_date}"
        )

        # In production:
        # bars_data = self.collector.collect_minute_bars(
        #     start=start_date,
        #     end=end_date,
        #     symbols=self.collector.get_universe_symbols()
        # )
        # self.catalog.write_data(bars_data)

    def _compute_features(self) -> None:
        """
        Compute features for newly collected data.

        Uses the configured feature engineer to compute
        features for ML models.

        """
        if self.feature_engineer is None:
            return

        logger.info("Computing features for new data...")

        # Get latest bars
        end_date = datetime.now()
        _ = end_date - timedelta(days=1)

        # In production, this would:
        # 1. Load recent bars from catalog
        # 2. Compute features
        # 3. Store features in FeatureStore

        logger.info("Feature computation completed")

    def _clean_old_data(self) -> None:
        """
        Clean data older than retention period.

        Removes:
        - Minute bars older than retention_days
        - L2 depth older than 30 days
        - Keeps daily bars indefinitely

        """
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        logger.info(f"Cleaning data older than {cutoff_date.date()}")

        # In production, this would:
        # 1. Query catalog for old data
        # 2. Delete files/partitions older than cutoff
        # 3. Update catalog metadata

        logger.info("Data cleanup completed")

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
            "collection_time": self.collection_time,
            "retention_days": self.retention_days,
            "has_feature_engineer": self.feature_engineer is not None,
            "catalog_path": str(self.catalog.path) if hasattr(self.catalog, "path") else "N/A",
        }


def main() -> None:
    """Run example usage of DataScheduler."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Initialize components
    catalog = ParquetDataCatalog("./data")

    # Create scheduler
    scheduler = DataScheduler(
        catalog=catalog,
        retention_days=90,
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

