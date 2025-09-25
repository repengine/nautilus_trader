"""
Integration examples for extended metrics (Updated for direct catalog usage).

This module demonstrates how to integrate the extended metrics collectors with Nautilus
ParquetDataCatalog and ML components directly.

"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from ml.monitoring.extended_metrics import ExtendedMetricsManager


if TYPE_CHECKING:
    from polars import DataFrame as PlDataFrame

    from ml.features.engineering import FeatureEngineer
    from ml.features.engineering import IndicatorManager
    from ml.ml_types import StandardScaler as StandardScalerT
    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


logger = logging.getLogger(__name__)


# =============================================================================
# ParquetDataCatalog with Metrics Integration
# =============================================================================


class MonitoredDataCatalog:
    """
    Example of ParquetDataCatalog with integrated metrics collection.

    This demonstrates how to add comprehensive metrics to the data loading process using
    Nautilus native components directly.

    """

    def __init__(
        self,
        catalog: ParquetDataCatalog,
        metrics_manager: ExtendedMetricsManager | None = None,
    ) -> None:
        """
        Initialize monitored catalog wrapper.

        Parameters
        ----------
        catalog : ParquetDataCatalog
            The Nautilus data catalog to monitor.
        metrics_manager : ExtendedMetricsManager, optional
            The metrics manager for collecting stats.

        """
        self.catalog = catalog
        self.metrics = metrics_manager or ExtendedMetricsManager()

    def load_bars(
        self,
        instrument_ids: list[str],
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> PlDataFrame:
        """
        Load bars with quality metrics collection.

        Parameters
        ----------
        instrument_ids : list[str]
            List of instrument IDs to load.
        start : Any, optional
            Start time for data range.
        end : Any, optional
            End time for data range.

        Returns
        -------
        pl.DataFrame
            Loaded bar data with metrics collected.

        """
        from ml.data.catalog_utils import bars_to_dataframe

        start_time = time.perf_counter()

        try:
            # Load data using catalog utilities
            df = bars_to_dataframe(self.catalog, instrument_ids, start, end)

            # Collect quality metrics
            if self.metrics and self.metrics.data_quality:
                self.metrics.data_quality.collect_batch_metrics(
                    data=df,
                    data_type="bars",
                    source="catalog",
                )

            # Record load time
            load_time = time.perf_counter() - start_time
            logger.info(f"Loaded {len(df)} bars in {load_time:.2f}s")

            return df

        except Exception as e:
            logger.error(f"Failed to load bars: {e}")
            if self.metrics:
                self.metrics.data_quality.record_error("load_failure")
            raise

    def load_quotes(
        self,
        instrument_ids: list[str],
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> PlDataFrame:
        """
        Load quotes with quality metrics collection.

        Parameters
        ----------
        instrument_ids : list[str]
            List of instrument IDs to load.
        start : Any, optional
            Start time for data range.
        end : Any, optional
            End time for data range.

        Returns
        -------
        pl.DataFrame
            Loaded quote data with metrics collected.

        """
        from ml.data.catalog_utils import quotes_to_dataframe

        start_time = time.perf_counter()

        try:
            # Load data using catalog utilities
            df = quotes_to_dataframe(self.catalog, instrument_ids, start, end)

            # Collect quality metrics
            if self.metrics and self.metrics.data_quality:
                self.metrics.data_quality.collect_batch_metrics(
                    data=df,
                    data_type="quotes",
                    source="catalog",
                )

            # Record load time
            load_time = time.perf_counter() - start_time
            logger.info(f"Loaded {len(df)} quotes in {load_time:.2f}s")

            return df

        except Exception as e:
            logger.error(f"Failed to load quotes: {e}")
            if self.metrics:
                self.metrics.data_quality.record_error("load_failure")
            raise


# =============================================================================
# Feature Engineering with Metrics
# =============================================================================


class MonitoredFeatureEngineer:
    """
    Example of FeatureEngineer with integrated metrics collection.

    This demonstrates how to add performance and quality metrics to feature engineering
    operations.

    """

    def __init__(
        self,
        feature_engineer: FeatureEngineer,
        metrics_manager: ExtendedMetricsManager | None = None,
    ) -> None:
        """
        Initialize monitored feature engineer.

        Parameters
        ----------
        feature_engineer : FeatureEngineer
            The base feature engineer to wrap.
        metrics_manager : ExtendedMetricsManager, optional
            The metrics manager for collecting stats.

        """
        self.engineer = feature_engineer
        self.metrics = metrics_manager or ExtendedMetricsManager()

    def calculate_features(
        self,
        data: object,
        mode: str = "batch",
        **kwargs: object,
    ) -> object:
        """
        Calculate features with comprehensive metrics.

        Parameters
        ----------
        data : Any
            Input data for feature calculation.
        mode : str
            Calculation mode ("batch" or "online").
        **kwargs : Any
            Additional arguments for feature calculation.

        Returns
        -------
        Any
            Calculated features with metrics collected.

        """
        start_time = time.perf_counter()

        try:
            # Calculate features
            from typing import cast

            from ml.ml_types import DataFrameLike

            features: object
            if mode == "batch":
                features_df, _ = self.engineer.calculate_features(
                    cast(DataFrameLike, data),
                    mode="batch",
                )
                features = features_df
            else:
                # Expect required online kwargs
                indicator_manager = cast("IndicatorManager", kwargs.get("indicator_manager"))
                scaler = cast("StandardScalerT | None", kwargs.get("scaler"))
                features = self.engineer.calculate_features(
                    cast(dict[str, float], data),
                    mode="online",
                    indicator_manager=indicator_manager,
                    scaler=scaler,
                )

            # Collect engineering metrics
            if self.metrics and self.metrics.feature_engineering:
                calc_time = time.perf_counter() - start_time

                if mode == "batch":
                    try:
                        num_samples = len(cast(Any, data))
                    except Exception:
                        num_samples = 1
                    self.metrics.feature_engineering.record_batch_computation(
                        num_samples=num_samples,
                        computation_time=calc_time,
                    )
                else:
                    self.metrics.feature_engineering.record_online_computation(
                        computation_time=calc_time,
                    )

                # Check for feature quality issues
                if hasattr(features, "isna"):
                    feats_any = cast(Any, features)
                    nan_count = feats_any.isna().sum()
                    if nan_count > 0:
                        self.metrics.feature_engineering.record_feature_quality_issue(
                            issue_type="nan_values",
                            feature_names=feats_any.columns[feats_any.isna().any()].tolist(),
                        )

            return features

        except Exception as e:
            logger.error(f"Feature calculation failed: {e}")
            if self.metrics:
                self.metrics.feature_engineering.record_computation_error(str(e))
            raise


# =============================================================================
# Complete ML Pipeline with Metrics
# =============================================================================


class MonitoredMLPipeline:
    """
    Complete ML pipeline with integrated metrics at every stage.

    This demonstrates end-to-end metrics collection for:
    - Data loading
    - Feature engineering
    - Model inference
    - Signal generation

    """

    def __init__(
        self,
        catalog: ParquetDataCatalog,
        feature_engineer: FeatureEngineer,
        metrics_manager: ExtendedMetricsManager | None = None,
    ) -> None:
        """
        Initialize monitored ML pipeline.

        Parameters
        ----------
        catalog : ParquetDataCatalog
            The Nautilus data catalog.
        feature_engineer : FeatureEngineer
            The feature engineering component.
        metrics_manager : ExtendedMetricsManager, optional
            The metrics manager for collecting stats.

        """
        self.metrics = metrics_manager or ExtendedMetricsManager()

        # Wrap components with monitoring
        self.data_catalog = MonitoredDataCatalog(catalog, self.metrics)
        self.feature_engineer = MonitoredFeatureEngineer(feature_engineer, self.metrics)

        logger.info("Initialized monitored ML pipeline")

    def process_batch(
        self,
        instrument_ids: list[str],
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> tuple[object, object]:
        """
        Process batch data through complete pipeline.

        Parameters
        ----------
        instrument_ids : list[str]
            Instruments to process.
        start : Any, optional
            Start time for data range.
        end : Any, optional
            End time for data range.

        Returns
        -------
        tuple[Any, Any]
            Raw data and computed features.

        """
        pipeline_start = time.perf_counter()

        try:
            # Stage 1: Load data
            logger.info("Stage 1: Loading data...")
            bars = self.data_catalog.load_bars(instrument_ids, start, end)

            # Stage 2: Compute features
            logger.info("Stage 2: Computing features...")
            features = self.feature_engineer.calculate_features(bars, mode="batch")

            # Record pipeline metrics
            pipeline_time = time.perf_counter() - pipeline_start
            logger.info(f"Pipeline completed in {pipeline_time:.2f}s")

            if self.metrics:
                self.metrics.ml_inference.record_pipeline_execution(
                    pipeline_name="batch_processing",
                    execution_time=pipeline_time,
                    stages=["load", "features"],
                )

            return bars, features

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            if self.metrics:
                self.metrics.ml_inference.record_pipeline_error(
                    pipeline_name="batch_processing",
                    error=str(e),
                )
            raise

    def get_metrics_summary(self) -> dict[str, Any]:
        """
        Get comprehensive metrics summary.

        Returns
        -------
        dict[str, Any]
            Summary of all collected metrics.

        """
        if not self.metrics:
            return {}

        return {
            "data_quality": (
                self.metrics.data_quality.get_summary() if self.metrics.data_quality else {}
            ),
            "feature_engineering": (
                self.metrics.feature_engineering.get_summary()
                if self.metrics.feature_engineering
                else {}
            ),
            "ml_inference": (
                self.metrics.ml_inference.get_summary() if self.metrics.ml_inference else {}
            ),
        }


# =============================================================================
# Usage Examples
# =============================================================================


def example_basic_monitoring() -> None:
    """
    Demonstrate basic monitoring setup.
    """
    from ml.features.engineering import FeatureEngineer
    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

    # Initialize components
    catalog = ParquetDataCatalog("./data")
    feature_engineer = FeatureEngineer()
    metrics_manager = ExtendedMetricsManager()

    # Create monitored pipeline
    pipeline = MonitoredMLPipeline(
        catalog=catalog,
        feature_engineer=feature_engineer,
        metrics_manager=metrics_manager,
    )

    # Process data
    _bars, _features = pipeline.process_batch(
        instrument_ids=["EURUSD.SIM"],
        start="2023-01-01",
        end="2023-12-31",
    )

    # Get metrics summary
    summary = pipeline.get_metrics_summary()
    logger.info(f"Metrics summary: {summary}")


def example_production_monitoring() -> None:
    """
    Demonstrate production-grade monitoring with Prometheus export.
    """
    from ml.features.engineering import FeatureEngineer
    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

    # Initialize with Prometheus export
    metrics_manager = ExtendedMetricsManager(enable_prometheus=True)

    # Setup components
    catalog = ParquetDataCatalog("./data")
    feature_engineer = FeatureEngineer()

    # Create monitored components
    monitored_catalog = MonitoredDataCatalog(catalog, metrics_manager)
    monitored_engineer = MonitoredFeatureEngineer(feature_engineer, metrics_manager)

    # Simulate production workload
    for i in range(10):
        try:
            # Load and process data
            bars = monitored_catalog.load_bars(["SPY.NYSE"], start="2024-01-01")
            from typing import Any as _Any

            features_df = cast(_Any, monitored_engineer.calculate_features(bars, mode="batch"))

            logger.info(f"Iteration {i}: Processed {len(features_df)} samples")

        except Exception as e:
            logger.error(f"Iteration {i} failed: {e}")

    # Export metrics
    metrics_manager.export_metrics()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Run examples
    logger.info("Running basic monitoring example...")
    example_basic_monitoring()

    logger.info("\nRunning production monitoring example...")
    example_production_monitoring()
