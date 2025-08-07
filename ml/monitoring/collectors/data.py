# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Data quality metrics collector for ML monitoring.

This module provides comprehensive tracking of data quality, integrity, and loading
performance with Prometheus metrics.

"""

from __future__ import annotations

import time
import types
from typing import Any, Self

from ml._imports import HAS_PROMETHEUS
from ml._imports import Counter
from ml._imports import Gauge
from ml._imports import Histogram
from ml.monitoring._config import MonitoringConfig
from ml.monitoring.collectors.base import BaseMetricsCollector


class DataQualityCollector(BaseMetricsCollector):
    """
    Collector for data quality and integrity metrics.

    This collector tracks data loading performance, quality metrics,
    validation results, and data freshness for comprehensive observability
    of the ML data pipeline.

    Key Metrics
    -----------
    - Data loading performance and cache efficiency
    - Missing values and data quality ratios
    - Outlier detection and validation failures
    - Data staleness and freshness monitoring

    Parameters
    ----------
    config : MonitoringConfig
        Configuration for metrics collection.

    """

    def __init__(self, config: MonitoringConfig) -> None:
        """
        Initialize the data quality collector.

        Parameters
        ----------
        config : MonitoringConfig
            Configuration for metrics collection.

        """
        super().__init__(config)

    def _initialize_metrics(self) -> None:
        """
        Initialize Prometheus metrics for data quality tracking.
        """
        if not HAS_PROMETHEUS:
            return

        prefix = self._config.metrics_prefix
        buckets = self._config.get_histogram_buckets()

        # Data loading metrics
        self._data_rows_loaded_total = Counter(
            f"{prefix}_data_rows_loaded_total",
            "Total number of data rows loaded",
            ["instrument", "data_type"],
        )
        self._register_metric("data_rows_loaded_total", self._data_rows_loaded_total)

        self._data_load_duration_seconds = Histogram(
            f"{prefix}_data_load_duration_seconds",
            "Time taken to load data",
            ["instrument", "data_type"],
            buckets=buckets,
        )
        self._register_metric("data_load_duration_seconds", self._data_load_duration_seconds)

        # Cache metrics
        self._data_cache_hit_ratio = Gauge(
            f"{prefix}_data_cache_hit_ratio",
            "Cache hit ratio for data loading",
            ["instrument", "data_type"],
        )
        self._register_metric("data_cache_hit_ratio", self._data_cache_hit_ratio)

        self._data_cache_size_entries = Gauge(
            f"{prefix}_data_cache_size_entries",
            "Number of entries in data cache",
            [],
        )
        self._register_metric("data_cache_size_entries", self._data_cache_size_entries)

        # Data quality metrics
        self._data_missing_values_ratio = Gauge(
            f"{prefix}_data_missing_values_ratio",
            "Ratio of missing values in data",
            ["instrument", "data_type", "column"],
        )
        self._register_metric("data_missing_values_ratio", self._data_missing_values_ratio)

        self._data_outliers_detected_total = Counter(
            f"{prefix}_data_outliers_detected_total",
            "Total number of outliers detected",
            ["instrument", "data_type", "detection_method"],
        )
        self._register_metric("data_outliers_detected_total", self._data_outliers_detected_total)

        # Data validation metrics
        self._data_validation_failures_total = Counter(
            f"{prefix}_data_validation_failures_total",
            "Total number of data validation failures",
            ["instrument", "data_type", "validation_type"],
        )
        self._register_metric(
            "data_validation_failures_total",
            self._data_validation_failures_total,
        )

        self._data_validation_checks_total = Counter(
            f"{prefix}_data_validation_checks_total",
            "Total number of data validation checks performed",
            ["instrument", "data_type", "validation_type"],
        )
        self._register_metric("data_validation_checks_total", self._data_validation_checks_total)

        # Data freshness and staleness
        self._data_staleness_seconds = Gauge(
            f"{prefix}_data_staleness_seconds",
            "Time since last data update",
            ["instrument", "data_type"],
        )
        self._register_metric("data_staleness_seconds", self._data_staleness_seconds)

        self._data_last_updated_timestamp = Gauge(
            f"{prefix}_data_last_updated_timestamp",
            "Timestamp of last data update",
            ["instrument", "data_type"],
        )
        self._register_metric("data_last_updated_timestamp", self._data_last_updated_timestamp)

        # Data loading errors
        self._data_load_errors_total = Counter(
            f"{prefix}_data_load_errors_total",
            "Total number of data loading errors",
            ["instrument", "data_type", "error_type"],
        )
        self._register_metric("data_load_errors_total", self._data_load_errors_total)

    def record_data_load(
        self,
        instrument: str,
        data_type: str,
        rows_loaded: int,
        duration_seconds: float,
        cache_hit: bool = False,
        success: bool = True,
        error_type: str | None = None,
    ) -> None:
        """
        Record data loading metrics.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        data_type : str
            Type of data loaded (e.g., "bars", "quotes", "trades").
        rows_loaded : int
            Number of rows loaded.
        duration_seconds : float
            Time taken to load data.
        cache_hit : bool, default False
            Whether this was a cache hit.
        success : bool, default True
            Whether the load was successful.
        error_type : str, optional
            Type of error if load failed.

        """

        def _record() -> None:
            # Row count
            if success and self._data_rows_loaded_total is not None:
                self._data_rows_loaded_total.labels(
                    instrument=instrument,
                    data_type=data_type,
                ).inc(rows_loaded)

            # Duration
            if self._data_load_duration_seconds is not None:
                self._data_load_duration_seconds.labels(
                    instrument=instrument,
                    data_type=data_type,
                ).observe(duration_seconds)

            # Update last loaded timestamp
            if success and self._data_last_updated_timestamp is not None:
                self._data_last_updated_timestamp.labels(
                    instrument=instrument,
                    data_type=data_type,
                ).set(time.time())

            # Record errors
            if not success and error_type and self._data_load_errors_total is not None:
                self._data_load_errors_total.labels(
                    instrument=instrument,
                    data_type=data_type,
                    error_type=error_type,
                ).inc()

        self._safe_record("data_load", _record)

    def record_cache_stats(
        self,
        instrument: str,
        data_type: str,
        hit_ratio: float,
        cache_size: int | None = None,
    ) -> None:
        """
        Record cache performance metrics.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        data_type : str
            Type of data.
        hit_ratio : float
            Cache hit ratio (0.0-1.0).
        cache_size : int, optional
            Total cache size in entries.

        """

        def _record() -> None:
            if self._data_cache_hit_ratio is not None:
                self._data_cache_hit_ratio.labels(
                    instrument=instrument,
                    data_type=data_type,
                ).set(max(0.0, min(1.0, hit_ratio)))

            if cache_size is not None and self._data_cache_size_entries is not None:
                self._data_cache_size_entries.set(cache_size)

        self._safe_record("cache_stats", _record)

    def record_data_quality(
        self,
        instrument: str,
        data_type: str,
        missing_ratios: dict[str, float],
        outlier_counts: dict[str, int] | None = None,
        total_rows: int | None = None,
    ) -> None:
        """
        Record data quality metrics.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        data_type : str
            Type of data.
        missing_ratios : Dict[str, float]
            Missing value ratios per column (0.0-1.0).
        outlier_counts : Dict[str, int], optional
            Outlier counts per detection method.
        total_rows : int, optional
            Total number of rows for staleness calculation.

        """

        def _record() -> None:
            # Missing values
            if self._data_missing_values_ratio is not None:
                for column, ratio in missing_ratios.items():
                    self._data_missing_values_ratio.labels(
                        instrument=instrument,
                        data_type=data_type,
                        column=column,
                    ).set(max(0.0, min(1.0, ratio)))

            # Outliers
            if outlier_counts and self._data_outliers_detected_total is not None:
                for method, count in outlier_counts.items():
                    self._data_outliers_detected_total.labels(
                        instrument=instrument,
                        data_type=data_type,
                        detection_method=method,
                    ).inc(count)

            # Update staleness
            current_time = time.time()
            if self._data_last_updated_timestamp is not None:
                last_update = self.get_metric_value(
                    "data_last_updated_timestamp",
                    {"instrument": instrument, "data_type": data_type},
                )
                if last_update:
                    staleness = current_time - last_update
                    if self._data_staleness_seconds is not None:
                        self._data_staleness_seconds.labels(
                            instrument=instrument,
                            data_type=data_type,
                        ).set(staleness)

        self._safe_record("data_quality", _record)

    def record_data_validation(
        self,
        instrument: str,
        data_type: str,
        validation_type: str,
        passed: bool,
        total_checks: int = 1,
    ) -> None:
        """
        Record data validation results.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        data_type : str
            Type of data.
        validation_type : str
            Type of validation (e.g., "schema", "range", "consistency").
        passed : bool
            Whether validation passed.
        total_checks : int, default 1
            Total number of validation checks performed.

        """

        def _record() -> None:
            # Total checks
            if self._data_validation_checks_total is not None:
                self._data_validation_checks_total.labels(
                    instrument=instrument,
                    data_type=data_type,
                    validation_type=validation_type,
                ).inc(total_checks)

            # Failures
            if not passed and self._data_validation_failures_total is not None:
                self._data_validation_failures_total.labels(
                    instrument=instrument,
                    data_type=data_type,
                    validation_type=validation_type,
                ).inc()

        self._safe_record("data_validation", _record)

    def update_data_staleness(
        self,
        instrument: str,
        data_type: str,
        last_updated_timestamp: float | None = None,
    ) -> None:
        """
        Update data staleness metrics.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        data_type : str
            Type of data.
        last_updated_timestamp : float, optional
            Unix timestamp of last update. If None, uses current time.

        """

        def _record() -> None:
            current_time = time.time()

            if last_updated_timestamp is not None:
                # Set last updated timestamp
                if self._data_last_updated_timestamp is not None:
                    self._data_last_updated_timestamp.labels(
                        instrument=instrument,
                        data_type=data_type,
                    ).set(last_updated_timestamp)

                # Calculate and set staleness
                staleness = current_time - last_updated_timestamp
                if self._data_staleness_seconds is not None:
                    self._data_staleness_seconds.labels(
                        instrument=instrument,
                        data_type=data_type,
                    ).set(staleness)
            else:
                # Just update last timestamp to current time
                if self._data_last_updated_timestamp is not None:
                    self._data_last_updated_timestamp.labels(
                        instrument=instrument,
                        data_type=data_type,
                    ).set(current_time)

                # Reset staleness to 0
                if self._data_staleness_seconds is not None:
                    self._data_staleness_seconds.labels(
                        instrument=instrument,
                        data_type=data_type,
                    ).set(0.0)

        self._safe_record("data_staleness", _record)

    def time_data_load(
        self,
        instrument: str,
        data_type: str,
    ) -> DataLoadTimer:
        """
        Create a context manager for timing data loading.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        data_type : str
            Type of data being loaded.

        Returns
        -------
        DataLoadTimer
            Context manager for timing data loading operations.

        """
        return DataLoadTimer(self, instrument, data_type)

    def get_data_quality_summary(
        self,
        instrument: str,
        data_type: str,
    ) -> dict[str, Any]:
        """
        Get data quality summary for an instrument and data type.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        data_type : str
            Type of data.

        Returns
        -------
        Dict[str, Any]
            Summary of data quality metrics.

        """
        labels = {"instrument": instrument, "data_type": data_type}

        summary = {
            "instrument": instrument,
            "data_type": data_type,
            "cache_hit_ratio": self.get_metric_value("data_cache_hit_ratio", labels),
            "staleness_seconds": self.get_metric_value("data_staleness_seconds", labels),
            "last_updated": self.get_metric_value("data_last_updated_timestamp", labels),
        }

        return {k: v for k, v in summary.items() if v is not None}


class DataLoadTimer:
    """
    Context manager for timing data loading operations.
    """

    def __init__(
        self,
        collector: DataQualityCollector,
        instrument: str,
        data_type: str,
    ) -> None:
        """
        Initialize the data load timer.

        Parameters
        ----------
        collector : DataQualityCollector
            The collector to record metrics to.
        instrument : str
            Instrument identifier.
        data_type : str
            Type of data being loaded.

        """
        self._collector = collector
        self._instrument = instrument
        self._data_type = data_type
        self._start_time: float = 0.0
        self._rows_loaded: int = 0
        self._cache_hit: bool = False
        self._error_type: str | None = None
        self._missing_ratios: dict[str, float] = {}
        self._outlier_counts: dict[str, int] = {}

    def __enter__(self) -> Self:
        self._start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        duration = time.perf_counter() - self._start_time
        success = exc_type is None

        if not success and exc_type:
            self._error_type = exc_type.__name__

        # Record loading metrics
        self._collector.record_data_load(
            instrument=self._instrument,
            data_type=self._data_type,
            rows_loaded=self._rows_loaded,
            duration_seconds=duration,
            cache_hit=self._cache_hit,
            success=success,
            error_type=self._error_type,
        )

        # Record quality metrics if available
        if self._missing_ratios and success:
            self._collector.record_data_quality(
                instrument=self._instrument,
                data_type=self._data_type,
                missing_ratios=self._missing_ratios,
                outlier_counts=self._outlier_counts if self._outlier_counts else None,
                total_rows=self._rows_loaded,
            )

    def set_load_result(
        self,
        rows: int,
        cache_hit: bool = False,
        missing_ratios: dict[str, float] | None = None,
        outlier_counts: dict[str, int] | None = None,
    ) -> None:
        """
        Set results of the data loading operation.

        Parameters
        ----------
        rows : int
            Number of rows loaded.
        cache_hit : bool, default False
            Whether this was a cache hit.
        missing_ratios : Dict[str, float], optional
            Missing value ratios per column.
        outlier_counts : Dict[str, int], optional
            Outlier counts per detection method.

        """
        self._rows_loaded = rows
        self._cache_hit = cache_hit
        if missing_ratios:
            self._missing_ratios = missing_ratios
        if outlier_counts:
            self._outlier_counts = outlier_counts
