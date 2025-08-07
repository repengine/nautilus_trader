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
Extended metrics collectors for comprehensive ML system observability.

This module provides specialized collectors for different aspects of ML operations,
including model lifecycle, data quality, feature engineering, and performance
monitoring.

"""

from __future__ import annotations

import threading
import time
from abc import ABC
from abc import abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Self

from ml._imports import HAS_PROMETHEUS
from ml._imports import Counter
from ml._imports import Gauge
from ml._imports import Histogram
from ml.monitoring._config import MonitoringConfig
from ml.monitoring.collector import MLMetricsCollector


if TYPE_CHECKING:
    pass


class BaseMetricsCollector(ABC):
    """
    Abstract base class for all specialized metrics collectors.

    Provides common functionality for metric collection with graceful degradation
    when Prometheus is not available.

    Parameters
    ----------
    config : MonitoringConfig
        Configuration for metrics collection.

    """

    def __init__(self, config: MonitoringConfig) -> None:
        """
        Initialize the base metrics collector.

        Parameters
        ----------
        config : MonitoringConfig
            Configuration for metrics collection.

        """
        self._config = config
        self._enabled = config.enabled and HAS_PROMETHEUS
        self._lock = threading.RLock()

        if self._enabled:
            self._initialize_metrics()

    @abstractmethod
    def _initialize_metrics(self) -> None:
        """
        Initialize Prometheus metrics.
        """

    def is_enabled(self) -> bool:
        """
        Check if metrics collection is enabled.

        Returns
        -------
        bool
            True if metrics collection is enabled.

        """
        return self._enabled


class ModelLifecycleCollector(BaseMetricsCollector):
    """
    Collector for model lifecycle metrics.

    Tracks model versioning, deployment, training duration, and model metadata.

    """

    def _initialize_metrics(self) -> None:
        """
        Initialize model lifecycle metrics.
        """
        if not HAS_PROMETHEUS:
            return

        from prometheus_client import Info

        prefix = self._config.metrics_prefix

        # Model information (using Info metric for metadata)
        self._model_info = Info(
            f"{prefix}_model_info",
            "Model deployment information",
        )

        # Training metrics
        self._model_last_trained = Gauge(
            f"{prefix}_model_last_trained_timestamp",
            "Unix timestamp of last model training",
            ["model", "instrument"],
        )

        self._model_training_duration = Histogram(
            f"{prefix}_model_training_duration_seconds",
            "Model training duration by phase",
            ["model", "phase"],
            buckets=(1, 10, 60, 300, 600, 1800, 3600, 7200),
        )

        self._model_size_bytes = Gauge(
            f"{prefix}_model_size_bytes",
            "Model size in bytes",
            ["model", "format"],
        )

        self._model_load_time = Histogram(
            f"{prefix}_model_load_time_seconds",
            "Model load time",
            ["model", "location"],
            buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10),
        )

    def record_model_deployment(
        self,
        model: str,
        version: str,
        instrument: str,
        git_commit: str | None = None,
    ) -> None:
        """
        Record model deployment event.

        Parameters
        ----------
        model : str
            Model identifier.
        version : str
            Model version.
        instrument : str
            Instrument identifier.
        git_commit : str, optional
            Git commit hash for model code.

        """
        if not self._enabled:
            return

        with self._lock:
            if self._model_info is not None:
                info = {
                    "model": model,
                    "version": version,
                    "instrument": instrument,
                    "deployment_time": datetime.utcnow().isoformat(),
                }
                if git_commit:
                    info["git_commit"] = git_commit

                self._model_info.info(info)

    def record_training_completed(
        self,
        model: str,
        instrument: str,
        duration_seconds: float,
        phase_durations: dict[str, float] | None = None,
    ) -> None:
        """
        Record model training completion.

        Parameters
        ----------
        model : str
            Model identifier.
        instrument : str
            Instrument identifier.
        duration_seconds : float
            Total training duration in seconds.
        phase_durations : dict[str, float], optional
            Duration for each training phase.

        """
        if not self._enabled:
            return

        with self._lock:
            # Record last trained timestamp
            if self._model_last_trained is not None:
                self._model_last_trained.labels(
                    model=model,
                    instrument=instrument,
                ).set(time.time())

            # Record phase durations
            if self._model_training_duration is not None and phase_durations:
                for phase, duration in phase_durations.items():
                    self._model_training_duration.labels(
                        model=model,
                        phase=phase,
                    ).observe(duration)

    def record_model_size(
        self,
        model: str,
        size_bytes: int,
        format: str = "pickle",
    ) -> None:
        """
        Record model size.

        Parameters
        ----------
        model : str
            Model identifier.
        size_bytes : int
            Model size in bytes.
        format : str, default "pickle"
            Model serialization format.

        """
        if not self._enabled:
            return

        with self._lock:
            if self._model_size_bytes is not None:
                self._model_size_bytes.labels(
                    model=model,
                    format=format,
                ).set(size_bytes)


class DataQualityCollector(BaseMetricsCollector):
    """
    Collector for data quality metrics.

    Monitors data integrity, missing values, outliers, and data freshness.

    """

    def _initialize_metrics(self) -> None:
        """
        Initialize data quality metrics.
        """
        if not HAS_PROMETHEUS:
            return

        prefix = self._config.metrics_prefix

        self._missing_values_ratio = Gauge(
            f"{prefix}_data_missing_values_ratio",
            "Ratio of missing values in data",
            ["instrument", "data_type", "column"],
        )

        self._outliers_detected = Counter(
            f"{prefix}_data_outliers_detected_total",
            "Total outliers detected",
            ["instrument", "detection_method"],
        )

        self._data_staleness = Gauge(
            f"{prefix}_data_staleness_seconds",
            "Time since last data update",
            ["instrument", "data_type"],
        )

        self._data_validation_failures = Counter(
            f"{prefix}_data_validation_failures_total",
            "Data validation failures",
            ["instrument", "validation_type"],
        )

        self._data_load_latency = Histogram(
            f"{prefix}_data_load_latency_seconds",
            "Data loading latency",
            ["instrument", "data_type", "source"],
            buckets=self._config.get_histogram_buckets(),
        )

    def record_data_quality(
        self,
        instrument: str,
        data_type: str,
        missing_ratios: dict[str, float] | None = None,
        outlier_count: int = 0,
        detection_method: str = "zscore",
    ) -> None:
        """
        Record data quality metrics.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        data_type : str
            Type of data (bars, quotes, trades).
        missing_ratios : dict[str, float], optional
            Missing value ratios per column.
        outlier_count : int, default 0
            Number of outliers detected.
        detection_method : str, default "zscore"
            Outlier detection method used.

        """
        if not self._enabled:
            return

        with self._lock:
            # Record missing values
            if self._missing_values_ratio is not None and missing_ratios:
                for column, ratio in missing_ratios.items():
                    self._missing_values_ratio.labels(
                        instrument=instrument,
                        data_type=data_type,
                        column=column,
                    ).set(ratio)

            # Record outliers
            if self._outliers_detected is not None and outlier_count > 0:
                self._outliers_detected.labels(
                    instrument=instrument,
                    detection_method=detection_method,
                ).inc(outlier_count)

    def record_data_staleness(
        self,
        instrument: str,
        data_type: str,
        staleness_seconds: float,
    ) -> None:
        """
        Record data staleness.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        data_type : str
            Type of data.
        staleness_seconds : float
            Time since last update in seconds.

        """
        if not self._enabled:
            return

        with self._lock:
            if self._data_staleness is not None:
                self._data_staleness.labels(
                    instrument=instrument,
                    data_type=data_type,
                ).set(staleness_seconds)


class FeatureEngineeringCollector(BaseMetricsCollector):
    """
    Collector for feature engineering metrics.

    Tracks feature computation performance, cache efficiency, feature drift, and feature
    importance.

    """

    def _initialize_metrics(self) -> None:
        """
        Initialize feature engineering metrics.
        """
        if not HAS_PROMETHEUS:
            return

        prefix = self._config.metrics_prefix

        self._feature_computation_errors = Counter(
            f"{prefix}_feature_computation_errors_total",
            "Feature computation errors",
            ["instrument", "feature_type", "error_type"],
        )

        self._feature_cache_hit_ratio = Gauge(
            f"{prefix}_feature_cache_hit_ratio",
            "Feature cache hit ratio",
            ["instrument", "cache_level"],
        )

        self._feature_drift_score = Gauge(
            f"{prefix}_feature_drift_score",
            "Feature drift score",
            ["instrument", "feature", "reference_window"],
        )

        self._feature_importance = Gauge(
            f"{prefix}_feature_importance_score",
            "Feature importance score",
            ["model", "feature"],
        )

        self._feature_computation_latency = Histogram(
            f"{prefix}_feature_computation_latency_seconds",
            "Feature computation latency",
            ["feature_set", "computation_mode"],
            buckets=self._config.get_histogram_buckets(),
        )

    def record_feature_drift(
        self,
        instrument: str,
        feature: str,
        drift_score: float,
        reference_window: str = "training",
    ) -> None:
        """
        Record feature drift.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        feature : str
            Feature name.
        drift_score : float
            Drift score (e.g., KL divergence).
        reference_window : str, default "training"
            Reference window for drift calculation.

        """
        if not self._enabled:
            return

        with self._lock:
            if self._feature_drift_score is not None:
                self._feature_drift_score.labels(
                    instrument=instrument,
                    feature=feature,
                    reference_window=reference_window,
                ).set(drift_score)

    def record_feature_importance(
        self,
        model: str,
        importances: dict[str, float],
    ) -> None:
        """
        Record feature importance scores.

        Parameters
        ----------
        model : str
            Model identifier.
        importances : dict[str, float]
            Feature importance scores.

        """
        if not self._enabled:
            return

        with self._lock:
            if self._feature_importance is not None:
                for feature, importance in importances.items():
                    self._feature_importance.labels(
                        model=model,
                        feature=feature,
                    ).set(importance)

    def record_cache_hit(
        self,
        instrument: str,
        cache_level: str,
        hit_ratio: float,
    ) -> None:
        """
        Record cache hit ratio.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        cache_level : str
            Cache level (memory, disk).
        hit_ratio : float
            Cache hit ratio (0.0-1.0).

        """
        if not self._enabled:
            return

        with self._lock:
            if self._feature_cache_hit_ratio is not None:
                self._feature_cache_hit_ratio.labels(
                    instrument=instrument,
                    cache_level=cache_level,
                ).set(hit_ratio)


class PerformanceDegradationMonitor(BaseMetricsCollector):
    """
    Monitor for ML model performance degradation.

    Tracks model accuracy over time, prediction distribution shifts, and identifies when
    retraining is needed.

    """

    def _initialize_metrics(self) -> None:
        """
        Initialize performance degradation metrics.
        """
        if not HAS_PROMETHEUS:
            return

        prefix = self._config.metrics_prefix

        self._model_accuracy_rolling = Gauge(
            f"{prefix}_model_accuracy_rolling",
            "Rolling model accuracy",
            ["model", "window", "metric_type"],
        )

        self._prediction_distribution_shift = Gauge(
            f"{prefix}_prediction_distribution_shift",
            "Prediction distribution shift",
            ["model", "shift_metric"],
        )

        self._inference_timeout_ratio = Gauge(
            f"{prefix}_inference_timeout_ratio",
            "Ratio of inference timeouts",
            ["model", "threshold_ms"],
        )

        self._model_retraining_required = Gauge(
            f"{prefix}_model_retraining_required",
            "Model retraining required flag",
            ["model", "reason"],
        )

        self._prediction_confidence_percentiles = Gauge(
            f"{prefix}_prediction_confidence_percentiles",
            "Prediction confidence percentiles",
            ["model", "percentile"],
        )

    def record_model_accuracy(
        self,
        model: str,
        window: str,
        metric_type: str,
        value: float,
    ) -> None:
        """
        Record model accuracy metrics.

        Parameters
        ----------
        model : str
            Model identifier.
        window : str
            Time window (1h, 24h, 7d).
        metric_type : str
            Metric type (accuracy, precision, recall, f1).
        value : float
            Metric value.

        """
        if not self._enabled:
            return

        with self._lock:
            if self._model_accuracy_rolling is not None:
                self._model_accuracy_rolling.labels(
                    model=model,
                    window=window,
                    metric_type=metric_type,
                ).set(value)

    def record_distribution_shift(
        self,
        model: str,
        shift_metric: str,
        value: float,
    ) -> None:
        """
        Record prediction distribution shift.

        Parameters
        ----------
        model : str
            Model identifier.
        shift_metric : str
            Shift metric type (psi, kl_divergence, wasserstein).
        value : float
            Shift value.

        """
        if not self._enabled:
            return

        with self._lock:
            if self._prediction_distribution_shift is not None:
                self._prediction_distribution_shift.labels(
                    model=model,
                    shift_metric=shift_metric,
                ).set(value)

    def set_retraining_required(
        self,
        model: str,
        reason: str,
        required: bool,
    ) -> None:
        """
        Set model retraining required flag.

        Parameters
        ----------
        model : str
            Model identifier.
        reason : str
            Reason for retraining (drift, performance, schedule).
        required : bool
            Whether retraining is required.

        """
        if not self._enabled:
            return

        with self._lock:
            if self._model_retraining_required is not None:
                self._model_retraining_required.labels(
                    model=model,
                    reason=reason,
                ).set(1 if required else 0)


class ResourceUtilizationCollector(BaseMetricsCollector):
    """
    Collector for ML resource utilization metrics.

    Tracks GPU usage, memory consumption, and storage metrics.

    """

    def _initialize_metrics(self) -> None:
        """
        Initialize resource utilization metrics.
        """
        if not HAS_PROMETHEUS:
            return

        prefix = self._config.metrics_prefix

        self._gpu_utilization = Gauge(
            f"{prefix}_gpu_utilization_percent",
            "GPU utilization percentage",
            ["device", "metric"],
        )

        self._model_memory_usage = Gauge(
            f"{prefix}_model_memory_usage_bytes",
            "Model memory usage",
            ["model", "memory_type"],
        )

        self._feature_store_size = Gauge(
            f"{prefix}_feature_store_size_bytes",
            "Feature store size",
            ["storage_type"],
        )

        self._inference_batch_size = Gauge(
            f"{prefix}_inference_batch_size",
            "Inference batch size",
            ["model"],
        )

        self._training_data_rows = Counter(
            f"{prefix}_training_data_rows_processed_total",
            "Training data rows processed",
            ["dataset"],
        )

    def record_gpu_utilization(
        self,
        device: str,
        compute_percent: float,
        memory_percent: float,
    ) -> None:
        """
        Record GPU utilization.

        Parameters
        ----------
        device : str
            GPU device identifier (e.g., cuda:0).
        compute_percent : float
            GPU compute utilization percentage.
        memory_percent : float
            GPU memory utilization percentage.

        """
        if not self._enabled:
            return

        with self._lock:
            if self._gpu_utilization is not None:
                self._gpu_utilization.labels(
                    device=device,
                    metric="compute",
                ).set(compute_percent)

                self._gpu_utilization.labels(
                    device=device,
                    metric="memory",
                ).set(memory_percent)

    def record_model_memory(
        self,
        model: str,
        memory_bytes: int,
        memory_type: str = "resident",
    ) -> None:
        """
        Record model memory usage.

        Parameters
        ----------
        model : str
            Model identifier.
        memory_bytes : int
            Memory usage in bytes.
        memory_type : str, default "resident"
            Memory type (resident, virtual, gpu).

        """
        if not self._enabled:
            return

        with self._lock:
            if self._model_memory_usage is not None:
                self._model_memory_usage.labels(
                    model=model,
                    memory_type=memory_type,
                ).set(memory_bytes)


class MLMetricsRegistry:
    """
    Central registry for all ML metrics collectors.

    Provides unified access to all specialized collectors and manages
    the metrics server.

    Parameters
    ----------
    config : MonitoringConfig
        Configuration for metrics collection.

    """

    def __init__(self, config: MonitoringConfig) -> None:
        """
        Initialize the metrics registry.

        Parameters
        ----------
        config : MonitoringConfig
            Configuration for metrics collection.

        """
        self.config = config

        # Initialize all collectors
        self.ml_metrics = MLMetricsCollector(config)  # Existing core collector
        self.model_lifecycle = ModelLifecycleCollector(config)
        self.data_quality = DataQualityCollector(config)
        self.feature_engineering = FeatureEngineeringCollector(config)
        self.performance = PerformanceDegradationMonitor(config)
        self.resources = ResourceUtilizationCollector(config)

        # Import server only if needed
        from ml.monitoring.server import MetricsServer

        self.server = MetricsServer(config)

    def start(self) -> None:
        """
        Start metrics collection and server.
        """
        self.server.start()

    def stop(self) -> None:
        """
        Stop metrics collection and server.
        """
        self.server.stop()

    def get_collector(self, collector_type: str) -> BaseMetricsCollector | None:
        """
        Get specific collector by type.

        Parameters
        ----------
        collector_type : str
            Type of collector to retrieve.

        Returns
        -------
        BaseMetricsCollector | None
            The requested collector or None if not found.

        """
        collectors = {
            "ml": self.ml_metrics,
            "model": self.model_lifecycle,
            "data": self.data_quality,
            "features": self.feature_engineering,
            "performance": self.performance,
            "resources": self.resources,
        }
        return collectors.get(collector_type)

    def __enter__(self) -> Self:
        """
        Context manager entry.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Context manager exit.
        """
        self.stop()
