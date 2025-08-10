
"""
Feature engineering metrics collector for ML monitoring.

This module provides comprehensive tracking of feature computation performance, quality,
drift detection, and cache efficiency with Prometheus metrics.

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


class FeatureEngineeringCollector(BaseMetricsCollector):
    """
    Collector for feature engineering and computation metrics.

    This collector tracks feature computation performance, quality metrics,
    drift detection, cache efficiency, and feature importance for comprehensive
    observability of the feature engineering pipeline.

    Key Metrics
    -----------
    - Feature computation latency and performance
    - Feature cache hit ratios and efficiency
    - Feature drift detection and monitoring
    - Feature importance and quality metrics

    Parameters
    ----------
    config : MonitoringConfig
        Configuration for metrics collection.

    """

    def __init__(self, config: MonitoringConfig) -> None:
        """
        Initialize the feature engineering collector.

        Parameters
        ----------
        config : MonitoringConfig
            Configuration for metrics collection.

        """
        super().__init__(config)

    def _initialize_metrics(self) -> None:
        """
        Initialize Prometheus metrics for feature engineering tracking.
        """
        if not HAS_PROMETHEUS:
            return

        prefix = self._config.metrics_prefix
        buckets = self._config.get_histogram_buckets()

        # Feature computation metrics
        self._feature_computation_duration_seconds = Histogram(
            f"{prefix}_feature_computation_duration_seconds",
            "Time taken for feature computation",
            ["instrument", "feature_type", "computation_mode"],
            buckets=buckets,
        )
        self._register_metric(
            "feature_computation_duration_seconds",
            self._feature_computation_duration_seconds,
        )

        self._features_computed_total = Counter(
            f"{prefix}_features_computed_total",
            "Total number of features computed",
            ["instrument", "feature_type", "computation_mode"],
        )
        self._register_metric("features_computed_total", self._features_computed_total)

        # Feature computation errors
        self._feature_computation_errors_total = Counter(
            f"{prefix}_feature_computation_errors_total",
            "Total number of feature computation errors",
            ["instrument", "feature_type", "error_type"],
        )
        self._register_metric(
            "feature_computation_errors_total",
            self._feature_computation_errors_total,
        )

        # Feature cache metrics
        self._feature_cache_hit_ratio = Gauge(
            f"{prefix}_feature_cache_hit_ratio",
            "Cache hit ratio for feature computation",
            ["instrument", "cache_level"],
        )
        self._register_metric("feature_cache_hit_ratio", self._feature_cache_hit_ratio)

        self._feature_cache_hits_total = Counter(
            f"{prefix}_feature_cache_hits_total",
            "Total number of feature cache hits",
            ["instrument", "cache_level"],
        )
        self._register_metric("feature_cache_hits_total", self._feature_cache_hits_total)

        self._feature_cache_misses_total = Counter(
            f"{prefix}_feature_cache_misses_total",
            "Total number of feature cache misses",
            ["instrument", "cache_level"],
        )
        self._register_metric("feature_cache_misses_total", self._feature_cache_misses_total)

        self._feature_cache_size_entries = Gauge(
            f"{prefix}_feature_cache_size_entries",
            "Number of entries in feature cache",
            ["cache_level"],
        )
        self._register_metric("feature_cache_size_entries", self._feature_cache_size_entries)

        # Feature drift metrics
        self._feature_drift_score = Gauge(
            f"{prefix}_feature_drift_score",
            "Feature drift score compared to reference",
            ["instrument", "feature", "reference_window"],
        )
        self._register_metric("feature_drift_score", self._feature_drift_score)

        self._feature_drift_alerts_total = Counter(
            f"{prefix}_feature_drift_alerts_total",
            "Total number of feature drift alerts",
            ["instrument", "feature", "drift_type"],
        )
        self._register_metric("feature_drift_alerts_total", self._feature_drift_alerts_total)

        # Feature importance and quality
        self._feature_importance_score = Gauge(
            f"{prefix}_feature_importance_score",
            "Feature importance score from model",
            ["model", "feature"],
        )
        self._register_metric("feature_importance_score", self._feature_importance_score)

        self._feature_null_ratio = Gauge(
            f"{prefix}_feature_null_ratio",
            "Ratio of null values in computed features",
            ["instrument", "feature"],
        )
        self._register_metric("feature_null_ratio", self._feature_null_ratio)

        self._feature_infinite_ratio = Gauge(
            f"{prefix}_feature_infinite_ratio",
            "Ratio of infinite values in computed features",
            ["instrument", "feature"],
        )
        self._register_metric("feature_infinite_ratio", self._feature_infinite_ratio)

        # Feature freshness
        self._feature_last_computed_timestamp = Gauge(
            f"{prefix}_feature_last_computed_timestamp",
            "Timestamp when features were last computed",
            ["instrument", "feature_type"],
        )
        self._register_metric(
            "feature_last_computed_timestamp",
            self._feature_last_computed_timestamp,
        )

    def record_feature_computation(
        self,
        instrument: str,
        feature_type: str,
        computation_duration: float,
        features_computed: int,
        computation_mode: str = "batch",
        success: bool = True,
        error_type: str | None = None,
    ) -> None:
        """
        Record feature computation metrics.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        feature_type : str
            Type of features computed (e.g., "technical", "microstructure").
        computation_duration : float
            Time taken for computation in seconds.
        features_computed : int
            Number of features computed.
        computation_mode : str, default "batch"
            Computation mode ("batch", "streaming", "online").
        success : bool, default True
            Whether computation was successful.
        error_type : str, optional
            Type of error if computation failed.

        """

        def _record() -> None:
            # Computation duration
            if self._feature_computation_duration_seconds is not None:
                self._feature_computation_duration_seconds.labels(
                    instrument=instrument,
                    feature_type=feature_type,
                    computation_mode=computation_mode,
                ).observe(computation_duration)

            # Features computed count
            if success and self._features_computed_total is not None:
                self._features_computed_total.labels(
                    instrument=instrument,
                    feature_type=feature_type,
                    computation_mode=computation_mode,
                ).inc(features_computed)

            # Update last computed timestamp
            if success and self._feature_last_computed_timestamp is not None:
                self._feature_last_computed_timestamp.labels(
                    instrument=instrument,
                    feature_type=feature_type,
                ).set(time.time())

            # Record errors
            if not success and error_type and self._feature_computation_errors_total is not None:
                self._feature_computation_errors_total.labels(
                    instrument=instrument,
                    feature_type=feature_type,
                    error_type=error_type,
                ).inc()

        self._safe_record("feature_computation", _record)

    def record_cache_hit(
        self,
        instrument: str,
        cache_level: str = "memory",
    ) -> None:
        """
        Record a feature cache hit.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        cache_level : str, default "memory"
            Level of cache hit ("memory", "disk").

        """

        def _record() -> None:
            if self._feature_cache_hits_total is not None:
                self._feature_cache_hits_total.labels(
                    instrument=instrument,
                    cache_level=cache_level,
                ).inc()

        self._safe_record("cache_hit", _record)

    def record_cache_miss(
        self,
        instrument: str,
        cache_level: str = "memory",
    ) -> None:
        """
        Record a feature cache miss.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        cache_level : str, default "memory"
            Level of cache miss ("memory", "disk").

        """

        def _record() -> None:
            if self._feature_cache_misses_total is not None:
                self._feature_cache_misses_total.labels(
                    instrument=instrument,
                    cache_level=cache_level,
                ).inc()

        self._safe_record("cache_miss", _record)

    def update_cache_stats(
        self,
        instrument: str,
        hit_ratio: float,
        cache_level: str = "memory",
        cache_size: int | None = None,
    ) -> None:
        """
        Update feature cache statistics.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        hit_ratio : float
            Cache hit ratio (0.0-1.0).
        cache_level : str, default "memory"
            Cache level ("memory", "disk").
        cache_size : int, optional
            Total cache size in entries.

        """

        def _record() -> None:
            # Hit ratio
            if self._feature_cache_hit_ratio is not None:
                self._feature_cache_hit_ratio.labels(
                    instrument=instrument,
                    cache_level=cache_level,
                ).set(max(0.0, min(1.0, hit_ratio)))

            # Cache size
            if cache_size is not None and self._feature_cache_size_entries is not None:
                self._feature_cache_size_entries.labels(
                    cache_level=cache_level,
                ).set(cache_size)

        self._safe_record("cache_stats", _record)

    def record_feature_drift(
        self,
        instrument: str,
        feature: str,
        drift_score: float,
        reference_window: str = "training",
        drift_threshold: float = 0.1,
        drift_type: str = "statistical",
    ) -> None:
        """
        Record feature drift metrics.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        feature : str
            Feature name.
        drift_score : float
            Drift score (e.g., KL divergence, PSI).
        reference_window : str, default "training"
            Reference window for drift comparison.
        drift_threshold : float, default 0.1
            Threshold for drift alerts.
        drift_type : str, default "statistical"
            Type of drift detected.

        """

        def _record() -> None:
            # Drift score
            if self._feature_drift_score is not None:
                self._feature_drift_score.labels(
                    instrument=instrument,
                    feature=feature,
                    reference_window=reference_window,
                ).set(drift_score)

            # Drift alerts
            if drift_score > drift_threshold and self._feature_drift_alerts_total is not None:
                self._feature_drift_alerts_total.labels(
                    instrument=instrument,
                    feature=feature,
                    drift_type=drift_type,
                ).inc()

        self._safe_record("feature_drift", _record)

    def record_feature_importance(
        self,
        model: str,
        feature_importances: dict[str, float],
    ) -> None:
        """
        Record feature importance scores from a model.

        Parameters
        ----------
        model : str
            Model identifier.
        feature_importances : Dict[str, float]
            Feature importance scores.

        """

        def _record() -> None:
            if self._feature_importance_score is not None:
                for feature, importance in feature_importances.items():
                    self._feature_importance_score.labels(
                        model=model,
                        feature=feature,
                    ).set(max(0.0, min(1.0, importance)))

        self._safe_record("feature_importance", _record)

    def record_feature_quality(
        self,
        instrument: str,
        feature_qualities: dict[str, dict[str, float]],
    ) -> None:
        """
        Record feature quality metrics.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        feature_qualities : Dict[str, Dict[str, float]]
            Quality metrics per feature (null_ratio, infinite_ratio, etc.).

        """

        def _record() -> None:
            for feature, qualities in feature_qualities.items():
                # Null ratio
                if "null_ratio" in qualities and self._feature_null_ratio is not None:
                    self._feature_null_ratio.labels(
                        instrument=instrument,
                        feature=feature,
                    ).set(max(0.0, min(1.0, qualities["null_ratio"])))

                # Infinite ratio
                if "infinite_ratio" in qualities and self._feature_infinite_ratio is not None:
                    self._feature_infinite_ratio.labels(
                        instrument=instrument,
                        feature=feature,
                    ).set(max(0.0, min(1.0, qualities["infinite_ratio"])))

        self._safe_record("feature_quality", _record)

    def time_feature_computation(
        self,
        instrument: str,
        feature_type: str,
        computation_mode: str = "batch",
    ) -> FeatureComputationTimer:
        """
        Create a context manager for timing feature computation.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        feature_type : str
            Type of features being computed.
        computation_mode : str, default "batch"
            Computation mode.

        Returns
        -------
        FeatureComputationTimer
            Context manager for timing feature computation operations.

        """
        return FeatureComputationTimer(self, instrument, feature_type, computation_mode)

    def get_feature_stats(
        self,
        instrument: str,
        feature_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Get feature engineering statistics for an instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        feature_type : str, optional
            Specific feature type to get stats for.

        Returns
        -------
        Dict[str, Any]
            Dictionary of feature statistics.

        """
        base_labels = {"instrument": instrument}
        if feature_type:
            base_labels["feature_type"] = feature_type

        stats = {
            "instrument": instrument,
            "cache_hit_ratio": self.get_metric_value(
                "feature_cache_hit_ratio",
                {"instrument": instrument, "cache_level": "memory"},
            ),
            "last_computed": self.get_metric_value("feature_last_computed_timestamp", base_labels),
        }

        if feature_type:
            stats["feature_type"] = feature_type

        return {k: v for k, v in stats.items() if v is not None}


class FeatureComputationTimer:
    """
    Context manager for timing feature computation operations.
    """

    def __init__(
        self,
        collector: FeatureEngineeringCollector,
        instrument: str,
        feature_type: str,
        computation_mode: str,
    ) -> None:
        """
        Initialize the feature computation timer.

        Parameters
        ----------
        collector : FeatureEngineeringCollector
            The collector to record metrics to.
        instrument : str
            Instrument identifier.
        feature_type : str
            Type of features being computed.
        computation_mode : str
            Mode of computation (batch, streaming, online).

        """
        self._collector = collector
        self._instrument = instrument
        self._feature_type = feature_type
        self._computation_mode = computation_mode
        self._start_time: float = 0.0
        self._features_computed: int = 0
        self._error_type: str | None = None
        self._feature_qualities: dict[str, dict[str, float]] = {}
        self._cache_hit: bool = False
        self._cache_level: str = "memory"

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

        # Record computation metrics
        self._collector.record_feature_computation(
            instrument=self._instrument,
            feature_type=self._feature_type,
            computation_duration=duration,
            features_computed=self._features_computed,
            computation_mode=self._computation_mode,
            success=success,
            error_type=self._error_type,
        )

        # Record cache metrics
        if self._cache_hit:
            self._collector.record_cache_hit(
                instrument=self._instrument,
                cache_level=self._cache_level,
            )
        else:
            self._collector.record_cache_miss(
                instrument=self._instrument,
                cache_level=self._cache_level,
            )

        # Record quality metrics if available
        if self._feature_qualities and success:
            self._collector.record_feature_quality(
                instrument=self._instrument,
                feature_qualities=self._feature_qualities,
            )

    def set_computation_result(
        self,
        features_computed: int,
        cache_hit: bool = False,
        cache_level: str = "memory",
        feature_qualities: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """
        Set results of the feature computation operation.

        Parameters
        ----------
        features_computed : int
            Number of features computed.
        cache_hit : bool, default False
            Whether this was a cache hit.
        cache_level : str, default "memory"
            Cache level used.
        feature_qualities : Dict[str, Dict[str, float]], optional
            Quality metrics per feature.

        """
        self._features_computed = features_computed
        self._cache_hit = cache_hit
        self._cache_level = cache_level
        if feature_qualities:
            self._feature_qualities = feature_qualities
