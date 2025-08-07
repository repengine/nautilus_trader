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
Performance degradation monitoring collector for ML models.

This module provides comprehensive tracking of model performance over time, degradation
detection, and retraining alerts with Prometheus metrics.

"""

from __future__ import annotations

import time
from typing import Any

from ml._imports import HAS_PROMETHEUS
from ml._imports import Counter
from ml._imports import Gauge
from ml._imports import Histogram
from ml.monitoring._config import MonitoringConfig
from ml.monitoring.collectors.base import BaseMetricsCollector


class PerformanceDegradationMonitor(BaseMetricsCollector):
    """
    Monitor for model performance degradation and drift.

    This collector tracks model accuracy over time, detects performance
    degradation, monitors prediction distributions, and provides alerts
    for when models need retraining.

    Key Metrics
    -----------
    - Rolling accuracy and performance metrics
    - Prediction distribution shifts and drift
    - Inference timeout ratios and latency
    - Retraining alerts and triggers

    Parameters
    ----------
    config : MonitoringConfig
        Configuration for metrics collection.

    """

    def __init__(self, config: MonitoringConfig) -> None:
        """
        Initialize the performance degradation monitor.

        Parameters
        ----------
        config : MonitoringConfig
            Configuration for metrics collection.

        """
        super().__init__(config)

    def _initialize_metrics(self) -> None:
        """
        Initialize Prometheus metrics for performance monitoring.
        """
        if not HAS_PROMETHEUS:
            return

        prefix = self._config.metrics_prefix
        self._config.get_histogram_buckets()

        # Model accuracy and performance
        self._model_accuracy_rolling = Gauge(
            f"{prefix}_model_accuracy_rolling",
            "Rolling model accuracy over time windows",
            ["model", "window", "metric_type"],
        )
        self._register_metric("model_accuracy_rolling", self._model_accuracy_rolling)

        self._model_performance_score = Gauge(
            f"{prefix}_model_performance_score",
            "Current model performance score",
            ["model", "metric_type"],
        )
        self._register_metric("model_performance_score", self._model_performance_score)

        # Prediction distribution monitoring
        self._prediction_distribution_shift = Gauge(
            f"{prefix}_prediction_distribution_shift",
            "Distribution shift in model predictions",
            ["model", "shift_metric"],
        )
        self._register_metric("prediction_distribution_shift", self._prediction_distribution_shift)

        self._prediction_confidence_percentiles = Gauge(
            f"{prefix}_prediction_confidence_percentiles",
            "Percentiles of prediction confidence scores",
            ["model", "percentile"],
        )
        self._register_metric(
            "prediction_confidence_percentiles",
            self._prediction_confidence_percentiles,
        )

        # Inference performance
        self._inference_timeout_ratio = Gauge(
            f"{prefix}_inference_timeout_ratio",
            "Ratio of inference operations that timed out",
            ["model", "threshold_ms"],
        )
        self._register_metric("inference_timeout_ratio", self._inference_timeout_ratio)

        self._inference_latency_p99 = Gauge(
            f"{prefix}_inference_latency_p99",
            "99th percentile inference latency",
            ["model"],
        )
        self._register_metric("inference_latency_p99", self._inference_latency_p99)

        # Retraining and alerts
        self._model_retraining_required = Gauge(
            f"{prefix}_model_retraining_required",
            "Whether model requires retraining",
            ["model", "reason"],
        )
        self._register_metric("model_retraining_required", self._model_retraining_required)

        self._model_performance_alerts_total = Counter(
            f"{prefix}_model_performance_alerts_total",
            "Total number of performance alerts triggered",
            ["model", "alert_type"],
        )
        self._register_metric(
            "model_performance_alerts_total",
            self._model_performance_alerts_total,
        )

        # Prediction quality tracking
        self._predictions_evaluated_total = Counter(
            f"{prefix}_predictions_evaluated_total",
            "Total number of predictions evaluated",
            ["model", "result"],
        )
        self._register_metric("predictions_evaluated_total", self._predictions_evaluated_total)

        self._prediction_accuracy_window = Histogram(
            f"{prefix}_prediction_accuracy_window",
            "Accuracy within time windows",
            ["model", "window"],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )
        self._register_metric("prediction_accuracy_window", self._prediction_accuracy_window)

        # Model degradation tracking
        self._model_degradation_score = Gauge(
            f"{prefix}_model_degradation_score",
            "Overall model degradation score",
            ["model"],
        )
        self._register_metric("model_degradation_score", self._model_degradation_score)

        self._model_last_retrained_timestamp = Gauge(
            f"{prefix}_model_last_retrained_timestamp",
            "Timestamp when model was last retrained",
            ["model"],
        )
        self._register_metric(
            "model_last_retrained_timestamp",
            self._model_last_retrained_timestamp,
        )

    def record_model_performance(
        self,
        model: str,
        accuracy: float,
        metric_type: str = "accuracy",
        window: str = "1h",
        confidence_scores: list[float] | None = None,
    ) -> None:
        """
        Record model performance metrics.

        Parameters
        ----------
        model : str
            Model identifier.
        accuracy : float
            Model accuracy (0.0-1.0).
        metric_type : str, default "accuracy"
            Type of performance metric.
        window : str, default "1h"
            Time window for rolling metrics.
        confidence_scores : List[float], optional
            List of confidence scores for percentile calculation.

        """

        def _record() -> None:
            # Rolling accuracy
            if self._model_accuracy_rolling is not None:
                self._model_accuracy_rolling.labels(
                    model=model,
                    window=window,
                    metric_type=metric_type,
                ).set(max(0.0, min(1.0, accuracy)))

            # Current performance score
            if self._model_performance_score is not None:
                self._model_performance_score.labels(
                    model=model,
                    metric_type=metric_type,
                ).set(max(0.0, min(1.0, accuracy)))

            # Record in histogram
            if self._prediction_accuracy_window is not None:
                self._prediction_accuracy_window.labels(
                    model=model,
                    window=window,
                ).observe(accuracy)

            # Confidence percentiles
            if confidence_scores and self._prediction_confidence_percentiles is not None:
                import numpy as np

                percentiles = [50, 75, 90, 95, 99]
                confidence_percentiles = np.percentile(confidence_scores, percentiles)

                for p, value in zip(percentiles, confidence_percentiles):
                    self._prediction_confidence_percentiles.labels(
                        model=model,
                        percentile=f"p{p}",
                    ).set(value)

        self._safe_record("model_performance", _record)

    def record_prediction_evaluation(
        self,
        model: str,
        prediction_correct: bool,
        confidence: float,
        latency_ms: float | None = None,
    ) -> None:
        """
        Record individual prediction evaluation.

        Parameters
        ----------
        model : str
            Model identifier.
        prediction_correct : bool
            Whether the prediction was correct.
        confidence : float
            Prediction confidence score.
        latency_ms : float, optional
            Inference latency in milliseconds.

        """

        def _record() -> None:
            # Prediction result counter
            if self._predictions_evaluated_total is not None:
                result = "correct" if prediction_correct else "incorrect"
                self._predictions_evaluated_total.labels(
                    model=model,
                    result=result,
                ).inc()

            # Check for timeout if latency provided
            if latency_ms is not None:
                timeout_thresholds = [5.0, 10.0, 50.0]  # milliseconds
                for threshold in timeout_thresholds:
                    if latency_ms > threshold and self._inference_timeout_ratio is not None:
                        # This is a simplified approach - in practice you'd track ratios over time
                        current_ratio = self.get_metric_value(
                            "inference_timeout_ratio",
                            {"model": model, "threshold_ms": str(int(threshold))},
                        )
                        if current_ratio is None:
                            current_ratio = 0.0

                        # Simple exponential moving average update
                        alpha = 0.1  # Smoothing factor
                        new_ratio = alpha * 1.0 + (1 - alpha) * current_ratio

                        self._inference_timeout_ratio.labels(
                            model=model,
                            threshold_ms=str(int(threshold)),
                        ).set(min(1.0, new_ratio))

        self._safe_record("prediction_evaluation", _record)

    def record_distribution_shift(
        self,
        model: str,
        shift_score: float,
        shift_metric: str = "psi",
        threshold: float = 0.1,
    ) -> None:
        """
        Record prediction distribution shift.

        Parameters
        ----------
        model : str
            Model identifier.
        shift_score : float
            Distribution shift score.
        shift_metric : str, default "psi"
            Type of shift metric (psi, kl_divergence, wasserstein).
        threshold : float, default 0.1
            Threshold for triggering alerts.

        """

        def _record() -> None:
            # Record shift score
            if self._prediction_distribution_shift is not None:
                self._prediction_distribution_shift.labels(
                    model=model,
                    shift_metric=shift_metric,
                ).set(shift_score)

            # Trigger alert if threshold exceeded
            if shift_score > threshold and self._model_performance_alerts_total is not None:
                self._model_performance_alerts_total.labels(
                    model=model,
                    alert_type="distribution_shift",
                ).inc()

        self._safe_record("distribution_shift", _record)

    def update_degradation_score(
        self,
        model: str,
        degradation_score: float,
        retraining_threshold: float = 0.7,
    ) -> None:
        """
        Update overall model degradation score.

        Parameters
        ----------
        model : str
            Model identifier.
        degradation_score : float
            Overall degradation score (0.0-1.0, higher = more degraded).
        retraining_threshold : float, default 0.7
            Score threshold for triggering retraining alerts.

        """

        def _record() -> None:
            # Update degradation score
            if self._model_degradation_score is not None:
                self._model_degradation_score.labels(
                    model=model,
                ).set(max(0.0, min(1.0, degradation_score)))

            # Check if retraining is required
            if degradation_score > retraining_threshold:
                if self._model_retraining_required is not None:
                    self._model_retraining_required.labels(
                        model=model,
                        reason="performance",
                    ).set(1)

                if self._model_performance_alerts_total is not None:
                    self._model_performance_alerts_total.labels(
                        model=model,
                        alert_type="retraining_required",
                    ).inc()
            else:
                if self._model_retraining_required is not None:
                    self._model_retraining_required.labels(
                        model=model,
                        reason="performance",
                    ).set(0)

        self._safe_record("degradation_score", _record)

    def record_inference_latency_percentiles(
        self,
        model: str,
        latencies_ms: list[float],
    ) -> None:
        """
        Record inference latency percentiles.

        Parameters
        ----------
        model : str
            Model identifier.
        latencies_ms : List[float]
            List of latencies in milliseconds.

        """

        def _record() -> None:
            if not latencies_ms:
                return

            import numpy as np

            # Calculate P99 latency
            p99_latency = np.percentile(latencies_ms, 99)

            if self._inference_latency_p99 is not None:
                self._inference_latency_p99.labels(
                    model=model,
                ).set(float(p99_latency))

            # Update timeout ratios
            timeout_thresholds = [5.0, 10.0, 50.0]  # milliseconds
            for threshold in timeout_thresholds:
                timeout_count = sum(1 for latency in latencies_ms if latency > threshold)
                timeout_ratio = timeout_count / len(latencies_ms)

                if self._inference_timeout_ratio is not None:
                    self._inference_timeout_ratio.labels(
                        model=model,
                        threshold_ms=str(int(threshold)),
                    ).set(timeout_ratio)

        self._safe_record("inference_latency", _record)

    def trigger_retraining_alert(
        self,
        model: str,
        reason: str,
        alert_data: dict[str, Any] | None = None,
    ) -> None:
        """
        Manually trigger a retraining alert.

        Parameters
        ----------
        model : str
            Model identifier.
        reason : str
            Reason for retraining (drift, performance, schedule).
        alert_data : Dict[str, Any], optional
            Additional alert metadata.

        """

        def _record() -> None:
            # Set retraining required flag
            if self._model_retraining_required is not None:
                self._model_retraining_required.labels(
                    model=model,
                    reason=reason,
                ).set(1)

            # Increment alert counter
            if self._model_performance_alerts_total is not None:
                self._model_performance_alerts_total.labels(
                    model=model,
                    alert_type="retraining_required",
                ).inc()

        self._safe_record("retraining_alert", _record)

    def record_retraining_completion(
        self,
        model: str,
        retrain_timestamp: float | None = None,
    ) -> None:
        """
        Record completion of model retraining.

        Parameters
        ----------
        model : str
            Model identifier.
        retrain_timestamp : float, optional
            Unix timestamp of retraining. If None, uses current time.

        """
        if retrain_timestamp is None:
            retrain_timestamp = time.time()

        def _record() -> None:
            # Update last retrained timestamp
            if self._model_last_retrained_timestamp is not None:
                self._model_last_retrained_timestamp.labels(
                    model=model,
                ).set(retrain_timestamp)

            # Reset retraining required flags
            if self._model_retraining_required is not None:
                for reason in ["drift", "performance", "schedule"]:
                    self._model_retraining_required.labels(
                        model=model,
                        reason=reason,
                    ).set(0)

            # Reset degradation score
            if self._model_degradation_score is not None:
                self._model_degradation_score.labels(
                    model=model,
                ).set(0.0)

        self._safe_record("retraining_completion", _record)

    def get_performance_summary(
        self,
        model: str,
    ) -> dict[str, Any]:
        """
        Get comprehensive performance summary for a model.

        Parameters
        ----------
        model : str
            Model identifier.

        Returns
        -------
        Dict[str, Any]
            Performance summary metrics.

        """
        summary = {
            "model": model,
            "accuracy_1h": self.get_metric_value(
                "model_accuracy_rolling",
                {"model": model, "window": "1h", "metric_type": "accuracy"},
            ),
            "accuracy_24h": self.get_metric_value(
                "model_accuracy_rolling",
                {"model": model, "window": "24h", "metric_type": "accuracy"},
            ),
            "degradation_score": self.get_metric_value(
                "model_degradation_score",
                {"model": model},
            ),
            "retraining_required": self.get_metric_value(
                "model_retraining_required",
                {"model": model, "reason": "performance"},
            ),
            "p99_latency": self.get_metric_value(
                "inference_latency_p99",
                {"model": model},
            ),
            "last_retrained": self.get_metric_value(
                "model_last_retrained_timestamp",
                {"model": model},
            ),
        }

        return {k: v for k, v in summary.items() if v is not None}
