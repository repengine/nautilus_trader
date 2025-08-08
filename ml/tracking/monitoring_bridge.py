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
MLflow and Prometheus monitoring integration bridge.

This module provides integration between MLflow experiment tracking and Prometheus
monitoring, enabling synchronized observability across ML workflows and production
systems.

"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from ml._imports import Counter
from ml._imports import Gauge
from ml._imports import Histogram
from ml._imports import check_ml_dependencies
from ml.monitoring._config import MonitoringConfig
from ml.monitoring.collectors.base import BaseMetricsCollector
from ml.tracking.mlflow_manager import MLflowManager


if TYPE_CHECKING:
    from ml.config.shared import MLflowConfig

# Configure module logger
logger = logging.getLogger(__name__)


class MLflowMonitoringBridge(BaseMetricsCollector):
    """
    Bridge between MLflow tracking and Prometheus monitoring.

    This class synchronizes MLflow experiment metadata with Prometheus metrics,
    enabling unified observability across ML training, deployment, and production
    monitoring systems.

    Features:
    - Sync MLflow run metrics to Prometheus
    - Track model registry operations
    - Monitor experiment health and performance
    - Export MLflow metadata as Prometheus metrics
    - Real-time synchronization with configurable intervals
    - Graceful degradation when MLflow is unavailable

    Parameters
    ----------
    monitoring_config : MonitoringConfig
        Configuration for Prometheus monitoring.
    mlflow_config : MLflowConfig
        Configuration for MLflow integration.
    sync_interval_seconds : int, default 300
        Interval between MLflow sync operations (5 minutes).

    """

    def __init__(
        self,
        monitoring_config: MonitoringConfig,
        mlflow_config: MLflowConfig,
        sync_interval_seconds: int = 300,
    ) -> None:
        """
        Initialize MLflow monitoring bridge.

        Parameters
        ----------
        monitoring_config : MonitoringConfig
            Prometheus monitoring configuration.
        mlflow_config : MLflowConfig
            MLflow configuration.
        sync_interval_seconds : int, default 300
            Sync interval in seconds.

        """
        super().__init__(monitoring_config)

        self.mlflow_config = mlflow_config
        self.sync_interval = sync_interval_seconds
        self._mlflow_manager: MLflowManager | None = None
        self._sync_thread: threading.Thread | None = None
        self._stop_sync = threading.Event()
        self._last_sync_time = 0.0

        # MLflow availability tracking
        self._mlflow_available = False
        self._last_mlflow_check = 0.0
        self._mlflow_check_interval = 60.0  # Check every minute

        # Delta tracking for Prometheus counters
        # Structure: {(experiment_name, status): last_count}
        self._run_counter_states: dict[tuple[str, str], int] = {}

    def _initialize_metrics(self) -> None:
        """
        Initialize Prometheus metrics for MLflow monitoring.
        """
        prefix = self.config.metrics_prefix

        # MLflow connectivity metrics
        self._register_metric(
            "mlflow_connectivity",
            Gauge(
                f"{prefix}_mlflow_connectivity_status",
                "MLflow connectivity status (1=connected, 0=disconnected)",
            ),
        )

        # Experiment tracking metrics
        self._register_metric(
            "mlflow_experiments_total",
            Gauge(
                f"{prefix}_mlflow_experiments_total",
                "Total number of MLflow experiments",
            ),
        )

        self._register_metric(
            "mlflow_runs_total",
            Counter(
                f"{prefix}_mlflow_runs_total",
                "Total number of MLflow runs",
                labelnames=["experiment_name", "status"],
            ),
        )

        self._register_metric(
            "mlflow_runs_duration_seconds",
            Histogram(
                f"{prefix}_mlflow_runs_duration_seconds",
                "Duration of MLflow runs in seconds",
                labelnames=["experiment_name"],
                buckets=[1, 10, 30, 60, 300, 600, 1800, 3600, 7200],
            ),
        )

        # Model registry metrics
        self._register_metric(
            "mlflow_models_total",
            Gauge(
                f"{prefix}_mlflow_models_total",
                "Total number of registered models",
            ),
        )

        self._register_metric(
            "mlflow_model_versions_total",
            Gauge(
                f"{prefix}_mlflow_model_versions_total",
                "Total number of model versions",
                labelnames=["model_name", "stage"],
            ),
        )

        self._register_metric(
            "mlflow_model_transitions_total",
            Counter(
                f"{prefix}_mlflow_model_transitions_total",
                "Total number of model stage transitions",
                labelnames=["model_name", "from_stage", "to_stage"],
            ),
        )

        # Performance metrics from MLflow runs
        self._register_metric(
            "mlflow_run_metrics",
            Gauge(
                f"{prefix}_mlflow_run_metrics",
                "Metrics from MLflow runs",
                labelnames=[
                    "experiment_name",
                    "run_id",
                    "metric_name",
                    "model_type",
                ],
            ),
        )

        # Sync operation metrics
        self._register_metric(
            "mlflow_sync_duration_seconds",
            Histogram(
                f"{prefix}_mlflow_sync_duration_seconds",
                "Duration of MLflow sync operations",
                buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
            ),
        )

        self._register_metric(
            "mlflow_sync_errors_total",
            Counter(
                f"{prefix}_mlflow_sync_errors_total",
                "Total number of MLflow sync errors",
                labelnames=["error_type"],
            ),
        )

        self._register_metric(
            "mlflow_last_sync_timestamp",
            Gauge(
                f"{prefix}_mlflow_last_sync_timestamp",
                "Timestamp of last successful MLflow sync",
            ),
        )

    def start_monitoring(self) -> None:
        """
        Start background monitoring and sync operations.
        """
        if not self._enabled:
            logger.info("MLflow monitoring bridge is disabled")
            return

        if self._sync_thread and self._sync_thread.is_alive():
            logger.info("MLflow monitoring is already running")
            return

        logger.info(f"MLflow monitoring bridge (sync interval: {self.sync_interval}s)")

        self._stop_sync.clear()
        self._sync_thread = threading.Thread(
            target=self._sync_loop,
            name="MLflowMonitoringBridge",
            daemon=True,
        )
        self._sync_thread.start()

    def stop_monitoring(self) -> None:
        """
        Stop background monitoring operations.
        """
        if self._sync_thread and self._sync_thread.is_alive():
            logger.info("Stopping MLflow monitoring bridge...")
            self._stop_sync.set()
            self._sync_thread.join(timeout=10)

            if self._sync_thread.is_alive():
                logger.warning(f"Failed to sync run metrics: {e}")
            else:
                logger.info("MLflow monitoring bridge stopped")

    def _sync_loop(self) -> None:
        """
        Main sync loop running in background thread.
        """
        while not self._stop_sync.wait(self.sync_interval):
            try:
                self.sync_mlflow_metrics()
            except Exception as sync_error:
                # Create a local copy for the lambda
                error_type_name = type(sync_error).__name__
                self._safe_record(
                    "sync_error",
                    lambda: self._metrics["mlflow_sync_errors_total"]
                    .labels(error_type=error_type_name)
                    .inc(),
                )
                logger.info(f"Error in MLflow sync: {sync_error}")

    def _ensure_mlflow_manager(self) -> bool:
        """
        Ensure MLflow manager is available and connected.

        Returns
        -------
        bool
            True if MLflow manager is available.

        """
        current_time = time.time()

        # Check MLflow availability periodically
        if current_time - self._last_mlflow_check > self._mlflow_check_interval:
            self._last_mlflow_check = current_time

            try:
                if self._mlflow_manager is None:
                    check_ml_dependencies(["mlflow"])
                    self._mlflow_manager = MLflowManager(self.mlflow_config)

                # Test connectivity
                health = self._mlflow_manager.health_check()
                self._mlflow_available = health.get("connectivity", False)

                # Update connectivity metric
                self._safe_record(
                    "connectivity_update",
                    lambda: self._metrics["mlflow_connectivity"].set(
                        1.0 if self._mlflow_available else 0.0,
                    ),
                )

            except Exception as e:
                self._mlflow_available = False
                self._safe_record(
                    "connectivity_error",
                    lambda: self._metrics["mlflow_connectivity"].set(0.0),
                )
                logger.info(f"MLflow connectivity check failed: {e}")

        return self._mlflow_available

    def sync_mlflow_metrics(self) -> dict[str, Any]:
        """
        Synchronize MLflow metrics with Prometheus.

        Returns
        -------
        dict[str, Any]
            Sync operation results and statistics.

        """
        if not self._ensure_mlflow_manager():
            return {"status": "mlflow_unavailable"}

        sync_start = time.time()
        stats = {
            "experiments_synced": 0,
            "runs_synced": 0,
            "models_synced": 0,
            "errors": 0,
        }

        try:
            # Sync experiment metrics
            self._sync_experiments(stats)

            # Sync run metrics for configured experiment
            if self.mlflow_config.experiment_name:
                self._sync_experiment_runs(self.mlflow_config.experiment_name, stats)

            # Sync model registry metrics
            self._sync_model_registry(stats)

            # Update sync timestamp
            self._last_sync_time = time.time()
            self._safe_record(
                "sync_timestamp_update",
                lambda: self._metrics["mlflow_last_sync_timestamp"].set(self._last_sync_time),
            )

        except Exception as e:
            stats["errors"] += 1
            # Create a local copy for the lambda
            error_type_name = type(e).__name__
            self._safe_record(
                "sync_error",
                lambda: self._metrics["mlflow_sync_errors_total"]
                .labels(error_type=error_type_name)
                .inc(),
            )
            raise
        finally:
            # Record sync duration
            sync_duration = time.time() - sync_start
            self._safe_record(
                "sync_duration",
                lambda: self._metrics["mlflow_sync_duration_seconds"].observe(sync_duration),
            )

        return stats

    def _sync_experiments(self, stats: dict[str, Any]) -> None:
        """
        Sync experiment count metrics.
        """
        try:
            if not self._mlflow_manager:
                return

            # Get experiments from MLflow client directly
            client = self._mlflow_manager._client
            experiments = client.search_experiments()

            self._safe_record(
                "experiments_total_update",
                lambda: self._metrics["mlflow_experiments_total"].set(len(experiments)),
            )

            stats["experiments_synced"] = len(experiments)

        except Exception as e:
            stats["errors"] += 1
            logger.info(f"Error syncing experiments: {e}")

    def _sync_experiment_runs(self, experiment_name: str, stats: dict[str, Any]) -> None:
        """
        Sync run metrics for a specific experiment.
        """
        try:
            if not self._mlflow_manager:
                return

            # Get experiment summary
            summary = self._mlflow_manager.get_experiment_summary(experiment_name)

            # Update run counts by status
            for status in ["completed", "active", "failed"]:
                count_key = f"{status}_runs"
                if count_key in summary:
                    # Note: Counter metrics can't be set to arbitrary values,
                    # so we track the delta since last sync
                    count = summary[count_key]

                    # Type hint the lambda parameters
                    def _update_counter(c: int = count, s: str = status) -> None:
                        self._update_run_counter(experiment_name, s, c)

                    self._safe_record(f"runs_{status}_update", _update_counter)

            # Sync recent run metrics
            client = self._mlflow_manager._client
            experiment = client.get_experiment_by_name(experiment_name)

            if experiment:
                # Get recent runs (last 100)
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["attribute.start_time DESC"],
                    max_results=100,
                )

                for run in runs:
                    self._sync_run_metrics(experiment_name, run)
                    stats["runs_synced"] += 1

        except Exception as e:
            stats["errors"] += 1
            logger.info(f"Error syncing experiment runs: {e}")

    def _update_run_counter(self, experiment_name: str, status: str, count: int) -> None:
        """
        Update run counter metric with proper delta tracking.

        Prometheus counters can only increment, never decrease. This method tracks
        the last known count for each experiment/status combination and only
        increments by the delta since the last sync.

        Parameters
        ----------
        experiment_name : str
            Name of the MLflow experiment.
        status : str
            Run status (completed, active, failed).
        count : int
            Current total count from MLflow.

        """
        # Use lock for thread safety when accessing shared state
        with self._lock:
            # Create a unique key for this counter
            counter_key = (experiment_name, status)

            # Get the last known count for this counter
            last_count = self._run_counter_states.get(counter_key, 0)

            # Calculate the delta since last sync
            delta = count - last_count

            # Only increment if there's a positive delta
            if delta > 0:
                # Increment the counter by the delta
                self._metrics["mlflow_runs_total"].labels(
                    experiment_name=experiment_name,
                    status=status,
                ).inc(delta)

                # Update the state for next sync
                self._run_counter_states[counter_key] = count

            elif delta < 0:
                # Log a warning if count decreased (shouldn't happen in normal operation)
                logger.info(
                    f"Warning: Run count decreased for {experiment_name}/{status}: "
                    f"previous={last_count}, current={count}. "
                    f"This may indicate data inconsistency in MLflow.",
                )
                # Don't update the counter (Prometheus counters can't decrease)
                # But update our state to match reality to avoid accumulating errors
                self._run_counter_states[counter_key] = count

            # If delta == 0, no action needed (count hasn't changed)

    def _sync_run_metrics(self, experiment_name: str, run: Any) -> None:
        """
        Sync individual run metrics to Prometheus.
        """
        try:
            run_info = run.info
            run_data = run.data

            # Determine model type from tags or parameters
            model_type = "unknown"
            if run_data.tags:
                model_type = run_data.tags.get("model_type", "unknown")

            # Sync key metrics from the run
            for metric_name, metric_value in run_data.metrics.items():
                if isinstance(metric_value, int | float) and abs(metric_value) < 1e10:
                    self._safe_record(
                        f"run_metric_{metric_name}",
                        lambda: self._metrics["mlflow_run_metrics"]
                        .labels(
                            experiment_name=experiment_name,
                            run_id=run_info.run_id[:8],  # Short ID
                            metric_name=metric_name,
                            model_type=model_type,
                        )
                        .set(metric_value),
                    )

            # Record run duration if available
            if run_info.start_time and run_info.end_time:
                duration = (run_info.end_time - run_info.start_time) / 1000.0  # ms to s
                self._safe_record(
                    "run_duration",
                    lambda: self._metrics["mlflow_runs_duration_seconds"]
                    .labels(experiment_name=experiment_name)
                    .observe(duration),
                )

        except Exception as e:
            logger.info(f"Error syncing run metrics for {run.info.run_id}: {e}")

    def _sync_model_registry(self, stats: dict[str, Any]) -> None:
        """
        Sync model registry metrics.
        """
        try:
            if not self._mlflow_manager:
                return

            client = self._mlflow_manager._client

            # Get all registered models
            models = client.search_registered_models()

            self._safe_record(
                "models_total_update",
                lambda: self._metrics["mlflow_models_total"].set(len(models)),
            )

            # Count versions by stage for each model
            stage_counts = {}
            for model in models:
                model_name = model.name

                # Get latest versions for each stage
                try:
                    for stage in ["None", "Staging", "Production", "Archived"]:
                        versions = client.get_latest_versions(model_name, stages=[stage])
                        count = len(versions)

                        if count > 0:
                            key = (model_name, stage)
                            stage_counts[key] = count

                except Exception as e:
                    logger.info(f"Error getting versions for model {model_name}: {e}")
                    continue

            # Update version count metrics
            for (model_name, stage), count in stage_counts.items():
                # Type hint the lambda parameters
                def _set_metric(m: str = model_name, s: str = stage, c: int = count) -> None:
                    self._metrics["mlflow_model_versions_total"].labels(model_name=m, stage=s).set(
                        c,
                    )

                self._safe_record(f"model_versions_{model_name}_{stage}", _set_metric)

            stats["models_synced"] = len(models)

        except Exception as e:
            stats["errors"] += 1
            logger.info(f"Error syncing model registry: {e}")

    def record_model_transition(
        self,
        model_name: str,
        from_stage: str,
        to_stage: str,
    ) -> None:
        """
        Record a model stage transition event.

        Parameters
        ----------
        model_name : str
            Name of the model.
        from_stage : str
            Source stage.
        to_stage : str
            Target stage.

        """
        self._safe_record(
            "model_transition",
            lambda: self._metrics["mlflow_model_transitions_total"]
            .labels(
                model_name=model_name,
                from_stage=from_stage,
                to_stage=to_stage,
            )
            .inc(),
        )

    def export_mlflow_metadata(self) -> dict[str, Any]:
        """
        Export current MLflow metadata for external monitoring systems.

        Returns
        -------
        dict[str, Any]
            Comprehensive MLflow metadata export.

        """
        if not self._ensure_mlflow_manager():
            return {"status": "mlflow_unavailable"}

        try:
            metadata = {
                "timestamp": time.time(),
                "mlflow_available": self._mlflow_available,
                "last_sync": self._last_sync_time,
                "tracking_uri": self.mlflow_config.tracking_uri,
            }

            # Get experiment info
            if self.mlflow_config.experiment_name and self._mlflow_manager is not None:
                try:
                    summary = self._mlflow_manager.get_experiment_summary(
                        self.mlflow_config.experiment_name,
                    )
                    metadata["experiment"] = summary
                except Exception as e:
                    metadata["experiment_error"] = str(e)

            # Get model registry summary
            if self._mlflow_manager is not None:
                try:
                    client = self._mlflow_manager._client
                    models = client.search_registered_models()

                    model_summary = {
                        "total_models": len(models),
                        "models": [],
                    }

                    for model in models[:10]:  # Limit to prevent huge exports
                        model_info = {
                            "name": model.name,
                            "creation_timestamp": model.creation_timestamp,
                            "description": model.description,
                        }

                        # Get stage info
                        try:
                            versions = client.get_latest_versions(model.name)
                            model_info["stages"] = {v.current_stage: v.version for v in versions}
                        except Exception:
                            pass

                        # Ensure models list exists and is properly typed
                        if "models" not in model_summary:
                            model_summary["models"] = []
                        models_list = model_summary["models"]
                        if isinstance(models_list, list):
                            models_list.append(model_info)

                    metadata["model_registry"] = model_summary

                except Exception as e:
                    metadata["model_registry_error"] = str(e)

            return metadata

        except Exception as e:
            return {"status": "export_error", "error": str(e)}

    def get_sync_status(self) -> dict[str, Any]:
        """
        Get current sync status and health information.

        Returns
        -------
        dict[str, Any]
            Sync status information.

        """
        current_time = time.time()

        with self._lock:
            counter_states_count = len(self._run_counter_states)

        return {
            "bridge_enabled": self._enabled,
            "mlflow_available": self._mlflow_available,
            "sync_thread_alive": (self._sync_thread.is_alive() if self._sync_thread else False),
            "last_sync_timestamp": self._last_sync_time,
            "seconds_since_sync": current_time - self._last_sync_time,
            "sync_interval": self.sync_interval,
            "next_sync_in": self.sync_interval - (current_time % self.sync_interval),
            "prometheus_metrics_count": len(self._metrics),
            "tracked_counter_states": counter_states_count,
        }

    def reset_counter_states(self) -> None:
        """
        Reset all counter state tracking.

        This method clears the internal state tracking for Prometheus counters.
        Use with caution as it may cause temporary inconsistencies in metrics
        until the next sync establishes new baselines.

        This is primarily useful for:
        - Recovery from corrupted state
        - Testing scenarios
        - Re-initialization after major changes

        """
        with self._lock:
            old_count = len(self._run_counter_states)
            self._run_counter_states.clear()
            logger.info(f"Reset {old_count} counter states for MLflow run tracking")

    def force_sync(self) -> dict[str, Any]:
        """
        Force immediate synchronization of MLflow metrics.

        Returns
        -------
        dict[str, Any]
            Sync operation results.

        """
        if not self._enabled:
            return {"status": "disabled"}

        logger.info("Forcing MLflow metrics sync...")
        return self.sync_mlflow_metrics()
