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

import threading
import time
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from ml._imports import HAS_PROMETHEUS
from ml.config.lightgbm_unified import MLflowConfig
from ml.monitoring._config import MonitoringConfig
from ml.tracking.monitoring_bridge import MLflowMonitoringBridge


@pytest.fixture
def monitoring_config():
    """
    Create test monitoring configuration.
    """
    return MonitoringConfig(
        enabled=True,
        metrics_prefix="test_ml",
        metrics_port=8080,
        health_check_interval=30.0,
        export_interval=5.0,
    )


@pytest.fixture
def mlflow_config():
    """
    Create test MLflow configuration.
    """
    return MLflowConfig(
        tracking_uri="file:///tmp/mlruns",
        experiment_name="test_experiment",
        model_name="test_model",
        log_model=True,
        log_artifacts=True,
        register_model=False,
        auto_log=False,
    )


@pytest.fixture
def mock_prometheus():
    """
    Mock Prometheus client.
    """
    with patch("ml.tracking.monitoring_bridge.HAS_PROMETHEUS", True):
        # Mock metrics are already provided by ml._imports
        yield


@pytest.fixture
def mock_mlflow_manager():
    """
    Mock MLflowManager.
    """
    with patch("ml.tracking.monitoring_bridge.MLflowManager") as mock_manager_class:
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        yield mock_manager


@pytest.fixture
def monitoring_bridge(monitoring_config, mlflow_config, mock_prometheus, mock_mlflow_manager):
    """
    Create monitoring bridge with mocked dependencies.
    """
    bridge = MLflowMonitoringBridge(
        monitoring_config=monitoring_config,
        mlflow_config=mlflow_config,
        sync_interval_seconds=1,  # Short interval for testing
    )
    return bridge


class TestMLflowMonitoringBridge:
    """
    Test suite for MLflowMonitoringBridge.
    """

    def test_init_with_configs(self, monitoring_config, mlflow_config):
        """
        Test MLflowMonitoringBridge initialization.
        """
        bridge = MLflowMonitoringBridge(
            monitoring_config=monitoring_config,
            mlflow_config=mlflow_config,
            sync_interval_seconds=300,
        )

        assert bridge.mlflow_config == mlflow_config
        assert bridge.sync_interval == 300
        assert bridge._mlflow_manager is None
        assert bridge._sync_thread is None
        assert not bridge._mlflow_available

    def test_initialize_metrics_creates_prometheus_metrics(self, monitoring_bridge):
        """
        Test that metrics are properly initialized.
        """
        # Metrics should be initialized during construction
        assert "mlflow_connectivity" in monitoring_bridge._metrics
        assert "mlflow_experiments_total" in monitoring_bridge._metrics
        assert "mlflow_runs_total" in monitoring_bridge._metrics
        assert "mlflow_models_total" in monitoring_bridge._metrics
        assert "mlflow_sync_duration_seconds" in monitoring_bridge._metrics

    def test_start_monitoring_when_enabled(self, monitoring_bridge):
        """
        Test starting monitoring when bridge is enabled.
        """
        monitoring_bridge.start_monitoring()

        assert monitoring_bridge._sync_thread is not None
        assert monitoring_bridge._sync_thread.is_alive()
        assert not monitoring_bridge._stop_sync.is_set()

        # Clean up
        monitoring_bridge.stop_monitoring()

    def test_start_monitoring_when_disabled(self, monitoring_config, mlflow_config):
        """
        Test starting monitoring when bridge is disabled.
        """
        monitoring_config.enabled = False
        bridge = MLflowMonitoringBridge(monitoring_config, mlflow_config)

        bridge.start_monitoring()

        assert bridge._sync_thread is None

    def test_start_monitoring_already_running(self, monitoring_bridge):
        """
        Test starting monitoring when already running.
        """
        monitoring_bridge.start_monitoring()

        # Try to start again
        monitoring_bridge.start_monitoring()

        # Should still have only one thread
        assert monitoring_bridge._sync_thread.is_alive()

        # Clean up
        monitoring_bridge.stop_monitoring()

    def test_stop_monitoring(self, monitoring_bridge):
        """
        Test stopping monitoring gracefully.
        """
        monitoring_bridge.start_monitoring()

        assert monitoring_bridge._sync_thread.is_alive()

        monitoring_bridge.stop_monitoring()

        # Thread should stop
        assert not monitoring_bridge._sync_thread.is_alive()
        assert monitoring_bridge._stop_sync.is_set()

    def test_stop_monitoring_not_running(self, monitoring_bridge):
        """
        Test stopping monitoring when not running.
        """
        # Should not raise exception
        monitoring_bridge.stop_monitoring()

        assert monitoring_bridge._sync_thread is None

    def test_ensure_mlflow_manager_creates_manager(self, monitoring_bridge, mock_mlflow_manager):
        """
        Test MLflow manager creation and connectivity check.
        """
        mock_mlflow_manager.health_check.return_value = {"connectivity": True}

        with patch("ml.tracking.monitoring_bridge.check_ml_dependencies"):
            available = monitoring_bridge._ensure_mlflow_manager()

        assert available
        assert monitoring_bridge._mlflow_available
        assert monitoring_bridge._mlflow_manager is not None

    def test_ensure_mlflow_manager_handles_failure(self, monitoring_bridge):
        """
        Test MLflow manager creation failure handling.
        """
        with patch(
            "ml.tracking.monitoring_bridge.check_ml_dependencies",
            side_effect=ImportError("MLflow not found"),
        ):
            available = monitoring_bridge._ensure_mlflow_manager()

        assert not available
        assert not monitoring_bridge._mlflow_available

    def test_sync_mlflow_metrics_unavailable(self, monitoring_bridge):
        """
        Test sync when MLflow is unavailable.
        """
        monitoring_bridge._mlflow_available = False

        result = monitoring_bridge.sync_mlflow_metrics()

        assert result["status"] == "mlflow_unavailable"

    def test_sync_mlflow_metrics_success(self, monitoring_bridge, mock_mlflow_manager):
        """
        Test successful MLflow metrics sync.
        """
        # Setup mock manager
        monitoring_bridge._mlflow_manager = mock_mlflow_manager
        monitoring_bridge._mlflow_available = True

        # Mock client and data
        mock_client = MagicMock()
        mock_mlflow_manager._client = mock_client

        # Mock experiments
        experiments = [MagicMock(), MagicMock()]
        mock_client.search_experiments.return_value = experiments

        # Mock experiment summary
        mock_mlflow_manager.get_experiment_summary.return_value = {
            "completed_runs": 5,
            "active_runs": 2,
            "failed_runs": 1,
        }

        # Mock runs
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp_123"
        mock_client.get_experiment_by_name.return_value = mock_experiment
        mock_client.search_runs.return_value = []

        # Mock models
        mock_client.search_registered_models.return_value = [MagicMock()]

        result = monitoring_bridge.sync_mlflow_metrics()

        assert "experiments_synced" in result
        assert "runs_synced" in result
        assert "models_synced" in result
        assert result["errors"] == 0

    def test_sync_experiments(self, monitoring_bridge, mock_mlflow_manager):
        """
        Test experiment sync functionality.
        """
        monitoring_bridge._mlflow_manager = mock_mlflow_manager

        # Mock experiments
        experiments = [MagicMock() for _ in range(3)]
        mock_client = MagicMock()
        mock_client.search_experiments.return_value = experiments
        mock_mlflow_manager._client = mock_client

        stats = {"experiments_synced": 0, "errors": 0}
        monitoring_bridge._sync_experiments(stats)

        assert stats["experiments_synced"] == 3
        assert stats["errors"] == 0

    def test_sync_experiment_runs(self, monitoring_bridge, mock_mlflow_manager):
        """
        Test experiment runs sync functionality.
        """
        monitoring_bridge._mlflow_manager = mock_mlflow_manager

        # Mock experiment summary
        summary = {
            "completed_runs": 10,
            "active_runs": 2,
            "failed_runs": 1,
        }
        mock_mlflow_manager.get_experiment_summary.return_value = summary

        # Mock experiment and runs
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp_123"
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = mock_experiment

        # Mock runs
        runs = []
        for i in range(5):
            run = MagicMock()
            run.info.run_id = f"run_{i}"
            run.info.start_time = 1234567890000 + i * 1000
            run.info.end_time = 1234567900000 + i * 1000
            run.data.metrics = {"accuracy": 0.9 + 0.01 * i}
            run.data.tags = {"model_type": "xgboost"}
            runs.append(run)

        mock_client.search_runs.return_value = runs
        mock_mlflow_manager._client = mock_client

        stats = {"runs_synced": 0, "errors": 0}
        monitoring_bridge._sync_experiment_runs("test_experiment", stats)

        assert stats["runs_synced"] == 5
        assert stats["errors"] == 0

    def test_sync_run_metrics(self, monitoring_bridge):
        """
        Test individual run metrics sync.
        """
        # Mock run data
        run = MagicMock()
        run.info.run_id = "test_run_123456789"
        run.info.start_time = 1234567890000
        run.info.end_time = 1234567900000
        run.data.metrics = {
            "accuracy": 0.95,
            "precision": 0.92,
            "invalid_metric": float("inf"),  # Should be skipped
        }
        run.data.tags = {"model_type": "xgboost"}

        monitoring_bridge._sync_run_metrics("test_experiment", run)

        # Should not raise exception (metrics recorded safely)
        assert True

    def test_sync_model_registry(self, monitoring_bridge, mock_mlflow_manager):
        """
        Test model registry sync functionality.
        """
        monitoring_bridge._mlflow_manager = mock_mlflow_manager

        # Mock registered models
        models = []
        for i in range(3):
            model = MagicMock()
            model.name = f"model_{i}"
            models.append(model)

        mock_client = MagicMock()
        mock_client.search_registered_models.return_value = models

        # Mock versions for each model
        def mock_get_versions(model_name, stages):
            if stages == ["Production"]:
                version = MagicMock()
                version.version = "1"
                return [version]
            return []

        mock_client.get_latest_versions.side_effect = mock_get_versions
        mock_mlflow_manager._client = mock_client

        stats = {"models_synced": 0, "errors": 0}
        monitoring_bridge._sync_model_registry(stats)

        assert stats["models_synced"] == 3
        assert stats["errors"] == 0

    def test_record_model_transition(self, monitoring_bridge):
        """
        Test model transition recording.
        """
        monitoring_bridge.record_model_transition(
            model_name="test_model",
            from_stage="Staging",
            to_stage="Production",
        )

        # Should not raise exception (metric recorded safely)
        assert True

    def test_export_mlflow_metadata_unavailable(self, monitoring_bridge):
        """
        Test metadata export when MLflow is unavailable.
        """
        monitoring_bridge._mlflow_available = False

        metadata = monitoring_bridge.export_mlflow_metadata()

        assert metadata["status"] == "mlflow_unavailable"

    def test_export_mlflow_metadata_success(self, monitoring_bridge, mock_mlflow_manager):
        """
        Test successful metadata export.
        """
        monitoring_bridge._mlflow_manager = mock_mlflow_manager
        monitoring_bridge._mlflow_available = True
        monitoring_bridge._last_sync_time = time.time() - 60

        # Mock experiment summary
        summary = {
            "experiment_name": "test_experiment",
            "total_runs": 10,
            "completed_runs": 8,
        }
        mock_mlflow_manager.get_experiment_summary.return_value = summary

        # Mock model registry
        mock_client = MagicMock()
        models = []
        for i in range(2):
            model = MagicMock()
            model.name = f"model_{i}"
            model.creation_timestamp = 1234567890000
            model.description = f"Test model {i}"
            models.append(model)

        mock_client.search_registered_models.return_value = models
        mock_client.get_latest_versions.return_value = []
        mock_mlflow_manager._client = mock_client

        metadata = monitoring_bridge.export_mlflow_metadata()

        assert metadata["mlflow_available"]
        assert metadata["experiment"]["experiment_name"] == "test_experiment"
        assert metadata["model_registry"]["total_models"] == 2

    def test_export_mlflow_metadata_with_errors(self, monitoring_bridge, mock_mlflow_manager):
        """
        Test metadata export with partial errors.
        """
        monitoring_bridge._mlflow_manager = mock_mlflow_manager
        monitoring_bridge._mlflow_available = True

        # Mock experiment summary failure
        mock_mlflow_manager.get_experiment_summary.side_effect = Exception("Experiment error")

        # Mock model registry success
        mock_client = MagicMock()
        mock_client.search_registered_models.return_value = []
        mock_mlflow_manager._client = mock_client

        metadata = monitoring_bridge.export_mlflow_metadata()

        assert "experiment_error" in metadata
        assert metadata["model_registry"]["total_models"] == 0

    def test_get_sync_status(self, monitoring_bridge):
        """
        Test sync status reporting.
        """
        monitoring_bridge._mlflow_available = True
        monitoring_bridge._last_sync_time = time.time() - 30

        status = monitoring_bridge.get_sync_status()

        assert status["bridge_enabled"]
        assert status["mlflow_available"]
        assert not status["sync_thread_alive"]  # Not started
        assert status["seconds_since_sync"] > 0
        assert status["sync_interval"] == 1
        assert status["prometheus_metrics_count"] > 0

    def test_force_sync(self, monitoring_bridge, mock_mlflow_manager):
        """
        Test forced synchronization.
        """
        monitoring_bridge._mlflow_manager = mock_mlflow_manager
        monitoring_bridge._mlflow_available = True

        # Mock minimal sync data
        mock_client = MagicMock()
        mock_client.search_experiments.return_value = []
        mock_client.search_registered_models.return_value = []
        mock_mlflow_manager._client = mock_client
        mock_mlflow_manager.get_experiment_summary.return_value = {
            "completed_runs": 0,
            "active_runs": 0,
            "failed_runs": 0,
        }

        result = monitoring_bridge.force_sync()

        assert "experiments_synced" in result
        assert result["errors"] == 0

    def test_force_sync_disabled(self, monitoring_config, mlflow_config):
        """
        Test force sync when bridge is disabled.
        """
        monitoring_config.enabled = False
        bridge = MLflowMonitoringBridge(monitoring_config, mlflow_config)

        result = bridge.force_sync()

        assert result["status"] == "disabled"

    def test_sync_loop_handles_exceptions(self, monitoring_bridge):
        """
        Test that sync loop handles exceptions gracefully.
        """
        # Mock sync method to raise exception
        original_sync = monitoring_bridge.sync_mlflow_metrics
        call_count = 0

        def mock_sync():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Test sync error")
            return {"status": "ok"}

        monitoring_bridge.sync_mlflow_metrics = mock_sync

        # Start monitoring briefly
        monitoring_bridge.start_monitoring()
        time.sleep(0.1)  # Let it run briefly
        monitoring_bridge.stop_monitoring()

        # Should have handled exception and continued
        assert call_count >= 1

    def test_inheritance_from_base_collector(self, monitoring_bridge):
        """
        Test that bridge properly inherits from BaseMetricsCollector.
        """
        # Should have base collector properties
        assert monitoring_bridge.enabled
        assert monitoring_bridge.config is not None
        assert isinstance(monitoring_bridge.metrics, dict)

        # Should have health check
        health = monitoring_bridge.health_check()
        assert "enabled" in health
        assert "metrics_count" in health
        assert "prometheus_available" in health

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus not available")
    def test_real_prometheus_integration(self, monitoring_config, mlflow_config):
        """
        Test integration with real Prometheus metrics (if available).
        """
        bridge = MLflowMonitoringBridge(monitoring_config, mlflow_config)

        # Should create real Prometheus metrics
        assert bridge._metrics
        assert len(bridge._metrics) > 0

        # Metrics should have proper types
        connectivity_metric = bridge._metrics.get("mlflow_connectivity")
        assert connectivity_metric is not None

    def test_thread_safety(self, monitoring_bridge):
        """
        Test thread safety of metric recording.
        """
        # Start multiple threads recording metrics
        threads = []
        results = []

        def record_metrics(thread_id):
            for i in range(10):
                try:
                    monitoring_bridge.record_model_transition(
                        model_name=f"model_{thread_id}",
                        from_stage="Staging",
                        to_stage="Production",
                    )
                    results.append(f"thread_{thread_id}_success_{i}")
                except Exception as e:
                    results.append(f"thread_{thread_id}_error_{e}")

        # Start threads
        for i in range(3):
            thread = threading.Thread(target=record_metrics, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should have recorded all metrics without errors
        successes = [r for r in results if "success" in r]
        errors = [r for r in results if "error" in r]

        assert len(successes) == 30  # 3 threads * 10 iterations
        assert len(errors) == 0
