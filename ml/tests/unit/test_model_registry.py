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

import time
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from ml._imports import HAS_MLFLOW
from ml.config.shared import MLflowConfig
from ml.tracking.model_registry import ModelRegistry


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
def mock_mlflow():
    """
    Mock MLflow module and dependencies.
    """
    # Mock multiple places where mlflow and dependencies are imported
    with (
        patch("ml.tracking.mlflow_manager.mlflow") as mock_mlflow,
        patch("ml.tracking.mlflow_manager.HAS_MLFLOW", True),
        patch("ml.tracking.mlflow_manager.check_ml_dependencies") as mock_check_deps,
    ):

        # Setup basic mock structure
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"

        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp_123"
        mock_experiment.name = "test_experiment"

        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        mock_mlflow.active_run.return_value = mock_run

        # Mock client
        mock_client = MagicMock()
        mock_mlflow.tracking.MlflowClient.return_value = mock_client

        # Mock dependency check to do nothing
        mock_check_deps.return_value = None

        yield mock_mlflow


@pytest.fixture
def model_registry(mlflow_config, mock_mlflow):
    """
    Create model registry with mocked dependencies.
    """
    registry = ModelRegistry(mlflow_config)
    return registry


class TestModelRegistry:
    """
    Test suite for ModelRegistry.
    """

    @pytest.mark.skipif(not HAS_MLFLOW, reason="MLflow not available")
    def test_init_extends_mlflow_manager(self, mlflow_config) -> None:
        """
        Test ModelRegistry initialization.
        """
        registry = ModelRegistry(mlflow_config)

        assert registry.config == mlflow_config
        assert isinstance(registry._ab_tests, dict)
        assert isinstance(registry._canary_deployments, dict)
        assert isinstance(registry._performance_history, dict)

    def test_setup_ab_test_with_versions(self, model_registry, mock_mlflow) -> None:
        """
        Test A/B test setup with specific versions.
        """
        mock_client = mock_mlflow.tracking.MlflowClient.return_value

        test_id = model_registry.setup_ab_test(
            test_name="accuracy_test",
            model_a_name="model_a",
            model_b_name="model_b",
            model_a_version="1",
            model_b_version="2",
            traffic_split=0.3,
            success_metric="accuracy",
        )

        assert test_id.startswith("ab_test_accuracy_test_")
        assert test_id in model_registry._ab_tests

        test_config = model_registry._ab_tests[test_id]
        assert test_config["status"] == "active"
        assert test_config["config"]["traffic_split"] == 0.3
        assert test_config["model_a"]["name"] == "model_a"
        assert test_config["model_b"]["name"] == "model_b"

        # Should tag models with test info
        assert mock_client.set_model_version_tag.call_count == 4

    def test_setup_ab_test_auto_versions(self, model_registry, mock_mlflow) -> None:
        """
        Test A/B test setup with automatic version detection.
        """
        mock_client = mock_mlflow.tracking.MlflowClient.return_value

        # Mock version responses
        prod_version = MagicMock()
        prod_version.version = "1"
        staging_version = MagicMock()
        staging_version.version = "2"

        mock_client.get_latest_versions.side_effect = [
            [prod_version],  # Production version for model A
            [staging_version],  # Staging version for model B
        ]

        test_id = model_registry.setup_ab_test(
            test_name="auto_version_test",
            model_a_name="model_a",
            model_b_name="model_b",
        )

        assert test_id in model_registry._ab_tests
        test_config = model_registry._ab_tests[test_id]
        assert test_config["model_a"]["version"] == "1"
        assert test_config["model_b"]["version"] == "2"

    def test_setup_ab_test_missing_versions(self, model_registry, mock_mlflow) -> None:
        """
        Test A/B test setup fails with missing versions.
        """
        mock_client = mock_mlflow.tracking.MlflowClient.return_value
        mock_client.get_latest_versions.return_value = []  # No versions found

        with pytest.raises(ValueError, match="Could not get production version"):
            model_registry.setup_ab_test(
                test_name="missing_test",
                model_a_name="model_a",
                model_b_name="model_b",
            )

    def test_record_ab_test_result(self, model_registry, mock_mlflow) -> None:
        """
        Test recording A/B test results.
        """
        # Setup test first
        test_id = model_registry.setup_ab_test(
            test_name="result_test",
            model_a_name="model_a",
            model_b_name="model_b",
            model_a_version="1",
            model_b_version="2",
        )

        # Record results for model A
        status = model_registry.record_ab_test_result(
            test_id=test_id,
            model_name="model_a",
            metric_value=0.85,
            sample_count=10,
        )

        assert status["test_id"] == test_id
        assert status["model_a_samples"] == 10
        assert status["model_a_mean"] == 0.85
        assert status["status"] == "active"  # Should still be active

    def test_record_ab_test_result_invalid_test(self, model_registry) -> None:
        """
        Test recording result for non-existent test.
        """
        with pytest.raises(ValueError, match="A/B test invalid_test not found"):
            model_registry.record_ab_test_result(
                test_id="invalid_test",
                model_name="model_a",
                metric_value=0.85,
            )

    def test_record_ab_test_result_invalid_model(self, model_registry, mock_mlflow) -> None:
        """
        Test recording result for model not in test.
        """
        test_id = model_registry.setup_ab_test(
            test_name="invalid_model_test",
            model_a_name="model_a",
            model_b_name="model_b",
            model_a_version="1",
            model_b_version="2",
        )

        with pytest.raises(ValueError, match="Model invalid_model is not part of A/B test"):
            model_registry.record_ab_test_result(
                test_id=test_id,
                model_name="invalid_model",
                metric_value=0.85,
            )

    def test_ab_test_conclusion_by_duration(self, model_registry, mock_mlflow) -> None:
        """
        Test A/B test conclusion by maximum duration.
        """
        test_id = model_registry.setup_ab_test(
            test_name="duration_test",
            model_a_name="model_a",
            model_b_name="model_b",
            model_a_version="1",
            model_b_version="2",
            max_duration_hours=1,  # Short duration for testing
        )

        # Manually set start time to simulate long duration
        model_registry._ab_tests[test_id]["created_at"] = time.time() - 3700  # > 1 hour ago

        status = model_registry.record_ab_test_result(
            test_id=test_id,
            model_name="model_a",
            metric_value=0.85,
        )

        assert status["status"] == "concluded"
        assert status["conclusion_reason"] == "max_duration_reached"

    def test_ab_test_statistical_significance(self, model_registry, mock_mlflow) -> None:
        """
        Test A/B test conclusion by statistical significance.
        """
        test_id = model_registry.setup_ab_test(
            test_name="significance_test",
            model_a_name="model_a",
            model_b_name="model_b",
            model_a_version="1",
            model_b_version="2",
            min_samples=10,
        )

        # Record many results for both models
        for i in range(110):
            # Model A consistently lower
            model_registry.record_ab_test_result(
                test_id=test_id,
                model_name="model_a",
                metric_value=0.80 + 0.01 * (i % 5),  # 0.80-0.84 range
            )

            # Model B consistently higher
            model_registry.record_ab_test_result(
                test_id=test_id,
                model_name="model_b",
                metric_value=0.90 + 0.01 * (i % 5),  # 0.90-0.94 range
            )

        # Check final status - might conclude due to statistical significance
        test_config = model_registry._ab_tests[test_id]
        final_status = model_registry._check_ab_test_conclusion(test_id)

        assert final_status["model_b_mean"] > final_status["model_a_mean"]
        assert final_status["difference"] > 0

    def test_setup_canary_deployment(self, model_registry, mock_mlflow) -> None:
        """
        Test canary deployment setup.
        """
        mock_client = mock_mlflow.tracking.MlflowClient.return_value

        # Mock production model for baseline
        prod_version = MagicMock()
        prod_version.run_id = "prod_run_123"
        prod_run = MagicMock()
        prod_run.data.metrics = {"accuracy": 0.90}

        mock_client.get_latest_versions.return_value = [prod_version]
        mock_client.get_run.return_value = prod_run

        deployment_id = model_registry.setup_canary_deployment(
            deployment_name="canary_test",
            model_name="test_model",
            model_version="2",
            traffic_percentage=10.0,
            success_metric="accuracy",
            baseline_threshold=0.95,
        )

        assert deployment_id.startswith("canary_canary_test_")
        assert deployment_id in model_registry._canary_deployments

        deployment = model_registry._canary_deployments[deployment_id]
        assert deployment["status"] == "active"
        assert deployment["config"]["traffic_percentage"] == 10.0
        assert deployment["baseline"]["performance"] == 0.90

        # Should tag model with canary info
        mock_client.set_model_version_tag.assert_called()

    def test_record_canary_metrics(self, model_registry, mock_mlflow) -> None:
        """
        Test recording canary deployment metrics.
        """
        deployment_id = model_registry.setup_canary_deployment(
            deployment_name="metrics_test",
            model_name="test_model",
            model_version="2",
        )

        # Record successful metrics
        status = model_registry.record_canary_metrics(
            deployment_id=deployment_id,
            metric_value=0.95,
            latency_ms=50.0,
            error_occurred=False,
        )

        assert status["deployment_id"] == deployment_id
        assert status["sample_count"] == 1
        assert status["success_count"] == 1
        assert status["error_count"] == 0
        assert status["current_performance"] == 0.95

    def test_record_canary_metrics_with_errors(self, model_registry, mock_mlflow) -> None:
        """
        Test recording canary metrics with errors.
        """
        deployment_id = model_registry.setup_canary_deployment(
            deployment_name="error_test",
            model_name="test_model",
            model_version="2",
        )

        # Record error
        status = model_registry.record_canary_metrics(
            deployment_id=deployment_id,
            metric_value=0.0,  # This will be ignored due to error
            error_occurred=True,
        )

        assert status["error_count"] == 1
        assert status["success_count"] == 0
        assert status["current_performance"] == 0.0

    def test_canary_rollback_on_performance(self, model_registry, mock_mlflow) -> None:
        """
        Test canary rollback due to poor performance.
        """
        deployment_id = model_registry.setup_canary_deployment(
            deployment_name="rollback_test",
            model_name="test_model",
            model_version="2",
            baseline_threshold=0.95,
        )

        # Set baseline performance
        deployment = model_registry._canary_deployments[deployment_id]
        deployment["baseline"]["performance"] = 0.90

        # Record many poor performance samples
        for _ in range(150):  # Exceed minimum for decision
            model_registry.record_canary_metrics(
                deployment_id=deployment_id,
                metric_value=0.80,  # Below threshold (0.80 < 0.90 * 0.95)
                error_occurred=False,
            )

        # Check final status
        status = model_registry._check_canary_status(deployment_id)

        assert status["status"] == "rolled_back"
        assert status["decision_reason"] == "performance_degradation"

    def test_canary_rollback_on_error_rate(self, model_registry, mock_mlflow) -> None:
        """
        Test canary rollback due to high error rate.
        """
        deployment_id = model_registry.setup_canary_deployment(
            deployment_name="error_rollback_test",
            model_name="test_model",
            model_version="2",
        )

        # Record high error rate
        for i in range(150):
            error = i < 10  # 10 errors out of 150 = 6.7% error rate
            model_registry.record_canary_metrics(
                deployment_id=deployment_id,
                metric_value=0.95 if not error else 0.0,
                error_occurred=error,
            )

        status = model_registry._check_canary_status(deployment_id)

        assert status["status"] == "rolled_back"
        assert status["decision_reason"] == "high_error_rate"

    def test_canary_promotion(self, model_registry, mock_mlflow) -> None:
        """
        Test canary promotion to production.
        """
        deployment_id = model_registry.setup_canary_deployment(
            deployment_name="promotion_test",
            model_name="test_model",
            model_version="2",
            monitoring_duration_hours=0,  # No duration requirement
            baseline_threshold=0.90,
        )

        # Set baseline
        deployment = model_registry._canary_deployments[deployment_id]
        deployment["baseline"]["performance"] = 0.85

        # Record excellent performance
        for _ in range(600):  # Exceed minimum samples
            model_registry.record_canary_metrics(
                deployment_id=deployment_id,
                metric_value=0.95,  # Well above baseline
                error_occurred=False,
            )

        status = model_registry._check_canary_status(deployment_id)

        assert status["status"] == "promoted"
        assert status["decision_reason"] == "successful_monitoring_period"

    def test_rollback_model(self, model_registry, mock_mlflow) -> None:
        """
        Test model rollback functionality.
        """
        mock_client = mock_mlflow.tracking.MlflowClient.return_value

        # Mock current production version
        current_version = MagicMock()
        current_version.version = "2"
        mock_client.get_latest_versions.return_value = [current_version]

        # Mock previous versions
        old_version = MagicMock()
        old_version.version = "1"
        old_version.tags = {"previous_production": "true"}
        mock_client.search_model_versions.return_value = [current_version, old_version]

        result = model_registry.rollback_model(
            model_name="test_model",
            reason="performance_degradation",
        )

        assert result["model_name"] == "test_model"
        assert result["from_version"] == "2"
        assert result["to_version"] == "1"
        assert result["reason"] == "performance_degradation"

        # Should tag and transition stages
        mock_client.set_model_version_tag.assert_called()

    def test_rollback_model_specific_version(self, model_registry, mock_mlflow) -> None:
        """
        Test rollback to specific version.
        """
        mock_client = mock_mlflow.tracking.MlflowClient.return_value

        current_version = MagicMock()
        current_version.version = "3"
        mock_client.get_latest_versions.return_value = [current_version]

        result = model_registry.rollback_model(
            model_name="test_model",
            target_version="1",
            reason="bug_found",
        )

        assert result["to_version"] == "1"
        assert result["reason"] == "bug_found"

    def test_get_deployment_history(self, model_registry, mock_mlflow) -> None:
        """
        Test deployment history retrieval.
        """
        mock_client = mock_mlflow.tracking.MlflowClient.return_value

        # Mock model versions
        version1 = MagicMock()
        version1.version = "1"
        version1.current_stage = "Production"
        version1.creation_timestamp = 1234567890
        version1.description = "First version"
        version1.tags = {"env": "prod"}
        version1.run_id = "run_1"

        version2 = MagicMock()
        version2.version = "2"
        version2.current_stage = "Staging"
        version2.creation_timestamp = 1234567900
        version2.description = "Second version"
        version2.tags = {}
        version2.run_id = "run_2"

        mock_client.search_model_versions.return_value = [version2, version1]

        # Mock runs
        run1 = MagicMock()
        run1.data.metrics = {"accuracy": 0.90}
        run1.data.params = {"lr": 0.01}

        run2 = MagicMock()
        run2.data.metrics = {"accuracy": 0.92}
        run2.data.params = {"lr": 0.02}

        mock_client.get_run.side_effect = [run2, run1]

        history = model_registry.get_deployment_history("test_model")

        assert len(history) == 2
        assert history[0]["version"] == "2"
        assert history[0]["current_stage"] == "Staging"
        assert history[0]["run_metrics"]["accuracy"] == 0.92

        assert history[1]["version"] == "1"
        assert history[1]["current_stage"] == "Production"
        assert history[1]["run_metrics"]["accuracy"] == 0.90

    def test_validate_model_quality(self, model_registry, mock_mlflow) -> None:
        """
        Test model quality validation against gates.
        """
        mock_client = mock_mlflow.tracking.MlflowClient.return_value

        # Mock model version and run
        model_version = MagicMock()
        model_version.run_id = "test_run_123"
        mock_client.get_model_version.return_value = model_version

        run = MagicMock()
        run.data.metrics = {
            "accuracy": 0.95,
            "precision": 0.90,
            "recall": 0.85,
            "f1_score": 0.87,
        }
        mock_client.get_run.return_value = run

        # Define quality gates
        quality_gates = {
            "accuracy": 0.90,  # Should pass
            "precision": 0.92,  # Should fail
            "recall": 0.80,  # Should pass
            "auc": 0.85,  # Missing metric - should fail
        }

        results = model_registry.validate_model_quality(
            model_name="test_model",
            version="1",
            quality_gates=quality_gates,
        )

        assert results["model_name"] == "test_model"
        assert results["version"] == "1"
        assert results["gates_passed"] == 2  # accuracy, recall
        assert results["gates_failed"] == 2  # precision, auc
        assert not results["overall_pass"]

        # Check individual gate results
        assert results["gate_results"]["accuracy"]["passed"]
        assert not results["gate_results"]["precision"]["passed"]
        assert results["gate_results"]["recall"]["passed"]
        assert not results["gate_results"]["auc"]["passed"]
        assert results["gate_results"]["auc"]["reason"] == "metric_not_found"

        # Should tag model with validation results
        mock_client.set_model_version_tag.assert_called()

    def test_validate_model_quality_all_pass(self, model_registry, mock_mlflow) -> None:
        """
        Test model quality validation with all gates passing.
        """
        mock_client = mock_mlflow.tracking.MlflowClient.return_value

        model_version = MagicMock()
        model_version.run_id = "test_run_123"
        mock_client.get_model_version.return_value = model_version

        run = MagicMock()
        run.data.metrics = {
            "accuracy": 0.95,
            "precision": 0.93,
        }
        mock_client.get_run.return_value = run

        quality_gates = {
            "accuracy": 0.90,
            "precision": 0.90,
        }

        results = model_registry.validate_model_quality(
            "test_model",
            "1",
            quality_gates,
        )

        assert results["gates_passed"] == 2
        assert results["gates_failed"] == 0
        assert results["overall_pass"]

    def test_ab_test_inactive_status(self, model_registry, mock_mlflow) -> None:
        """
        Test recording results for concluded A/B test.
        """
        test_id = model_registry.setup_ab_test(
            test_name="inactive_test",
            model_a_name="model_a",
            model_b_name="model_b",
            model_a_version="1",
            model_b_version="2",
        )

        # Manually conclude the test
        model_registry._ab_tests[test_id]["status"] = "concluded"

        with pytest.raises(ValueError, match="A/B test .* is not active"):
            model_registry.record_ab_test_result(
                test_id=test_id,
                model_name="model_a",
                metric_value=0.85,
            )

    def test_canary_inactive_status(self, model_registry, mock_mlflow) -> None:
        """
        Test recording metrics for inactive canary deployment.
        """
        deployment_id = model_registry.setup_canary_deployment(
            deployment_name="inactive_canary",
            model_name="test_model",
            model_version="2",
        )

        # Manually deactivate the deployment
        model_registry._canary_deployments[deployment_id]["status"] = "rolled_back"

        with pytest.raises(ValueError, match="Canary deployment .* is not active"):
            model_registry.record_canary_metrics(
                deployment_id=deployment_id,
                metric_value=0.95,
            )
