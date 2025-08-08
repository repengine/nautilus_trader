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

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from ml._imports import HAS_MLFLOW
from ml.config.shared import MLflowConfig
from ml.tracking.mlflow_manager import MLflowManager
from ml.tracking.mlflow_manager import ModelStage


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

        # Mock specific model logging modules
        mock_mlflow.lightgbm.log_model = MagicMock()
        mock_mlflow.sklearn.log_model = MagicMock()
        mock_mlflow.xgboost.log_model = MagicMock()

        # Mock dependency check to do nothing
        mock_check_deps.return_value = None

        yield mock_mlflow


@pytest.fixture
def mlflow_manager(mlflow_config, mock_mlflow):
    """
    Create MLflow manager with mocked dependencies.
    """
    manager = MLflowManager(mlflow_config)
    return manager


class TestMLflowManager:
    """
    Test suite for MLflowManager.
    """

    @pytest.mark.skipif(not HAS_MLFLOW, reason="MLflow not available")
    def test_init_with_config(self, mlflow_config) -> None:
        """
        Test MLflowManager initialization with configuration.
        """
        manager = MLflowManager(mlflow_config)

        assert manager.config == mlflow_config
        assert manager._mlflow is None
        assert manager._client is None
        assert manager._current_run_id is None
        assert not manager._initialized

    def test_ensure_initialized_sets_up_mlflow(self, mlflow_manager, mock_mlflow) -> None:
        """
        Test that _ensure_initialized properly configures MLflow.
        """
        mlflow_manager._ensure_initialized()

        assert mlflow_manager._initialized
        assert mlflow_manager._mlflow is not None
        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_mlflow.tracking.MlflowClient.assert_called_once()

    def test_run_context_manager(self, mlflow_manager, mock_mlflow) -> None:
        """
        Test run context manager functionality.
        """
        with mlflow_manager.run_context(run_name="test_run") as run_id:
            assert run_id == "test_run_123"
            mock_mlflow.start_run.assert_called_once()

        mock_mlflow.end_run.assert_called_once()

    def test_run_context_with_nested_run(self, mlflow_manager, mock_mlflow) -> None:
        """
        Test nested run context.
        """
        with mlflow_manager.run_context(run_name="parent", nested=False):
            with mlflow_manager.run_context(run_name="child", nested=True):
                # Should start nested run
                assert mock_mlflow.start_run.call_count == 2

        # Should end both runs
        assert mock_mlflow.end_run.call_count == 2

    def test_run_context_exception_handling(self, mlflow_manager, mock_mlflow) -> None:
        """
        Test that run context properly ends run on exception.
        """
        with pytest.raises(ValueError):
            with mlflow_manager.run_context(run_name="test_run"):
                raise ValueError("Test exception")

        mock_mlflow.end_run.assert_called_once()

    def test_log_training_session(self, mlflow_manager, mock_mlflow) -> None:
        """
        Test comprehensive training session logging.
        """
        model = MagicMock()
        params = {"learning_rate": 0.01, "max_depth": 6}
        metrics = {"accuracy": 0.95, "auc": 0.88}
        feature_importance = {"feature_1": 0.3, "feature_2": 0.7}
        feature_names = ["feature_1", "feature_2"]

        run_id = mlflow_manager.log_training_session(
            model=model,
            params=params,
            metrics=metrics,
            feature_importance=feature_importance,
            feature_names=feature_names,
        )

        assert run_id == "test_run_123"
        mock_mlflow.log_params.assert_called()
        mock_mlflow.log_metrics.assert_called()

    def test_log_params_batch_filtering(self, mlflow_manager, mock_mlflow) -> None:
        """
        Test parameter batch logging with filtering.
        """
        params = {
            "int_param": 42,
            "float_param": 3.14,
            "str_param": "test",
            "bool_param": True,
            "none_param": None,
            "object_param": {"nested": "dict"},
        }

        with mlflow_manager.run_context():
            mlflow_manager._log_params_batch(params)

        # Should call log_params with serializable values
        mock_mlflow.log_params.assert_called()
        call_args = mock_mlflow.log_params.call_args[0][0]

        assert call_args["int_param"] == 42
        assert call_args["float_param"] == 3.14
        assert call_args["str_param"] == "test"
        assert call_args["bool_param"] is True
        assert call_args["none_param"] == "None"
        assert isinstance(call_args["object_param"], str)

    def test_log_metrics_batch_validation(self, mlflow_manager, mock_mlflow) -> None:
        """
        Test metrics batch logging with validation.
        """
        metrics = {
            "valid_metric": 0.95,
            "inf_metric": float("inf"),
            "nan_metric": float("nan"),
            "large_metric": 1e20,
        }

        with mlflow_manager.run_context():
            mlflow_manager._log_metrics_batch(metrics)

        # Should only log finite, valid metrics
        mock_mlflow.log_metrics.assert_called()
        call_args = mock_mlflow.log_metrics.call_args[0][0]

        assert "valid_metric" in call_args
        assert "large_metric" in call_args
        assert "inf_metric" not in call_args
        assert "nan_metric" not in call_args

    def test_log_feature_importance(self, mlflow_manager, mock_mlflow) -> None:
        """
        Test feature importance logging.
        """
        feature_importance = {f"feature_{i}": 1.0 / (i + 1) for i in range(30)}

        with mlflow_manager.run_context():
            mlflow_manager._log_feature_importance(feature_importance, top_n=5)

        # Should log top features as metrics and full importance as artifact
        mock_mlflow.log_metrics.assert_called()
        mock_mlflow.log_artifact.assert_called()

    def test_log_model_generic_xgboost(self, mlflow_manager, mock_mlflow) -> None:
        """
        Test generic model logging with XGBoost detection.
        """
        # Mock XGBoost model
        model = MagicMock()
        model.__class__.__name__ = "XGBClassifier"

        with mlflow_manager.run_context():
            mlflow_manager._log_model_generic(model)

        mock_mlflow.xgboost.log_model.assert_called_once()

    def test_log_model_generic_lightgbm(self, mlflow_manager, mock_mlflow) -> None:
        """
        Test generic model logging with LightGBM detection.
        """

        # Create a mock with the right type name containing 'lightgbm'
        class MockLightGBMClassifier:
            pass

        model = MockLightGBMClassifier()

        with mlflow_manager.run_context():
            mlflow_manager._log_model_generic(model)

        mock_mlflow.lightgbm.log_model.assert_called_once()

    def test_log_model_generic_fallback(self, mlflow_manager, mock_mlflow) -> None:
        """
        Test generic model logging fallback to sklearn.
        """

        # Create a mock with a generic type name
        class MockGenericModel:
            pass

        model = MockGenericModel()

        with mlflow_manager.run_context():
            mlflow_manager._log_model_generic(model)

        mock_mlflow.sklearn.log_model.assert_called_once()

    def test_log_artifacts_batch(self, mlflow_manager, mock_mlflow) -> None:
        """
        Test batch artifact logging.
        """
        artifacts = {
            "config": {"param1": "value1"},
            "results": [1, 2, 3],
            "text": "some text",
        }

        with mlflow_manager.run_context():
            mlflow_manager._log_artifacts_batch(artifacts)

        # Should log each artifact as JSON file
        assert mock_mlflow.log_artifact.call_count == len(artifacts)

    def test_register_model(self, mlflow_manager, mock_mlflow) -> None:
        """
        Test model registration.
        """
        mock_version = MagicMock()
        mock_version.version = "1"

        mock_client = mock_mlflow.tracking.MlflowClient.return_value
        mock_client.create_model_version.return_value = mock_version

        version = mlflow_manager.register_model(
            run_id="test_run_123",
            model_name="test_model",
            stage=ModelStage.STAGING,
            description="Test model",
            tags={"env": "test"},
        )

        assert version == "1"
        mock_client.create_model_version.assert_called_once()
        mock_client.set_model_version_tag.assert_called()

    def test_transition_model_stage(self, mlflow_manager, mock_mlflow) -> None:
        """
        Test model stage transition.
        """
        mock_client = mock_mlflow.tracking.MlflowClient.return_value

        mlflow_manager.transition_model_stage(
            model_name="test_model",
            version="1",
            stage=ModelStage.PRODUCTION,
            archive_existing=True,
        )

        mock_client.transition_model_version_stage.assert_called_once_with(
            name="test_model",
            version="1",
            stage="Production",
            archive_existing_versions=True,
        )

    def test_load_model_xgboost(self, mlflow_manager, mock_mlflow) -> None:
        """
        Test model loading with XGBoost flavor.
        """
        mock_model = MagicMock()
        mock_mlflow.xgboost.load_model.return_value = mock_model

        model = mlflow_manager.load_model(
            model_name="test_model",
            stage=ModelStage.PRODUCTION,
        )

        assert model == mock_model
        mock_mlflow.xgboost.load_model.assert_called_once_with(
            "models:/test_model/Production",
        )

    def test_load_model_fallback_to_lightgbm(self, mlflow_manager, mock_mlflow) -> None:
        """
        Test model loading fallback to LightGBM.
        """
        mock_model = MagicMock()
        mock_mlflow.xgboost.load_model.side_effect = Exception("XGBoost failed")
        mock_mlflow.lightgbm.load_model.return_value = mock_model

        model = mlflow_manager.load_model("test_model", ModelStage.PRODUCTION)

        assert model == mock_model
        mock_mlflow.lightgbm.load_model.assert_called_once()

    def test_load_model_by_version(self, mlflow_manager, mock_mlflow) -> None:
        """
        Test loading specific model version.
        """
        mock_model = MagicMock()
        mock_mlflow.xgboost.load_model.return_value = mock_model

        model = mlflow_manager.load_model_by_version("test_model", "2")

        assert model == mock_model
        mock_mlflow.xgboost.load_model.assert_called_once_with(
            "models:/test_model/2",
        )

    def test_compare_models(self, mlflow_manager, mock_mlflow) -> None:
        """
        Test model comparison functionality.
        """
        mock_client = mock_mlflow.tracking.MlflowClient.return_value

        # Mock model version
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_version.run_id = "run_123"
        mock_version.creation_timestamp = 1234567890
        mock_version.description = "Test version"
        mock_version.tags = {"env": "test"}

        # Mock run
        mock_run = MagicMock()
        mock_run.data.metrics = {"accuracy": 0.95}
        mock_run.data.params = {"lr": 0.01}

        mock_client.get_latest_versions.return_value = [mock_version]
        mock_client.get_run.return_value = mock_run

        results = mlflow_manager.compare_models(
            model_names=["model_1", "model_2"],
            metric_name="accuracy",
            stage=ModelStage.PRODUCTION,
        )

        assert "model_1" in results
        assert "model_2" in results
        assert results["model_1"]["metric_value"] == 0.95

    def test_cleanup_old_runs(self, mlflow_manager, mock_mlflow) -> None:
        """
        Test cleanup of old runs.
        """
        mock_client = mock_mlflow.tracking.MlflowClient.return_value
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp_123"

        # Mock runs (more than max_runs)
        mock_runs = [MagicMock() for _ in range(150)]
        for i, run in enumerate(mock_runs):
            run.info.run_id = f"run_{i}"

        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        mock_client.search_runs.return_value = mock_runs

        stats = mlflow_manager.cleanup_old_runs(
            max_runs=100,
            experiment_name="test_experiment",
        )

        assert stats["runs_deleted"] == 50
        assert stats["runs_examined"] == 150
        assert mock_client.delete_run.call_count == 50

    def test_cleanup_old_runs_dry_run(self, mlflow_manager, mock_mlflow) -> None:
        """
        Test dry run cleanup mode.
        """
        mock_client = mock_mlflow.tracking.MlflowClient.return_value
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp_123"
        mock_experiment.name = "test_experiment"
        mock_runs = [MagicMock() for _ in range(150)]

        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        mock_client.search_runs.return_value = mock_runs

        stats = mlflow_manager.cleanup_old_runs(
            max_runs=100,
            experiment_name="test_experiment",
            dry_run=True,
        )

        assert stats["runs_deleted"] == 50
        assert mock_client.delete_run.call_count == 0  # No actual deletion

    def test_get_experiment_summary(self, mlflow_manager, mock_mlflow) -> None:
        """
        Test experiment summary generation.
        """
        mock_client = mock_mlflow.tracking.MlflowClient.return_value
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp_123"
        mock_experiment.creation_time = 1234567890
        mock_experiment.tags = {"env": "test"}

        # Mock runs with different statuses
        mock_runs = []
        statuses = ["FINISHED", "RUNNING", "FAILED"]
        for i in range(15):
            run = MagicMock()
            run.info.status = statuses[i % 3]
            run.data.metrics = {"accuracy": 0.9 + 0.01 * i}
            mock_runs.append(run)

        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        mock_client.search_runs.return_value = mock_runs

        summary = mlflow_manager.get_experiment_summary("test_experiment")

        assert summary["experiment_name"] == "test_experiment"
        assert summary["total_runs"] == 15
        assert summary["completed_runs"] == 5
        assert summary["active_runs"] == 5
        assert summary["failed_runs"] == 5
        assert "metric_statistics" in summary

    def test_health_check_without_mlflow(self, mlflow_config) -> None:
        """
        Test health check when MLflow is not available.
        """
        with patch("ml.tracking.mlflow_manager.HAS_MLFLOW", False):
            manager = MLflowManager(mlflow_config)
            status = manager.health_check()

            assert not status["mlflow_available"]
            assert not status["initialized"]
            assert not status["connectivity"]

    def test_health_check_with_mlflow(self, mlflow_manager, mock_mlflow) -> None:
        """
        Test health check with MLflow available.
        """
        mock_client = mock_mlflow.tracking.MlflowClient.return_value
        mock_client.search_experiments.return_value = [MagicMock(), MagicMock()]

        status = mlflow_manager.health_check()

        assert status["mlflow_available"]
        assert status["connectivity"]
        assert status["total_experiments"] == 2

    def test_health_check_with_connectivity_error(self, mlflow_manager, mock_mlflow) -> None:
        """
        Test health check with connectivity error.
        """
        mock_client = mock_mlflow.tracking.MlflowClient.return_value
        mock_client.search_experiments.side_effect = Exception("Connection failed")

        status = mlflow_manager.health_check()

        assert status["mlflow_available"]
        assert not status["connectivity"]
        assert "error" in status

    def test_context_manager_with_tags(self, mlflow_manager, mock_mlflow) -> None:
        """
        Test context manager with custom tags.
        """
        custom_tags = {"model_type": "xgboost", "dataset": "test_data"}

        with mlflow_manager.run_context(tags=custom_tags) as run_id:
            pass

        # Check that tags were passed to start_run
        call_args = mock_mlflow.start_run.call_args
        passed_tags = call_args[1]["tags"]

        assert passed_tags["model_type"] == "xgboost"
        assert passed_tags["dataset"] == "test_data"
        assert passed_tags["framework"] == "nautilus_trader"  # Default tag

    def test_log_session_metadata(self, mlflow_manager, mock_mlflow) -> None:
        """
        Test session metadata logging.
        """
        feature_names = ["f1", "f2", "f3"]
        feature_importance = {"f1": 0.5, "f2": 0.3, "f3": 0.2}

        with mlflow_manager.run_context():
            mlflow_manager._log_session_metadata(feature_names, feature_importance)

        # Should log metrics and set tags
        mock_mlflow.log_metric.assert_called()
        mock_mlflow.set_tag.assert_called()

    def test_invalid_experiment_name_handling(self, mlflow_manager, mock_mlflow) -> None:
        """
        Test handling of invalid experiment name.
        """
        mock_mlflow.get_experiment_by_name.return_value = None

        # Should not raise exception, just log warning
        mlflow_manager._setup_experiment()

        assert mlflow_manager._experiment_id is None
