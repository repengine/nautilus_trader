
"""
Unit tests for MLflowManager class.

Tests cover:
- Initialization and configuration
- Run context management
- Training session logging
- Model registration and transitions
- Cleanup operations
- Error handling and edge cases

NOTE: All MLflow server interactions are mocked to ensure tests are independent
and do not require an actual MLflow server.

"""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from ml.config.shared import MLflowConfig
from ml.tracking.mlflow_manager import MLflowManager


class TestMLflowManagerInitialization:
    """
    Test MLflowManager initialization and configuration.
    """

    def test_initialization_with_config(self) -> None:
        """
        Test MLflowManager initializes with proper config.
        """
        # Arrange
        config = MLflowConfig(
            tracking_uri="http://localhost:5000",
            experiment_name="test_experiment",
            model_name="test_model",
        )

        # Act
        manager = MLflowManager(config)

        # Assert
        assert manager.config == config
        assert manager._mlflow is None
        assert manager._client is None
        assert manager._current_run_id is None
        assert manager._experiment_id is None
        assert not manager._initialized

    @patch("ml.tracking.mlflow_manager.HAS_MLFLOW", True)
    @patch("ml.tracking.mlflow_manager.mlflow")
    def test_ensure_initialized_sets_up_mlflow(self, mock_mlflow: Mock) -> None:
        """
        Test _ensure_initialized properly configures MLflow.
        """
        # Arrange
        config = MLflowConfig(
            tracking_uri="http://localhost:5000",
            experiment_name="test_experiment",
        )
        manager = MLflowManager(config)

        mock_client = MagicMock()
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "exp123"

        # Act
        manager._ensure_initialized()

        # Assert
        assert manager._initialized
        assert manager._mlflow == mock_mlflow
        assert manager._client == mock_client
        assert manager._experiment_id == "exp123"
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
        mock_mlflow.create_experiment.assert_called_once()

    @patch("ml.tracking.mlflow_manager.HAS_MLFLOW", True)
    @patch("ml.tracking.mlflow_manager.mlflow")
    def test_ensure_initialized_uses_existing_experiment(self, mock_mlflow: Mock) -> None:
        """
        Test _ensure_initialized uses existing experiment if found.
        """
        # Arrange
        config = MLflowConfig(
            experiment_name="existing_experiment",
            tracking_uri="http://localhost:5000",
        )
        manager = MLflowManager(config)

        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "existing123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        # Act
        manager._ensure_initialized()

        # Assert
        assert manager._experiment_id == "existing123"
        mock_mlflow.create_experiment.assert_not_called()

    @patch("ml.tracking.mlflow_manager.HAS_MLFLOW", False)
    def test_ensure_initialized_raises_when_mlflow_unavailable(self) -> None:
        """
        Test _ensure_initialized raises error when MLflow not available.
        """
        # Arrange
        config = MLflowConfig()
        manager = MLflowManager(config)

        # Act & Assert
        with pytest.raises(ImportError, match="mlflow"):
            manager._ensure_initialized()


class TestMLflowManagerRunContext:
    """
    Test MLflowManager run context management.
    """

    @patch("ml.tracking.mlflow_manager.HAS_MLFLOW", True)
    @patch("ml.tracking.mlflow_manager.mlflow")
    def test_run_context_success(self, mock_mlflow: Mock) -> None:
        """
        Test run_context successfully manages run lifecycle.
        """
        # Arrange
        config = MLflowConfig(experiment_name="test")
        manager = MLflowManager(config)

        mock_run = MagicMock()
        mock_run.info.run_id = "run123"
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "exp123"

        # Act
        with manager.run_context(run_name="test_run", tags={"key": "value"}) as run_id:
            assert run_id == "run123"
            assert manager._current_run_id == "run123"

        # Assert
        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.end_run.assert_called_once()
        assert manager._current_run_id is None

    @patch("ml.tracking.mlflow_manager.HAS_MLFLOW", True)
    @patch("ml.tracking.mlflow_manager.mlflow")
    def test_run_context_with_exception(self, mock_mlflow: Mock) -> None:
        """
        Test run_context properly cleans up on exception.
        """
        # Arrange
        config = MLflowConfig(experiment_name="test")
        manager = MLflowManager(config)

        mock_run = MagicMock()
        mock_run.info.run_id = "run456"
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "exp123"

        # Act & Assert
        with pytest.raises(ValueError, match="Test error"):
            with manager.run_context(run_name="failing_run"):
                raise ValueError("Test error")

        # Ensure cleanup happened
        mock_mlflow.end_run.assert_called_once()
        assert manager._current_run_id is None

    @patch("ml.tracking.mlflow_manager.HAS_MLFLOW", True)
    @patch("ml.tracking.mlflow_manager.mlflow")
    def test_run_context_nested(self, mock_mlflow: Mock) -> None:
        """
        Test nested run context management.
        """
        # Arrange
        config = MLflowConfig(experiment_name="test")
        manager = MLflowManager(config)

        mock_parent_run = MagicMock()
        mock_parent_run.info.run_id = "parent123"
        mock_nested_run = MagicMock()
        mock_nested_run.info.run_id = "nested456"

        mock_mlflow.start_run.side_effect = [mock_parent_run, mock_nested_run]
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "exp123"

        # Act
        with manager.run_context(run_name="parent") as parent_id:
            assert parent_id == "parent123"
            assert manager._current_run_id == "parent123"

            with manager.run_context(run_name="nested", nested=True) as nested_id:
                assert nested_id == "nested456"
                # Current run should still be parent
                assert manager._current_run_id == "parent123"

        # Assert
        assert mock_mlflow.start_run.call_count == 2
        assert mock_mlflow.end_run.call_count == 2
        assert manager._current_run_id is None


class TestMLflowManagerLogging:
    """
    Test MLflowManager logging functionality.
    """

    @patch("ml.tracking.mlflow_manager.HAS_MLFLOW", True)
    @patch("ml.tracking.mlflow_manager.mlflow")
    def test_log_training_session_complete(self, mock_mlflow: Mock) -> None:
        """
        Test log_training_session with all parameters.
        """
        # Arrange
        config = MLflowConfig(
            experiment_name="test",
            log_model=True,
            log_artifacts=True,
        )
        manager = MLflowManager(config)
        manager._mlflow = mock_mlflow

        mock_run = MagicMock()
        mock_run.info.run_id = "run789"
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "exp123"

        # Create a more realistic mock that doesn't have xgboost/lightgbm attributes
        class TestModel:
            def __init__(self) -> None:
                pass

        model = TestModel()
        params = {"learning_rate": 0.01, "n_estimators": 100}
        metrics = {"accuracy": 0.95, "auc": 0.92}
        feature_importance = {"feature1": 0.5, "feature2": 0.3}
        feature_names = ["feature1", "feature2"]
        artifacts = {"config": {"test": "value"}}

        # Act
        run_id = manager.log_training_session(
            model=model,
            params=params,
            metrics=metrics,
            feature_importance=feature_importance,
            feature_names=feature_names,
            artifacts=artifacts,
            run_name="test_session",
            tags={"test": "true"},
        )

        # Assert
        assert run_id == "run789"
        mock_mlflow.log_params.assert_called()
        mock_mlflow.log_metrics.assert_called()
        mock_mlflow.log_artifact.assert_called()
        mock_mlflow.sklearn.log_model.assert_called_once()

    @patch("ml.tracking.mlflow_manager.HAS_MLFLOW", True)
    @patch("ml.tracking.mlflow_manager.mlflow")
    def test_log_params_batch_handles_types(self, mock_mlflow: Mock) -> None:
        """
        Test _log_params_batch handles different parameter types.
        """
        # Arrange
        config = MLflowConfig()
        manager = MLflowManager(config)
        manager._mlflow = mock_mlflow

        params = {
            "int_param": 42,
            "float_param": 3.14,
            "str_param": "test",
            "bool_param": True,
            "none_param": None,
            "complex_param": {"nested": "value"},
            "long_string": "x" * 300,  # Should be truncated
        }

        # Act
        manager._log_params_batch(params)

        # Assert
        mock_mlflow.log_params.assert_called_once()
        logged_params = mock_mlflow.log_params.call_args[0][0]
        assert logged_params["int_param"] == 42
        assert logged_params["float_param"] == 3.14
        assert logged_params["str_param"] == "test"
        assert logged_params["bool_param"] is True
        assert logged_params["none_param"] == "None"
        assert len(logged_params["long_string"]) == 250

    @patch("ml.tracking.mlflow_manager.HAS_MLFLOW", True)
    @patch("ml.tracking.mlflow_manager.mlflow")
    def test_log_metrics_batch_validates_values(self, mock_mlflow: Mock) -> None:
        """
        Test _log_metrics_batch validates and filters metrics.
        """
        # Arrange
        config = MLflowConfig()
        manager = MLflowManager(config)
        manager._mlflow = mock_mlflow

        metrics: dict[str, float | int | str] = {
            "valid_int": 42,
            "valid_float": 3.14,
            "nan_value": float("nan"),
            "inf_value": float("inf"),
            "string_value": "not_a_number",
        }

        # Act
        manager._log_metrics_batch(metrics)  # type: ignore[arg-type]

        # Assert
        mock_mlflow.log_metrics.assert_called_once()
        logged_metrics = mock_mlflow.log_metrics.call_args[0][0]
        assert "valid_int" in logged_metrics
        assert "valid_float" in logged_metrics
        assert "nan_value" not in logged_metrics
        assert "inf_value" not in logged_metrics
        assert "string_value" not in logged_metrics

    @patch("ml.tracking.mlflow_manager.HAS_MLFLOW", True)
    @patch("ml.tracking.mlflow_manager.mlflow")
    def test_log_feature_importance(self, mock_mlflow: Mock) -> None:
        """
        Test _log_feature_importance logs metrics and artifacts.
        """
        # Arrange
        config = MLflowConfig()
        manager = MLflowManager(config)
        manager._mlflow = mock_mlflow

        feature_importance = {
            "feature_a": 0.5,
            "feature_b": 0.3,
            "feature_c": 0.2,
        }

        # Act
        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_file = MagicMock()
            mock_file.name = "/tmp/test.json"
            mock_temp.return_value.__enter__.return_value = mock_file

            manager._log_feature_importance(feature_importance, top_n=2)

        # Assert
        mock_mlflow.log_metrics.assert_called_once()
        logged_metrics = mock_mlflow.log_metrics.call_args[0][0]
        assert len(logged_metrics) == 2  # Only top 2
        mock_mlflow.log_artifact.assert_called_once()
