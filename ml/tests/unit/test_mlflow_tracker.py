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
Unit tests for MLflow XGBoost tracker.

This test suite provides comprehensive coverage for the MLflowXGBoostTracker,
including MLflow integration, experiment tracking, model registry operations,
and artifact management functionality.

"""

import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml._imports import HAS_MLFLOW
from ml.config.xgboost_unified import MLflowConfig
from ml.training.mlflow_tracker import MLflowXGBoostTracker


class TestMLflowXGBoostTracker:
    """Test MLflow XGBoost tracker functionality."""

    @pytest.fixture
    def basic_config(self):
        """Create basic MLflow configuration."""
        return MLflowConfig(
            enabled=True,
            tracking_uri="http://localhost:5000",
            experiment_name="test_experiment",
            model_name="test_model"
        )

    @pytest.fixture
    def sample_model(self):
        """Create sample XGBoost model."""
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "XGBClassifier"
        return mock_model

    @pytest.fixture
    def sample_metrics(self):
        """Create sample training metrics."""
        return {
            "accuracy": 0.85,
            "precision": 0.87,
            "recall": 0.83,
            "f1_score": 0.85,
            "training_time": 120.5
        }

    @pytest.fixture
    def sample_feature_importance(self):
        """Create sample feature importance."""
        return {
            "feature_1": 0.35,
            "feature_2": 0.25,
            "feature_3": 0.20,
            "feature_4": 0.15,
            "feature_5": 0.05
        }

    def test_tracker_initialization(self, basic_config):
        """Test tracker initialization."""
        tracker = MLflowXGBoostTracker(basic_config)
        
        assert tracker.config == basic_config
        assert tracker._mlflow is None
        assert tracker._client is None
        assert tracker._current_run_id is None
        assert tracker._experiment_id is None

    @patch('ml.training.mlflow_tracker.HAS_MLFLOW', True)
    @patch('ml.training.mlflow_tracker.mlflow')
    def test_ensure_mlflow(self, mock_mlflow, basic_config):
        """Test MLflow initialization."""
        tracker = MLflowXGBoostTracker(basic_config)
        
        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "test_exp_123"
        mock_mlflow.set_experiment.return_value = mock_experiment
        
        # Mock client
        mock_client = MagicMock()
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        
        tracker._ensure_mlflow()
        
        # Verify MLflow configuration
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")
        assert tracker._experiment_id == "test_exp_123"
        assert tracker._client == mock_client

    @patch('ml.training.mlflow_tracker.HAS_MLFLOW', False)
    def test_ensure_mlflow_not_available(self, basic_config):
        """Test MLflow initialization when not available."""
        tracker = MLflowXGBoostTracker(basic_config)
        
        # Should handle gracefully - no exception raised
        tracker._ensure_mlflow()
        assert tracker._mlflow is None

    @patch('ml.training.mlflow_tracker.HAS_MLFLOW', True)
    @patch('ml.training.mlflow_tracker.mlflow')
    def test_ensure_mlflow_with_autolog(self, mock_mlflow, basic_config):
        """Test MLflow initialization with auto-logging."""
        config = MLflowConfig(**basic_config.__dict__, auto_log=True)
        tracker = MLflowXGBoostTracker(config)
        
        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "test_exp_123"
        mock_mlflow.set_experiment.return_value = mock_experiment
        
        tracker._ensure_mlflow()
        
        # Verify auto-logging was enabled
        mock_mlflow.xgboost.autolog.assert_called_once()

    @patch('ml.training.mlflow_tracker.HAS_MLFLOW', True)
    @patch('ml.training.mlflow_tracker.mlflow')
    def test_start_run(self, mock_mlflow, basic_config):
        """Test starting MLflow run."""
        tracker = MLflowXGBoostTracker(basic_config)
        
        # Mock run
        mock_run_info = MagicMock()
        mock_run_info.run_id = "test_run_123"
        mock_run = MagicMock()
        mock_run.info = mock_run_info
        mock_mlflow.start_run.return_value = mock_run
        
        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "test_exp_123"
        mock_mlflow.set_experiment.return_value = mock_experiment
        
        run_id = tracker.start_run("test_run", {"custom_tag": "test_value"})
        
        # Verify run was started with correct parameters
        mock_mlflow.start_run.assert_called_once()
        call_args = mock_mlflow.start_run.call_args
        assert call_args[1]["run_name"] == "test_run"
        
        # Verify tags include defaults and custom
        tags = call_args[1]["tags"]
        assert tags["model_type"] == "xgboost"
        assert tags["framework"] == "nautilus_trader"
        assert tags["custom_tag"] == "test_value"
        
        assert run_id == "test_run_123"
        assert tracker._current_run_id == "test_run_123"

    @patch('ml.training.mlflow_tracker.HAS_MLFLOW', True)
    @patch('ml.training.mlflow_tracker.mlflow')
    def test_start_run_auto_name(self, mock_mlflow, basic_config):
        """Test starting MLflow run with auto-generated name."""
        tracker = MLflowXGBoostTracker(basic_config)
        
        # Mock run
        mock_run_info = MagicMock()
        mock_run_info.run_id = "test_run_123"
        mock_run = MagicMock()
        mock_run.info = mock_run_info
        mock_mlflow.start_run.return_value = mock_run
        
        # Mock experiment
        mock_experiment = MagicMock()
        mock_mlflow.set_experiment.return_value = mock_experiment
        
        run_id = tracker.start_run()  # No name provided
        
        # Verify auto-generated name was used
        call_args = mock_mlflow.start_run.call_args
        run_name = call_args[1]["run_name"]
        assert run_name.startswith("xgboost_run_")

    @patch('ml.training.mlflow_tracker.HAS_MLFLOW', True)
    @patch('ml.training.mlflow_tracker.mlflow')
    def test_log_training_run(self, mock_mlflow, basic_config, sample_model, sample_metrics, sample_feature_importance):
        """Test comprehensive training run logging."""
        tracker = MLflowXGBoostTracker(basic_config)
        
        # Mock MLflow functions
        mock_run_info = MagicMock()
        mock_run_info.run_id = "test_run_123"
        mock_run = MagicMock()
        mock_run.info = mock_run_info
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=None)
        
        # Mock experiment
        mock_experiment = MagicMock()
        mock_mlflow.set_experiment.return_value = mock_experiment
        
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "objective": "binary:logistic"
        }
        
        feature_names = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]
        
        artifacts = {
            "config": {"gpu_enabled": True},
            "feature_stats": {"n_features": 5}
        }
        
        run_id = tracker.log_training_run(
            model=sample_model,
            params=params,
            metrics=sample_metrics,
            feature_importance=sample_feature_importance,
            feature_names=feature_names,
            artifacts=artifacts
        )
        
        # Verify run was started and ended
        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.end_run.assert_called_once()
        
        # Verify parameters were logged
        mock_mlflow.log_params.assert_called()
        
        # Verify metrics were logged
        mock_mlflow.log_metrics.assert_called()
        
        # Verify model was logged
        mock_mlflow.xgboost.log_model.assert_called_once()
        
        # Verify artifacts were logged
        mock_mlflow.log_dict.assert_called()
        
        assert run_id == "test_run_123"

    @patch('ml.training.mlflow_tracker.HAS_MLFLOW', True)
    @patch('ml.training.mlflow_tracker.mlflow')
    def test_log_parameters(self, mock_mlflow, basic_config):
        """Test parameter logging with filtering."""
        tracker = MLflowXGBoostTracker(basic_config)
        tracker._mlflow = mock_mlflow
        
        params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "objective": "binary:logistic",
            "tree_method": "hist",
            "complex_object": {"nested": "value"},  # Should be stringified
            "none_value": None,  # Should become "None"
            "boolean_value": True,
        }
        
        tracker._log_parameters(params)
        
        # Verify log_params was called
        mock_mlflow.log_params.assert_called()
        
        # Get the actual parameters that were logged
        logged_params = mock_mlflow.log_params.call_args[0][0]
        
        # Verify transformations
        assert logged_params["n_estimators"] == 100
        assert logged_params["learning_rate"] == 0.1
        assert logged_params["objective"] == "binary:logistic"
        assert logged_params["none_value"] == "None"
        assert logged_params["boolean_value"] is True
        assert isinstance(logged_params["complex_object"], str)

    @patch('ml.training.mlflow_tracker.HAS_MLFLOW', True)
    @patch('ml.training.mlflow_tracker.mlflow')
    def test_log_metrics_with_filtering(self, mock_mlflow, basic_config):
        """Test metric logging with NaN/inf filtering."""
        tracker = MLflowXGBoostTracker(basic_config)
        tracker._mlflow = mock_mlflow
        
        metrics = {
            "accuracy": 0.85,
            "precision": 0.87,
            "invalid_metric": float('inf'),  # Should be filtered out
            "nan_metric": float('nan'),      # Should be filtered out
            "training_time": 120.5,
        }
        
        tracker._log_metrics(metrics)
        
        # Verify log_metrics was called
        mock_mlflow.log_metrics.assert_called()
        
        # Get the actual metrics that were logged
        logged_metrics = mock_mlflow.log_metrics.call_args[0][0]
        
        # Verify filtering
        assert "accuracy" in logged_metrics
        assert "precision" in logged_metrics
        assert "training_time" in logged_metrics
        assert "invalid_metric" not in logged_metrics
        assert "nan_metric" not in logged_metrics

    @patch('ml.training.mlflow_tracker.HAS_MLFLOW', True)
    @patch('ml.training.mlflow_tracker.mlflow')
    def test_log_feature_importance_top_features(self, mock_mlflow, basic_config, sample_feature_importance):
        """Test feature importance logging (top features only)."""
        tracker = MLflowXGBoostTracker(basic_config)
        tracker._mlflow = mock_mlflow
        
        # Create importance dict with more than 20 features
        large_importance = {f"feature_{i}": 1.0 / (i + 1) for i in range(25)}
        
        tracker._log_feature_importance(large_importance)
        
        # Verify metrics were logged
        mock_mlflow.log_metrics.assert_called()
        
        # Get logged metrics
        logged_metrics = mock_mlflow.log_metrics.call_args[0][0]
        
        # Should only log top 20 features
        assert len(logged_metrics) == 20
        
        # Verify naming convention
        for key in logged_metrics.keys():
            assert key.startswith("importance_feature_")

    @patch('ml.training.mlflow_tracker.HAS_MLFLOW', True)
    @patch('ml.training.mlflow_tracker.HAS_XGBOOST', True)
    @patch('ml.training.mlflow_tracker.mlflow')
    def test_log_model_with_input_example(self, mock_mlflow, basic_config, sample_model):
        """Test model logging with input example."""
        tracker = MLflowXGBoostTracker(basic_config)
        tracker._mlflow = mock_mlflow
        
        feature_names = ["feature_1", "feature_2", "feature_3"]
        
        tracker._log_model(sample_model, feature_names)
        
        # Verify model was logged
        mock_mlflow.xgboost.log_model.assert_called_once()
        
        # Get call arguments
        call_args = mock_mlflow.xgboost.log_model.call_args
        assert call_args[1]["xgb_model"] == sample_model
        assert call_args[1]["artifact_path"] == "model"
        
        # Verify input example was created
        input_example = call_args[1]["input_example"]
        assert input_example is not None
        assert input_example.shape == (1, 3)  # 1 sample, 3 features

    @patch('ml.training.mlflow_tracker.HAS_MLFLOW', True)
    @patch('ml.training.mlflow_tracker.mlflow')
    def test_log_artifacts(self, mock_mlflow, basic_config):
        """Test artifact logging."""
        tracker = MLflowXGBoostTracker(basic_config)
        tracker._mlflow = mock_mlflow
        
        artifacts = {
            "config": {"n_estimators": 100, "gpu_enabled": True},
            "feature_stats": {"n_features": 10, "correlation_threshold": 0.95},
            "string_artifact": "Simple string content"
        }
        
        with patch('tempfile.TemporaryDirectory') as mock_temp_dir, \
             patch('pathlib.Path') as mock_path, \
             patch('builtins.open', create=True) as mock_open, \
             patch('json.dump') as mock_json_dump:
            
            # Mock temporary directory
            mock_temp_dir.return_value.__enter__.return_value = "/tmp/test"
            
            # Mock path operations
            mock_artifact_path = MagicMock()
            mock_path.return_value = mock_artifact_path
            mock_artifact_path.__truediv__.return_value = mock_artifact_path
            
            tracker._log_artifacts(artifacts)
            
            # Verify JSON dumps were called for each artifact
            assert mock_json_dump.call_count == 3
            
            # Verify MLflow log_artifact was called
            assert mock_mlflow.log_artifact.call_count == 3

    @patch('ml.training.mlflow_tracker.HAS_MLFLOW', True)
    @patch('ml.training.mlflow_tracker.mlflow')
    def test_log_metadata(self, mock_mlflow, basic_config):
        """Test metadata logging as tags."""
        tracker = MLflowXGBoostTracker(basic_config)
        tracker._mlflow = mock_mlflow
        
        feature_names = ["feature_1", "feature_2", "feature_3"]
        feature_importance = {"feature_1": 0.6, "feature_2": 0.3, "feature_3": 0.1}
        
        tracker._log_metadata(feature_names, feature_importance)
        
        # Verify set_tag was called multiple times
        assert mock_mlflow.set_tag.call_count >= 4  # At least n_features, timestamp, top_feature, importance_sum

    @patch('ml.training.mlflow_tracker.HAS_MLFLOW', True)
    @patch('ml.training.mlflow_tracker.mlflow')
    def test_register_model(self, mock_mlflow, basic_config):
        """Test model registration."""
        tracker = MLflowXGBoostTracker(basic_config)
        
        # Mock client
        mock_client = MagicMock()
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_client.create_model_version.return_value = mock_version
        tracker._client = mock_client
        
        # Mock experiment
        mock_experiment = MagicMock()
        mock_mlflow.set_experiment.return_value = mock_experiment
        
        version = tracker.register_model(
            run_id="test_run_123",
            model_name="test_model",
            stage="Production",
            description="Test model",
            tags={"version": "v1.0"}
        )
        
        # Verify model version was created
        mock_client.create_model_version.assert_called_once_with(
            name="test_model",
            source="runs:/test_run_123/model",
            run_id="test_run_123",
            description="Test model"
        )
        
        # Verify stage transition
        mock_client.transition_model_version_stage.assert_called_once_with(
            name="test_model",
            version="1",
            stage="Production",
            archive_existing_versions=False
        )
        
        # Verify tags were set
        mock_client.set_model_version_tag.assert_called_once_with(
            "test_model", "1", "version", "v1.0"
        )
        
        assert version == "1"

    @patch('ml.training.mlflow_tracker.HAS_MLFLOW', True)
    @patch('ml.training.mlflow_tracker.mlflow')
    def test_load_model(self, mock_mlflow, basic_config):
        """Test model loading from registry."""
        tracker = MLflowXGBoostTracker(basic_config)
        
        # Mock loaded model
        mock_loaded_model = MagicMock()
        mock_mlflow.xgboost.load_model.return_value = mock_loaded_model
        
        # Mock experiment
        mock_experiment = MagicMock()
        mock_mlflow.set_experiment.return_value = mock_experiment
        
        loaded_model = tracker.load_model("test_model", "Production")
        
        # Verify model was loaded
        mock_mlflow.xgboost.load_model.assert_called_once_with("models:/test_model/Production")
        assert loaded_model == mock_loaded_model

    @patch('ml.training.mlflow_tracker.HAS_MLFLOW', True)
    @patch('ml.training.mlflow_tracker.mlflow')
    def test_load_model_by_version(self, mock_mlflow, basic_config):
        """Test model loading by specific version."""
        tracker = MLflowXGBoostTracker(basic_config)
        
        # Mock loaded model
        mock_loaded_model = MagicMock()
        mock_mlflow.xgboost.load_model.return_value = mock_loaded_model
        
        # Mock experiment
        mock_experiment = MagicMock()
        mock_mlflow.set_experiment.return_value = mock_experiment
        
        loaded_model = tracker.load_model_by_version("test_model", "3")
        
        # Verify model was loaded with correct URI
        mock_mlflow.xgboost.load_model.assert_called_once_with("models:/test_model/3")
        assert loaded_model == mock_loaded_model

    @patch('ml.training.mlflow_tracker.HAS_MLFLOW', True)
    @patch('ml.training.mlflow_tracker.mlflow')
    def test_get_model_info(self, mock_mlflow, basic_config):
        """Test getting model information."""
        tracker = MLflowXGBoostTracker(basic_config)
        
        # Mock client and model
        mock_client = MagicMock()
        
        # Mock registered model
        mock_model = MagicMock()
        mock_model.name = "test_model"
        mock_model.description = "Test model description"
        mock_model.tags = {"env": "production"}
        mock_model.creation_timestamp = 1234567890
        mock_model.last_updated_timestamp = 1234567900
        
        # Mock model versions
        mock_version_1 = MagicMock()
        mock_version_1.version = "1"
        mock_version_1.current_stage = "Production"
        mock_version_1.description = "Version 1"
        mock_version_1.run_id = "run_123"
        mock_version_1.creation_timestamp = 1234567890
        mock_version_1.tags = {"version": "v1.0"}
        
        mock_version_2 = MagicMock()
        mock_version_2.version = "2"
        mock_version_2.current_stage = "Staging"
        mock_version_2.description = "Version 2"
        mock_version_2.run_id = "run_456"
        mock_version_2.creation_timestamp = 1234567900
        mock_version_2.tags = {"version": "v2.0"}
        
        mock_client.get_registered_model.return_value = mock_model
        mock_client.get_latest_versions.return_value = [mock_version_1, mock_version_2]
        
        tracker._client = mock_client
        
        # Mock experiment
        mock_experiment = MagicMock()
        mock_mlflow.set_experiment.return_value = mock_experiment
        
        model_info = tracker.get_model_info("test_model")
        
        # Verify client methods were called
        mock_client.get_registered_model.assert_called_once_with("test_model")
        mock_client.get_latest_versions.assert_called_once_with(
            "test_model", stages=["None", "Staging", "Production", "Archived"]
        )
        
        # Verify model info structure
        assert model_info["name"] == "test_model"
        assert model_info["description"] == "Test model description"
        assert model_info["tags"] == {"env": "production"}
        assert len(model_info["latest_versions"]) == 2
        
        # Verify version info
        version_info = model_info["latest_versions"][0]
        assert version_info["version"] == "1"
        assert version_info["stage"] == "Production"
        assert version_info["run_id"] == "run_123"

    @patch('ml.training.mlflow_tracker.HAS_MLFLOW', True)
    @patch('ml.training.mlflow_tracker.mlflow')
    def test_cleanup_old_runs(self, mock_mlflow, basic_config):
        """Test cleanup of old runs."""
        tracker = MLflowXGBoostTracker(basic_config)
        
        # Mock client
        mock_client = MagicMock()
        
        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp_123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        
        # Mock runs (more than max_runs)
        mock_runs = []
        for i in range(15):
            run = MagicMock()
            run.info.run_id = f"run_{i}"
            mock_runs.append(run)
        
        mock_client.search_runs.return_value = mock_runs
        tracker._client = mock_client
        
        tracker.cleanup_old_runs(max_runs=10)
        
        # Verify search was performed
        mock_client.search_runs.assert_called_once_with(
            experiment_ids=["exp_123"],
            order_by=["attribute.start_time DESC"]
        )
        
        # Verify 5 oldest runs were deleted (15 - 10 = 5)
        assert mock_client.delete_run.call_count == 5
        
        # Verify correct runs were deleted (oldest ones)
        deleted_run_ids = [call[0][0] for call in mock_client.delete_run.call_args_list]
        expected_deleted = [f"run_{i}" for i in range(10, 15)]  # Last 5 in the list (oldest)
        assert deleted_run_ids == expected_deleted

    @patch('ml.training.mlflow_tracker.HAS_MLFLOW', True)
    @patch('ml.training.mlflow_tracker.mlflow')
    def test_cleanup_old_runs_no_cleanup_needed(self, mock_mlflow, basic_config):
        """Test cleanup when no cleanup is needed."""
        tracker = MLflowXGBoostTracker(basic_config)
        
        # Mock client
        mock_client = MagicMock()
        
        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp_123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        
        # Mock runs (less than max_runs)
        mock_runs = [MagicMock() for _ in range(5)]
        mock_client.search_runs.return_value = mock_runs
        tracker._client = mock_client
        
        tracker.cleanup_old_runs(max_runs=10)
        
        # Verify no deletion occurred
        mock_client.delete_run.assert_not_called()

    @patch('ml.training.mlflow_tracker.HAS_MLFLOW', True)
    @patch('ml.training.mlflow_tracker.mlflow')
    def test_cleanup_old_runs_experiment_not_found(self, mock_mlflow, basic_config):
        """Test cleanup when experiment doesn't exist."""
        tracker = MLflowXGBoostTracker(basic_config)
        
        # Mock experiment not found
        mock_mlflow.get_experiment_by_name.return_value = None
        
        # Should handle gracefully
        tracker.cleanup_old_runs(max_runs=10)
        
        # Verify no further operations
        mock_mlflow.get_experiment_by_name.assert_called_once_with("test_experiment")

    def test_model_operations_without_mlflow(self, basic_config):
        """Test model operations when MLflow is not available."""
        config = MLflowConfig(**basic_config.__dict__, enabled=False)
        tracker = MLflowXGBoostTracker(config)
        
        # Should return None for operations when MLflow not available
        result = tracker.log_training_run(
            model=MagicMock(),
            params={},
            metrics={},
            feature_importance={}
        )
        assert result is None
        
        result = tracker.load_model("test_model")
        assert result is None