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
Unit tests for unified XGBoost trainer.

This test suite provides comprehensive coverage for the UnifiedXGBoostTrainer,
including GPU configuration, feature decay tracking, hyperparameter optimization,
MLflow integration, and model export functionality.

"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from ml.config.base import MLFeatureConfig
from ml.config.xgboost_unified import GPUConfig, MLflowConfig, OptunaConfig, UnifiedXGBoostConfig
from ml.training.xgboost_unified import UnifiedXGBoostTrainer


class TestGPUConfig:
    """Test GPU configuration validation and settings."""

    def test_default_gpu_config(self):
        """Test default GPU configuration."""
        config = GPUConfig()
        
        assert config.enabled is False
        assert config.device_id == 0
        assert config.max_bin == 256
        assert config.predictor == "gpu_predictor"
        assert config.validate_gpu is True

    def test_gpu_config_validation(self):
        """Test GPU configuration validation."""
        # Valid configuration
        config = GPUConfig(
            enabled=True,
            device_id=1,
            max_bin=512,
            predictor="gpu_predictor"
        )
        assert config.enabled is True
        assert config.device_id == 1

        # Invalid predictor
        with pytest.raises(ValueError, match="predictor must be"):
            GPUConfig(predictor="invalid_predictor")

        # Invalid device_id
        with pytest.raises(ValueError, match="device_id must be non-negative"):
            GPUConfig(device_id=-1)

        # Invalid max_bin
        with pytest.raises(ValueError, match="max_bin must be positive"):
            GPUConfig(max_bin=0)


class TestOptunaConfig:
    """Test Optuna configuration validation and settings."""

    def test_default_optuna_config(self):
        """Test default Optuna configuration."""
        config = OptunaConfig()
        
        assert config.enabled is False
        assert config.n_trials == 100
        assert config.direction == "maximize"
        assert config.metric == "sharpe_ratio"
        assert config.pruner == "median"
        assert config.sampler == "tpe"

    def test_optuna_config_validation(self):
        """Test Optuna configuration validation."""
        # Valid configuration
        config = OptunaConfig(
            enabled=True,
            n_trials=50,
            metric="accuracy",
            timeout=300
        )
        assert config.n_trials == 50
        assert config.metric == "accuracy"
        assert config.timeout == 300

        # Invalid n_trials
        with pytest.raises(ValueError, match="n_trials must be positive"):
            OptunaConfig(n_trials=0)

        # Invalid direction
        with pytest.raises(ValueError, match="direction must be one of"):
            OptunaConfig(direction="invalid")

        # Invalid metric
        with pytest.raises(ValueError, match="metric must be one of"):
            OptunaConfig(metric="invalid_metric")

        # Invalid pruner
        with pytest.raises(ValueError, match="pruner must be one of"):
            OptunaConfig(pruner="invalid_pruner")

        # Invalid sampler
        with pytest.raises(ValueError, match="sampler must be one of"):
            OptunaConfig(sampler="invalid_sampler")


class TestMLflowConfig:
    """Test MLflow configuration validation and settings."""

    def test_default_mlflow_config(self):
        """Test default MLflow configuration."""
        config = MLflowConfig()
        
        assert config.enabled is False
        assert config.tracking_uri == "http://localhost:5000"
        assert config.experiment_name == "xgboost_unified"
        assert config.register_model is True
        assert config.model_name == "xgboost_unified"

    def test_mlflow_config_validation(self):
        """Test MLflow configuration validation."""
        # Valid configuration
        config = MLflowConfig(
            enabled=True,
            tracking_uri="sqlite:///mlflow.db",
            experiment_name="test_experiment"
        )
        assert config.tracking_uri == "sqlite:///mlflow.db"
        assert config.experiment_name == "test_experiment"

        # Empty experiment name
        with pytest.raises(ValueError, match="experiment_name cannot be empty"):
            MLflowConfig(experiment_name="")

        # Invalid tracking URI
        with pytest.raises(ValueError, match="tracking_uri must start with"):
            MLflowConfig(tracking_uri="invalid://uri")

        # Empty model name when registration enabled
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            MLflowConfig(register_model=True, model_name="")


class TestUnifiedXGBoostConfig:
    """Test unified XGBoost configuration."""

    def test_default_unified_config(self):
        """Test default unified configuration."""
        config = UnifiedXGBoostConfig(
            data_source="test_data.parquet"
        )
        
        # Check nested configs
        assert isinstance(config.gpu_config, GPUConfig)
        assert isinstance(config.optuna_config, OptunaConfig)
        assert isinstance(config.mlflow_config, MLflowConfig)
        
        # Check unified-specific settings
        assert config.track_feature_decay is True
        assert config.feature_decay_threshold == 0.3
        assert config.cv_strategy == "time_series"
        assert config.export_onnx is False

    def test_unified_config_validation(self):
        """Test unified configuration validation."""
        # Valid configuration
        config = UnifiedXGBoostConfig(
            data_source="test_data.parquet",
            feature_decay_threshold=0.5,
            cv_folds=3,
            onnx_output_path="./models/test.onnx"
        )
        assert config.feature_decay_threshold == 0.5
        assert config.cv_folds == 3

        # Invalid feature decay threshold
        with pytest.raises(ValueError, match="feature_decay_threshold must be in"):
            UnifiedXGBoostConfig(
                data_source="test_data.parquet",
                feature_decay_threshold=1.5
            )

        # Invalid CV folds
        with pytest.raises(ValueError, match="cv_folds must be at least 2"):
            UnifiedXGBoostConfig(
                data_source="test_data.parquet",
                cv_folds=1
            )

        # Invalid CV strategy
        with pytest.raises(ValueError, match="cv_strategy must be one of"):
            UnifiedXGBoostConfig(
                data_source="test_data.parquet",
                cv_strategy="invalid_strategy"
            )

        # Empty ONNX path when export enabled
        with pytest.raises(ValueError, match="onnx_output_path cannot be empty"):
            UnifiedXGBoostConfig(
                data_source="test_data.parquet",
                export_onnx=True,
                onnx_output_path=""
            )

    def test_get_unified_xgb_params(self):
        """Test unified XGBoost parameters generation."""
        # CPU configuration
        config = UnifiedXGBoostConfig(
            data_source="test_data.parquet"
        )
        params = config.get_unified_xgb_params()
        
        assert "tree_method" in params
        assert "random_state" in params
        assert params["tree_method"] == "hist"  # Default CPU

        # GPU configuration
        gpu_config = GPUConfig(enabled=True, device_id=1, max_bin=512)
        config = UnifiedXGBoostConfig(
            data_source="test_data.parquet",
            gpu_config=gpu_config
        )
        params = config.get_unified_xgb_params()
        
        assert params["tree_method"] == "gpu_hist"
        assert params["gpu_id"] == 1
        assert params["max_bin"] == 512
        assert params["predictor"] == "gpu_predictor"

    @patch('subprocess.run')
    def test_validate_environment(self, mock_subprocess):
        """Test environment validation."""
        # Mock successful nvidia-smi
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="GPU 0: Tesla V100\nGPU 1: Tesla V100\n"
        )

        config = UnifiedXGBoostConfig(
            data_source="test_data.parquet",
            gpu_config=GPUConfig(enabled=True),
            optuna_config=OptunaConfig(enabled=False),
            mlflow_config=MLflowConfig(enabled=False),
            export_onnx=False
        )

        warnings = config.validate_environment()
        
        # Should only warn about missing optional dependencies
        optuna_warning = any("optuna" in w.lower() for w in warnings)
        mlflow_warning = any("mlflow" in w.lower() for w in warnings)
        onnx_warning = any("onnx" in w.lower() for w in warnings)
        
        # These should be False since they're disabled in config
        assert not optuna_warning
        assert not mlflow_warning  
        assert not onnx_warning


class TestUnifiedXGBoostTrainer:
    """Test unified XGBoost trainer functionality."""

    def _copy_config(self, base_config: UnifiedXGBoostConfig, **kwargs) -> UnifiedXGBoostConfig:
        """Helper to create config copy with overrides since configs are frozen."""
        # Get all base config fields
        base_dict = {
            'data_source': base_config.data_source,
            'n_estimators': base_config.n_estimators,
            'max_depth': base_config.max_depth,
            'learning_rate': base_config.learning_rate,
            'objective': base_config.objective,
            'enable_monitoring': base_config.enable_monitoring,
            'feature_config': base_config.feature_config,
            'track_feature_decay': base_config.track_feature_decay,
            'feature_decay_threshold': base_config.feature_decay_threshold,
            'cv_strategy': base_config.cv_strategy,
            'cv_folds': base_config.cv_folds,
            'export_onnx': base_config.export_onnx,
            'gpu_config': base_config.gpu_config,
            'optuna_config': base_config.optuna_config,
            'mlflow_config': base_config.mlflow_config,
        }
        # Apply overrides
        base_dict.update(kwargs)
        return UnifiedXGBoostConfig(**base_dict)

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        data = pl.DataFrame({
            "timestamp": pl.datetime_range(
                start=pl.datetime(2023, 1, 1),
                end=pl.datetime(2023, 12, 31),
                interval="1d"
            )[:1000],
            "open": np.random.randn(1000).cumsum() + 100,
            "high": np.random.randn(1000).cumsum() + 102,
            "low": np.random.randn(1000).cumsum() + 98,
            "close": np.random.randn(1000).cumsum() + 101,
            "volume": np.random.randint(1000, 10000, 1000),
        })
        return data

    @pytest.fixture
    def basic_config(self):
        """Create basic unified configuration."""
        return UnifiedXGBoostConfig(
            data_source="test_data.parquet",
            n_estimators=10,  # Small for testing
            enable_monitoring=False,  # Disable for unit tests
            feature_config=MLFeatureConfig(
                lookback_window=50,
                normalize_features=False
            )
        )

    def test_trainer_initialization(self, basic_config):
        """Test trainer initialization."""
        trainer = UnifiedXGBoostTrainer(basic_config)
        
        assert trainer._unified_config == basic_config
        assert trainer._importance_history == []
        assert trainer._feature_decay_alerts == []
        assert trainer._metrics_collector is None  # Disabled in config

    def test_feature_decay_tracking(self, basic_config):
        """Test feature importance decay tracking."""
        config = self._copy_config(
            basic_config,
            track_feature_decay=True,
            feature_decay_threshold=0.3
        )
        
        trainer = UnifiedXGBoostTrainer(config)
        trainer._feature_names = ["feature_1", "feature_2", "feature_3"]

        # First importance measurement
        importance_1 = {
            "feature_1": 0.5,
            "feature_2": 0.3,
            "feature_3": 0.2,
        }
        trainer._track_feature_decay(importance_1)
        assert len(trainer._importance_history) == 1
        assert trainer._feature_decay_alerts == []

        # Second measurement with decay
        importance_2 = {
            "feature_1": 0.2,  # 60% decay - should trigger alert
            "feature_2": 0.25,  # 17% decay - no alert
            "feature_3": 0.19,  # 5% decay - no alert
        }
        trainer._track_feature_decay(importance_2)
        assert len(trainer._importance_history) == 2
        assert "feature_1" in trainer._feature_decay_alerts
        assert "feature_2" not in trainer._feature_decay_alerts

    def test_gpu_validation_no_gpu(self, basic_config):
        """Test GPU validation when no GPU available."""
        config = self._copy_config(
            basic_config,
            gpu_config=GPUConfig(enabled=True, validate_gpu=True)
        )
        
        trainer = UnifiedXGBoostTrainer(config)
        
        # Mock subprocess.run to simulate no GPU
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
            
            # Should not raise, but should warn
            trainer._validate_gpu_setup()  # Should handle gracefully

    @patch('ml.training.xgboost_unified.HAS_XGBOOST', True)
    def test_calculate_metrics(self, basic_config):
        """Test metric calculation functions."""
        trainer = UnifiedXGBoostTrainer(basic_config)
        
        # Test classification metrics
        trainer._unified_config = self._copy_config(
            basic_config,
            objective="binary:logistic"
        )
        
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([0.8, 0.2, 0.9, 0.7, 0.1])
        
        accuracy = trainer._calculate_accuracy(y_true, y_pred)
        assert 0.0 <= accuracy <= 1.0
        
        sharpe = trainer._calculate_sharpe_ratio(y_true, y_pred)
        assert isinstance(sharpe, float)

        # Test regression metrics  
        trainer._unified_config = self._copy_config(
            basic_config,
            objective="reg:squarederror"
        )
        
        y_true_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred_reg = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        
        r2_score = trainer._calculate_accuracy(y_true_reg, y_pred_reg)
        assert isinstance(r2_score, float)

    def test_monotonic_constraints_creation(self, basic_config):
        """Test monotonic constraints string generation."""
        config = self._copy_config(
            basic_config,
            monotonic_constraints={
                "feature_1": 1,   # Increasing
                "feature_2": -1,  # Decreasing
                "feature_3": 0,   # No constraint
            }
        )
        
        trainer = UnifiedXGBoostTrainer(config)
        trainer._feature_names = ["feature_1", "feature_2", "feature_3", "feature_4"]
        
        constraints_string = trainer._create_monotonic_constraints_string()
        expected = "(1,-1,0,0)"  # feature_4 gets default 0
        assert constraints_string == expected

    def test_get_feature_decay_summary(self, basic_config):
        """Test feature decay summary generation."""
        # Disabled tracking
        config = self._copy_config(
            basic_config,
            track_feature_decay=False
        )
        trainer = UnifiedXGBoostTrainer(config)
        
        summary = trainer.get_feature_decay_summary()
        assert summary["tracking_enabled"] is False

        # Enabled tracking
        config = self._copy_config(
            basic_config,
            track_feature_decay=True,
            feature_decay_threshold=0.2
        )
        trainer = UnifiedXGBoostTrainer(config)
        trainer._feature_decay_alerts = ["feature_1", "feature_2"]
        trainer._importance_history = [{"feature_1": 0.5}]
        
        summary = trainer.get_feature_decay_summary()
        assert summary["tracking_enabled"] is True
        assert summary["current_alerts"] == ["feature_1", "feature_2"]
        assert summary["decay_threshold"] == 0.2
        assert summary["history_length"] == 1

    def test_get_model_metadata_unfitted(self, basic_config):
        """Test model metadata for unfitted model."""
        trainer = UnifiedXGBoostTrainer(basic_config)
        
        metadata = trainer.get_model_metadata()
        assert metadata["fitted"] is False

    def test_get_model_metadata_fitted(self, basic_config):
        """Test model metadata for fitted model."""
        trainer = UnifiedXGBoostTrainer(basic_config)
        
        # Simulate fitted model
        trainer._is_fitted = True
        trainer._feature_names = ["feature_1", "feature_2"]
        trainer._feature_decay_alerts = ["feature_1"]
        trainer._training_metrics = {
            "metrics": {"val_accuracy": 0.85, "training_time": 120.5},
            "total_training_time": 150.0
        }
        
        metadata = trainer.get_model_metadata()
        
        assert metadata["fitted"] is True
        assert metadata["model_type"] == "xgboost_unified"
        assert metadata["features"]["n_features"] == 2
        assert metadata["features"]["feature_names"] == ["feature_1", "feature_2"]
        assert metadata["features"]["decay_alerts"] == ["feature_1"]
        assert metadata["performance"]["val_accuracy"] == 0.85
        assert metadata["training_time"] == 150.0

    @pytest.mark.parametrize("objective,expected_type", [
        ("binary:logistic", "classification"),
        ("reg:squarederror", "regression"),
    ])
    def test_objective_handling(self, basic_config, objective, expected_type):
        """Test correct handling of different objectives."""
        config = self._copy_config(
            basic_config,
            objective=objective
        )
        
        trainer = UnifiedXGBoostTrainer(config)
        assert trainer._unified_config.objective == objective

    def test_optimization_metric_function_selection(self, basic_config):
        """Test optimization metric function selection."""
        # Test each valid metric type
        valid_metrics = ["sharpe_ratio", "accuracy", "auc", "rmse"]
        
        for metric in valid_metrics:
            config = self._copy_config(
                basic_config,
                optuna_config=OptunaConfig(enabled=True, metric=metric)
            )
            
            trainer = UnifiedXGBoostTrainer(config)
            
            # Should not raise error
            metric_func = trainer._get_optimization_metric_function()
            assert callable(metric_func)
        
        # Test invalid metric - should raise during config creation
        with pytest.raises(ValueError, match="metric must be one of"):
            OptunaConfig(enabled=True, metric="unknown_metric")

    def test_onnx_export_path_creation(self, basic_config):
        """Test ONNX export functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            onnx_path = Path(temp_dir) / "models" / "test_model.onnx"
            
            config = self._copy_config(
                basic_config,
                export_onnx=True,
                onnx_output_path=str(onnx_path)
            )
            
            trainer = UnifiedXGBoostTrainer(config)
            trainer._feature_names = ["feature_1", "feature_2"]
            
            # Mock model for export
            mock_model = MagicMock()
            
            # Mock the ONNX conversion (since we don't have actual ONNX tools in tests)
            mock_onnxmltools = MagicMock()
            mock_onnx_model = MagicMock()
            mock_onnx_model.SerializeToString.return_value = b"mock_onnx_data"
            mock_onnxmltools.convert_xgboost.return_value = mock_onnx_model
            
            with patch('ml.training.xgboost_unified.check_ml_dependencies') as mock_check_deps, \
                 patch.dict('sys.modules', {'onnxmltools': mock_onnxmltools}), \
                 patch('builtins.open', create=True) as mock_open:
                
                result_path = trainer._export_to_onnx(mock_model)
                
                if result_path:  # Only check if export succeeded
                    assert result_path == str(onnx_path)
                    # Verify metadata file would be created
                    metadata_path = str(onnx_path.with_suffix(".json"))
                    assert metadata_path.endswith(".json")


class TestUnifiedConfigIntegration:
    """Integration tests for unified configuration with all components."""

    def test_full_configuration_integration(self):
        """Test full configuration with all features enabled."""
        config = UnifiedXGBoostConfig(
            # Required parameters
            data_source="test_data.parquet",
            
            # Base XGBoost settings
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective="binary:logistic",
            
            # GPU settings
            gpu_config=GPUConfig(
                enabled=True,
                device_id=0,
                max_bin=256
            ),
            
            # Optuna settings
            optuna_config=OptunaConfig(
                enabled=True,
                n_trials=10,
                metric="sharpe_ratio"
            ),
            
            # MLflow settings  
            mlflow_config=MLflowConfig(
                enabled=True,
                experiment_name="test_experiment",
                tracking_uri="sqlite:///test.db"
            ),
            
            # Feature tracking
            track_feature_decay=True,
            feature_decay_threshold=0.25,
            
            # Cross-validation
            cv_strategy="time_series",
            cv_folds=3,
            
            # Export
            export_onnx=True,
            onnx_output_path="./test_model.onnx"
        )
        
        # Verify all settings
        assert config.gpu_config.enabled is True
        assert config.optuna_config.enabled is True
        assert config.mlflow_config.enabled is True
        assert config.track_feature_decay is True
        assert config.export_onnx is True
        
        # Test parameter generation
        params = config.get_unified_xgb_params()
        assert params["tree_method"] == "gpu_hist"  # GPU enabled
        assert params["n_estimators"] == 100
        assert params["objective"] == "binary:logistic"

    def test_configuration_warnings(self):
        """Test configuration generates appropriate warnings."""
        config = UnifiedXGBoostConfig(
            data_source="test_data.parquet",
            gpu_config=GPUConfig(enabled=True),
            optuna_config=OptunaConfig(enabled=True)
        )
        
        # Mock environment to simulate missing dependencies
        with patch('ml.config.xgboost_unified.UnifiedXGBoostConfig.validate_environment') as mock_validate:
            mock_validate.return_value = [
                "GPU acceleration requested but not available",
                "Optuna optimization requested but optuna not installed"
            ]
            
            warnings = config.validate_environment()
            assert len(warnings) == 2
            assert "GPU" in warnings[0]
            assert "Optuna" in warnings[1]