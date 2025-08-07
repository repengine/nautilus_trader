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
Unit tests for Optuna hyperparameter optimizer.

This test suite provides comprehensive coverage for the XGBoostOptunaOptimizer,
including parameter sampling, pruning strategies, optimization objective functions,
and study management functionality.

"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml._imports import HAS_OPTUNA
from ml.config.xgboost_unified import OptunaConfig
from ml.training.optuna_optimizer import XGBoostOptunaOptimizer


class TestXGBoostOptunaOptimizer:
    """Test XGBoost Optuna optimizer functionality."""

    @pytest.fixture
    def basic_config(self):
        """Create basic Optuna configuration."""
        return OptunaConfig(
            enabled=True,
            n_trials=10,
            direction="maximize",
            metric="sharpe_ratio",
            pruner="median",
            sampler="tpe"
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.randn(50, 5)
        y_val = np.random.randint(0, 2, 50)
        return X_train, y_train, X_val, y_val

    def test_optimizer_initialization(self, basic_config):
        """Test optimizer initialization."""
        optimizer = XGBoostOptunaOptimizer(basic_config)
        
        assert optimizer.config == basic_config
        assert optimizer._optuna is None
        assert optimizer._study is None

    @patch('ml.training.optuna_optimizer.HAS_OPTUNA', True)
    @patch('ml.training.optuna_optimizer.optuna')
    def test_ensure_optuna(self, mock_optuna, basic_config):
        """Test Optuna availability check."""
        optimizer = XGBoostOptunaOptimizer(basic_config)
        
        # First call should initialize
        optimizer._ensure_optuna()
        assert optimizer._optuna is not None

    @patch('ml.training.optuna_optimizer.HAS_OPTUNA', False)
    def test_ensure_optuna_not_available(self, basic_config):
        """Test Optuna availability check when not available."""
        optimizer = XGBoostOptunaOptimizer(basic_config)
        
        with pytest.raises(ImportError, match="Optuna required"):
            optimizer._ensure_optuna()

    @patch('ml.training.optuna_optimizer.HAS_OPTUNA', True)
    @patch('ml.training.optuna_optimizer.optuna')
    def test_create_study(self, mock_optuna, basic_config):
        """Test study creation."""
        optimizer = XGBoostOptunaOptimizer(basic_config)
        
        # Mock study
        mock_study = MagicMock()
        mock_optuna.create_study.return_value = mock_study
        
        study = optimizer.create_study()
        
        assert study == mock_study
        mock_optuna.create_study.assert_called_once()
        
        # Verify create_study arguments
        call_args = mock_optuna.create_study.call_args
        assert call_args[1]["direction"] == "maximize"
        assert call_args[1]["load_if_exists"] is True

    @patch('ml.training.optuna_optimizer.HAS_OPTUNA', True)
    @patch('ml.training.optuna_optimizer.optuna')
    def test_create_study_with_storage(self, mock_optuna, basic_config):
        """Test study creation with persistent storage."""
        config = OptunaConfig(
            **basic_config.__dict__,
            storage_url="sqlite:///test.db",
            study_name="test_study"
        )
        optimizer = XGBoostOptunaOptimizer(config)
        
        # Mock storage
        mock_storage = MagicMock()
        mock_optuna.storages.RDBStorage.return_value = mock_storage
        
        mock_study = MagicMock()
        mock_optuna.create_study.return_value = mock_study
        
        study = optimizer.create_study()
        
        # Verify storage was created
        mock_optuna.storages.RDBStorage.assert_called_once_with(
            url="sqlite:///test.db",
            engine_kwargs={"pool_pre_ping": True, "pool_recycle": 300}
        )
        
        # Verify study creation with storage
        call_args = mock_optuna.create_study.call_args
        assert call_args[1]["storage"] == mock_storage
        assert call_args[1]["study_name"] == "test_study"

    @pytest.mark.parametrize("sampler_type,expected_class", [
        ("tpe", "TPESampler"),
        ("random", "RandomSampler"),
        ("cmaes", "CmaEsSampler"),
        ("grid", "GridSampler"),
        ("unknown", "TPESampler"),  # Default fallback
    ])
    @patch('ml.training.optuna_optimizer.HAS_OPTUNA', True)
    @patch('ml.training.optuna_optimizer.optuna')
    def test_create_sampler(self, mock_optuna, basic_config, sampler_type, expected_class):
        """Test sampler creation for different types."""
        config = OptunaConfig(**basic_config.__dict__, sampler=sampler_type)
        optimizer = XGBoostOptunaOptimizer(config)
        
        # Mock samplers
        mock_sampler = MagicMock()
        getattr(mock_optuna.samplers, expected_class).return_value = mock_sampler
        
        sampler = optimizer._create_sampler()
        
        # Verify correct sampler was created
        getattr(mock_optuna.samplers, expected_class).assert_called_once()
        assert sampler == mock_sampler

    @pytest.mark.parametrize("pruner_type,expected_class", [
        ("none", None),
        ("median", "MedianPruner"),
        ("percentile", "PercentilePruner"), 
        ("hyperband", "HyperbandPruner"),
        ("unknown", "MedianPruner"),  # Default fallback
    ])
    @patch('ml.training.optuna_optimizer.HAS_OPTUNA', True)
    @patch('ml.training.optuna_optimizer.optuna')
    def test_create_pruner(self, mock_optuna, basic_config, pruner_type, expected_class):
        """Test pruner creation for different types."""
        config = OptunaConfig(**basic_config.__dict__, pruner=pruner_type)
        optimizer = XGBoostOptunaOptimizer(config)
        
        if expected_class is None:
            pruner = optimizer._create_pruner()
            assert pruner is None
        else:
            # Mock pruner
            mock_pruner = MagicMock()
            getattr(mock_optuna.pruners, expected_class).return_value = mock_pruner
            
            pruner = optimizer._create_pruner()
            
            # Verify correct pruner was created
            getattr(mock_optuna.pruners, expected_class).assert_called_once()
            assert pruner == mock_pruner

    @patch('ml.training.optuna_optimizer.HAS_OPTUNA', True)
    @patch('ml.training.optuna_optimizer.optuna')
    def test_sample_xgboost_params(self, mock_optuna, basic_config):
        """Test XGBoost parameter sampling."""
        optimizer = XGBoostOptunaOptimizer(basic_config)
        
        # Mock trial
        mock_trial = MagicMock()
        mock_trial.suggest_int.side_effect = [100, 6, 1]  # n_estimators, max_depth, min_child_weight
        mock_trial.suggest_float.side_effect = [
            0.1,   # learning_rate
            0.8,   # subsample
            0.8,   # colsample_bytree
            0.8,   # colsample_bylevel
            0.5,   # gamma
            1.0,   # reg_alpha
            1.0,   # reg_lambda
        ]
        
        base_params = {
            "objective": "binary:logistic",
            "random_state": 42
        }
        
        sampled_params = optimizer.sample_xgboost_params(mock_trial, base_params)
        
        # Verify base parameters are included
        assert sampled_params["objective"] == "binary:logistic"
        assert sampled_params["random_state"] == 42
        
        # Verify sampled parameters
        assert sampled_params["n_estimators"] == 100
        assert sampled_params["max_depth"] == 6
        assert sampled_params["learning_rate"] == 0.1

    @patch('ml.training.optuna_optimizer.HAS_OPTUNA', True)
    @patch('ml.training.optuna_optimizer.optuna')
    def test_sample_xgboost_params_gpu(self, mock_optuna, basic_config):
        """Test XGBoost parameter sampling for GPU training."""
        optimizer = XGBoostOptunaOptimizer(basic_config)
        
        # Mock trial
        mock_trial = MagicMock()
        mock_trial.suggest_int.side_effect = [100, 8, 1, 256]  # n_estimators, max_depth, min_child_weight, max_bin
        mock_trial.suggest_float.side_effect = [0.1, 0.8, 0.8, 0.8, 0.5, 1.0, 1.0]
        
        base_params = {
            "objective": "binary:logistic",
            "tree_method": "gpu_hist",
            "random_state": 42
        }
        
        sampled_params = optimizer.sample_xgboost_params(mock_trial, base_params)
        
        # Verify GPU-specific parameters
        assert sampled_params["tree_method"] == "gpu_hist"
        assert sampled_params["max_bin"] == 256
        # GPU training should limit max_depth
        assert sampled_params["max_depth"] <= 10

    @patch('ml.training.optuna_optimizer.HAS_OPTUNA', True)
    @patch('ml.training.optuna_optimizer.optuna')
    def test_sample_xgboost_params_regression(self, mock_optuna, basic_config):
        """Test XGBoost parameter sampling for regression."""
        optimizer = XGBoostOptunaOptimizer(basic_config)
        
        # Mock trial
        mock_trial = MagicMock()
        mock_trial.suggest_int.side_effect = [100, 6, 1]
        mock_trial.suggest_float.side_effect = [0.1, 0.8, 0.8, 0.8, 0.5, 1.0, 1.0, 1.5]  # Extra for huber_slope
        
        base_params = {
            "objective": "reg:squarederror",
            "random_state": 42
        }
        
        sampled_params = optimizer.sample_xgboost_params(mock_trial, base_params)
        
        # Verify regression-specific parameters
        assert sampled_params["objective"] == "reg:squarederror"
        assert sampled_params["huber_slope"] == 1.5

    @patch('ml.training.optuna_optimizer.HAS_XGBOOST', True)
    @patch('ml.training.optuna_optimizer.HAS_OPTUNA', True)
    @patch('ml.training.optuna_optimizer.xgb')
    @patch('ml.training.optuna_optimizer.optuna')
    def test_create_objective_function(self, mock_optuna, mock_xgb, basic_config, sample_data):
        """Test objective function creation."""
        optimizer = XGBoostOptunaOptimizer(basic_config)
        X_train, y_train, X_val, y_val = sample_data
        
        base_params = {"objective": "binary:logistic"}
        metric_function = lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2)
        
        # Create objective
        objective = optimizer.create_objective_function(
            X_train, y_train, X_val, y_val, base_params, metric_function
        )
        
        assert callable(objective)
        
        # Mock trial and XGBoost model
        mock_trial = MagicMock()
        mock_trial.suggest_int.side_effect = [100, 6, 1]
        mock_trial.suggest_float.side_effect = [0.1, 0.8, 0.8, 0.8, 0.5, 1.0, 1.0]
        
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.random.rand(len(y_val), 2)
        mock_xgb.XGBClassifier.return_value = mock_model
        
        # Mock pruning callback
        mock_callback = MagicMock()
        mock_optuna.integration.XGBoostPruningCallback.return_value = mock_callback
        
        # Execute objective
        result = objective(mock_trial)
        
        # Verify model was created and trained
        mock_xgb.XGBClassifier.assert_called_once()
        mock_model.fit.assert_called_once()
        mock_model.predict_proba.assert_called_once()
        
        # Verify result is numeric
        assert isinstance(result, (int, float))

    @patch('ml.training.optuna_optimizer.HAS_XGBOOST', True)
    @patch('ml.training.optuna_optimizer.HAS_OPTUNA', True)  
    @patch('ml.training.optuna_optimizer.xgb')
    @patch('ml.training.optuna_optimizer.optuna')
    def test_create_objective_function_regression(self, mock_optuna, mock_xgb, basic_config, sample_data):
        """Test objective function creation for regression."""
        optimizer = XGBoostOptunaOptimizer(basic_config)
        X_train, y_train, X_val, y_val = sample_data
        
        base_params = {"objective": "reg:squarederror"}
        metric_function = lambda y_true, y_pred: -np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        objective = optimizer.create_objective_function(
            X_train, y_train, X_val, y_val, base_params, metric_function
        )
        
        # Mock trial and regressor
        mock_trial = MagicMock()
        mock_trial.suggest_int.side_effect = [100, 6, 1]
        mock_trial.suggest_float.side_effect = [0.1, 0.8, 0.8, 0.8, 0.5, 1.0, 1.0, 1.5]
        
        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.rand(len(y_val))
        mock_xgb.XGBRegressor.return_value = mock_model
        
        result = objective(mock_trial)
        
        # Verify regressor was used
        mock_xgb.XGBRegressor.assert_called_once()
        mock_model.predict.assert_called_once()

    @patch('ml.training.optuna_optimizer.HAS_XGBOOST', True)
    @patch('ml.training.optuna_optimizer.HAS_OPTUNA', True)
    @patch('ml.training.optuna_optimizer.xgb')
    @patch('ml.training.optuna_optimizer.optuna')
    def test_objective_function_exception_handling(self, mock_optuna, mock_xgb, basic_config, sample_data):
        """Test objective function exception handling."""
        optimizer = XGBoostOptunaOptimizer(basic_config)
        X_train, y_train, X_val, y_val = sample_data
        
        base_params = {"objective": "binary:logistic"}
        metric_function = lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2)
        
        objective = optimizer.create_objective_function(
            X_train, y_train, X_val, y_val, base_params, metric_function
        )
        
        # Mock trial
        mock_trial = MagicMock()
        mock_trial.number = 5
        mock_trial.suggest_int.side_effect = [100, 6, 1]
        mock_trial.suggest_float.side_effect = [0.1, 0.8, 0.8, 0.8, 0.5, 1.0, 1.0]
        
        # Make XGBoost raise exception
        mock_xgb.XGBClassifier.side_effect = Exception("Training failed")
        
        # Should return worst score for maximization
        result = objective(mock_trial)
        assert result == float("-inf")

    @patch('ml.training.optuna_optimizer.HAS_OPTUNA', True)
    @patch('ml.training.optuna_optimizer.optuna')
    def test_optimize(self, mock_optuna, basic_config):
        """Test optimization execution."""
        optimizer = XGBoostOptunaOptimizer(basic_config)
        
        # Mock study
        mock_study = MagicMock()
        mock_study.best_params = {"n_estimators": 150, "max_depth": 8}
        mock_study.best_value = 0.85
        mock_study.best_trial = MagicMock()
        mock_study.trials = [MagicMock() for _ in range(5)]
        
        # Mock trials with states
        for i, trial in enumerate(mock_study.trials):
            if i < 3:
                trial.state = mock_optuna.trial.TrialState.COMPLETE
                trial.value = 0.8 + i * 0.01
            elif i == 3:
                trial.state = mock_optuna.trial.TrialState.FAIL
                trial.value = None
            else:
                trial.state = mock_optuna.trial.TrialState.PRUNED  
                trial.value = None
        
        mock_optuna.create_study.return_value = mock_study
        
        # Mock objective function
        objective = MagicMock()
        
        results = optimizer.optimize(objective, n_trials=5)
        
        # Verify study optimization was called
        mock_study.optimize.assert_called_once_with(
            objective,
            n_trials=5,
            timeout=None,
            callbacks=[],
            n_jobs=1,
            show_progress_bar=True,
        )
        
        # Verify results structure
        assert results["best_params"] == {"n_estimators": 150, "max_depth": 8}
        assert results["best_value"] == 0.85
        assert results["n_trials"] == 5
        assert "study" in results
        assert "optimization_history" in results
        assert "statistics" in results
        
        # Verify statistics
        stats = results["statistics"]
        assert stats["n_completed"] == 3
        assert stats["n_failed"] == 1
        assert stats["n_pruned"] == 1
        assert stats["success_rate"] == 0.6  # 3/5

    @patch('ml.training.optuna_optimizer.HAS_OPTUNA', True)
    @patch('ml.training.optuna_optimizer.optuna')
    def test_optimize_with_keyboard_interrupt(self, mock_optuna, basic_config):
        """Test optimization handling keyboard interrupt."""
        optimizer = XGBoostOptunaOptimizer(basic_config)
        
        # Mock study
        mock_study = MagicMock()
        mock_study.best_params = {"n_estimators": 100}
        mock_study.best_value = 0.7
        mock_study.trials = [MagicMock()]
        mock_study.optimize.side_effect = KeyboardInterrupt("User interrupted")
        
        mock_optuna.create_study.return_value = mock_study
        
        objective = MagicMock()
        
        results = optimizer.optimize(objective)
        
        # Should return partial results
        assert "interrupted" in results
        assert results["interrupted"] is True
        assert results["best_params"] == {"n_estimators": 100}

    @patch('ml.training.optuna_optimizer.HAS_OPTUNA', True)
    @patch('ml.training.optuna_optimizer.optuna')
    def test_get_study_summary(self, mock_optuna, basic_config):
        """Test study summary generation."""
        optimizer = XGBoostOptunaOptimizer(basic_config)
        
        # Mock study
        mock_study = MagicMock()
        mock_study.study_name = "test_study"
        mock_study.direction.name = "MAXIMIZE"
        mock_study.best_params = {"n_estimators": 200}
        mock_study.best_value = 0.9
        mock_study.trials = []
        
        # Create mock trials with different states
        for i in range(10):
            trial = MagicMock()
            if i < 6:
                trial.state.name = "COMPLETE"
                trial.value = 0.8 + i * 0.01
            elif i < 8:
                trial.state.name = "FAILED"
                trial.value = None
            else:
                trial.state.name = "PRUNED"
                trial.value = None
            mock_study.trials.append(trial)
        
        # Mock parameter importance
        mock_optuna.importance.get_param_importances.return_value = {
            "n_estimators": 0.6,
            "max_depth": 0.4
        }
        
        optimizer._study = mock_study
        
        summary = optimizer.get_study_summary()
        
        # Verify summary structure
        assert summary["study_name"] == "test_study"
        assert summary["direction"] == "MAXIMIZE"
        assert summary["n_trials"] == 10
        assert summary["best_value"] == 0.9
        assert summary["best_params"] == {"n_estimators": 200}
        
        # Verify trial states
        assert summary["trial_states"]["COMPLETE"] == 6
        assert summary["trial_states"]["FAILED"] == 2
        assert summary["trial_states"]["PRUNED"] == 2
        
        # Verify parameter importance
        assert summary["param_importance"]["n_estimators"] == 0.6
        assert summary["param_importance"]["max_depth"] == 0.4

    def test_get_study_summary_no_study(self, basic_config):
        """Test study summary with no study available."""
        optimizer = XGBoostOptunaOptimizer(basic_config)
        
        with pytest.raises(ValueError, match="No study available"):
            optimizer.get_study_summary()

    @patch('ml.training.optuna_optimizer.HAS_OPTUNA', True)
    @patch('ml.training.optuna_optimizer.optuna')
    def test_get_study_summary_with_values(self, mock_optuna, basic_config):
        """Test study summary with trial values for statistics."""
        optimizer = XGBoostOptunaOptimizer(basic_config)
        
        # Mock study with values
        mock_study = MagicMock()
        mock_study.study_name = "test_study"
        mock_study.direction.name = "MAXIMIZE" 
        mock_study.best_params = {}
        mock_study.best_value = 0.85
        mock_study.trials = []
        
        # Create trials with values
        values = [0.1, 0.3, 0.8, 0.7, 0.9]
        for val in values:
            trial = MagicMock()
            trial.state.name = "COMPLETE"
            trial.value = val
            mock_study.trials.append(trial)
        
        # Mock parameter importance to avoid exception
        mock_optuna.importance.get_param_importances.return_value = {}
        
        optimizer._study = mock_study
        
        summary = optimizer.get_study_summary()
        
        # Verify value statistics
        assert "value_statistics" in summary
        stats = summary["value_statistics"]
        assert stats["min"] == 0.1
        assert stats["max"] == 0.9
        assert stats["mean"] == np.mean(values)
        assert stats["std"] == np.std(values)
        assert stats["median"] == np.median(values)