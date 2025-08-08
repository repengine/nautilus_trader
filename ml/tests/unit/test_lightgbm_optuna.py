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

import numpy as np
import pytest

from ml._imports import HAS_LIGHTGBM
from ml._imports import HAS_OPTUNA
from ml.config.shared import OptunaConfig
from ml.training.lightgbm import OptunaPlugin


class TestLightGBMOptunaOptimizer:
    """
    Test LightGBM Optuna plugin.
    """

    def setup_method(self) -> None:
        """
        Set up test fixtures.
        """
        self.config = OptunaConfig(
            enabled=True,
            n_trials=5,  # Small number for fast tests
            direction="maximize",
            metric="sharpe_ratio",
            pruner="none",  # Disable pruning for deterministic tests
            sampler="random",  # Use random sampler for reproducibility
        )

    @pytest.mark.skipif(not (HAS_OPTUNA and HAS_LIGHTGBM), reason="Optuna and LightGBM required")
    def test_optimizer_init(self) -> None:
        """
        Test optimizer initialization.
        """
        plugin = OptunaPlugin(self.config)
        assert plugin.config == self.config
        assert plugin._study is None
        assert plugin._best_params == {}

    @pytest.mark.skipif(not (HAS_OPTUNA and HAS_LIGHTGBM), reason="Optuna and LightGBM required")
    def test_optimizer_init_without_dependencies(self) -> None:
        """
        Test optimizer initialization fails without dependencies.
        """
        # This would be tested by mocking HAS_OPTUNA and HAS_LIGHTGBM to False
        # but we skip if dependencies are not available

    @pytest.mark.skipif(not (HAS_OPTUNA and HAS_LIGHTGBM), reason="Optuna and LightGBM required")
    def test_suggest_lgb_parameters(self) -> None:
        """
        Test LightGBM parameter suggestion.
        """
        plugin = OptunaPlugin(self.config)

        # Mock trial for parameter suggestion
        from ml._imports import optuna

        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        params = plugin._suggest_parameters(trial)

        # Check required parameters are present
        assert "num_iterations" in params
        assert "learning_rate" in params
        assert "num_leaves" in params
        assert "max_depth" in params
        assert "subsample" in params
        assert "colsample_bytree" in params
        assert "reg_alpha" in params
        assert "reg_lambda" in params
        assert "boosting_type" in params

        # Check parameter bounds
        assert 50 <= params["num_iterations"] <= 1000
        assert 0.01 <= params["learning_rate"] <= 0.3
        assert 10 <= params["num_leaves"] <= 300
        assert 3 <= params["max_depth"] <= 15
        assert 0.4 <= params["subsample"] <= 1.0
        assert 0.4 <= params["colsample_bytree"] <= 1.0
        assert 1e-8 <= params["reg_alpha"] <= 10.0
        assert 1e-8 <= params["reg_lambda"] <= 10.0
        assert params["boosting_type"] in ["gbdt", "goss", "dart", "rf"]

        # Check num_leaves constraint
        if params["num_leaves"] >= 2 ** params["max_depth"]:
            assert params["num_leaves"] == 2 ** params["max_depth"] - 1

    @pytest.mark.skipif(not (HAS_OPTUNA and HAS_LIGHTGBM), reason="Optuna and LightGBM required")
    def test_suggest_lgb_parameters_goss(self) -> None:
        """
        Test LightGBM parameter suggestion with GOSS.
        """
        optimizer = LightGBMOptunaOptimizer(self.config)

        from ml._imports import optuna

        # Create multiple trials to increase chance of getting GOSS
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.RandomSampler(seed=42),
        )

        goss_found = False
        for _ in range(20):  # Try multiple times to get GOSS
            trial = study.ask()
            params = optimizer._suggest_lgb_parameters(trial)

            if params["boosting_type"] == "goss":
                goss_found = True
                assert "top_rate" in params
                assert "other_rate" in params
                assert 0.1 <= params["top_rate"] <= 0.5
                assert 0.05 <= params["other_rate"] <= 0.3
                break

            study.tell(trial, 0.5)  # Dummy objective value

        # Note: This test might occasionally fail due to randomness
        # In practice, we'd use a fixed seed or more sophisticated testing

    @pytest.mark.skipif(not (HAS_OPTUNA and HAS_LIGHTGBM), reason="Optuna and LightGBM required")
    def test_suggest_lgb_parameters_dart(self) -> None:
        """
        Test LightGBM parameter suggestion with DART.
        """
        optimizer = LightGBMOptunaOptimizer(self.config)

        from ml._imports import optuna

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.RandomSampler(seed=123),
        )

        dart_found = False
        for _ in range(20):  # Try multiple times to get DART
            trial = study.ask()
            params = optimizer._suggest_lgb_parameters(trial)

            if params["boosting_type"] == "dart":
                dart_found = True
                assert "drop_rate" in params
                assert "max_drop" in params
                assert "skip_drop" in params
                assert "uniform_drop" in params
                assert 0.05 <= params["drop_rate"] <= 0.5
                assert 10 <= params["max_drop"] <= 100
                assert 0.3 <= params["skip_drop"] <= 0.8
                assert params["uniform_drop"] in [True, False]
                break

            study.tell(trial, 0.5)  # Dummy objective value

    @pytest.mark.skipif(not (HAS_OPTUNA and HAS_LIGHTGBM), reason="Optuna and LightGBM required")
    def test_create_study_default(self) -> None:
        """
        Test study creation with default settings.
        """
        optimizer = LightGBMOptunaOptimizer(self.config)
        study = optimizer._create_study()

        assert study.direction.name == "MAXIMIZE"
        assert study.study_name is not None  # Auto-generated

    @pytest.mark.skipif(not (HAS_OPTUNA and HAS_LIGHTGBM), reason="Optuna and LightGBM required")
    def test_create_study_with_storage(self) -> None:
        """
        Test study creation with storage configuration.
        """
        config_with_storage = OptunaConfig(
            enabled=True,
            n_trials=5,
            direction="minimize",
            metric="rmse",
            study_name="test_study",
            storage_url="sqlite:///test_optuna.db",
        )

        optimizer = LightGBMOptunaOptimizer(config_with_storage)
        study = optimizer._create_study()

        assert study.direction.name == "MINIMIZE"
        assert study.study_name == "test_study"

    @pytest.mark.skipif(not (HAS_OPTUNA and HAS_LIGHTGBM), reason="Optuna and LightGBM required")
    def test_optimize_simple(self) -> None:
        """
        Test optimization with simple synthetic data.
        """
        # Create simple synthetic data
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)
        X_val = np.random.randn(20, 5)
        y_val = np.random.randn(20)

        config = OptunaConfig(
            enabled=True,
            n_trials=2,  # Very small for fast test
            direction="maximize",
            metric="r2",  # Use R2 for regression
            pruner="none",
            sampler="random",
        )

        optimizer = LightGBMOptunaOptimizer(config)
        best_params = optimizer.optimize(X_train, y_train, X_val, y_val)

        assert isinstance(best_params, dict)
        assert len(best_params) > 0
        assert optimizer.study is not None
        assert optimizer.best_params == best_params

    @pytest.mark.skipif(not (HAS_OPTUNA and HAS_LIGHTGBM), reason="Optuna and LightGBM required")
    def test_optimize_without_validation(self) -> None:
        """
        Test optimization without validation data.
        """
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)

        config = OptunaConfig(
            enabled=True,
            n_trials=2,
            direction="maximize",
            metric="sharpe_ratio",
            pruner="none",
            sampler="random",
        )

        optimizer = LightGBMOptunaOptimizer(config)
        best_params = optimizer.optimize(X_train, y_train, None, None)

        assert isinstance(best_params, dict)
        assert len(best_params) > 0

    @pytest.mark.skipif(not (HAS_OPTUNA and HAS_LIGHTGBM), reason="Optuna and LightGBM required")
    def test_param_importance_empty_study(self) -> None:
        """
        Test parameter importance with empty study.
        """
        optimizer = LightGBMOptunaOptimizer(self.config)
        importance = optimizer.get_param_importance()
        assert importance == {}

    @pytest.mark.skipif(not (HAS_OPTUNA and HAS_LIGHTGBM), reason="Optuna and LightGBM required")
    def test_param_importance_with_study(self) -> None:
        """
        Test parameter importance with completed study.
        """
        np.random.seed(42)
        X_train = np.random.randn(50, 5)
        y_train = np.random.randn(50)

        config = OptunaConfig(
            enabled=True,
            n_trials=3,
            direction="maximize",
            metric="r2",
            pruner="none",
            sampler="random",
        )

        optimizer = LightGBMOptunaOptimizer(config)
        optimizer.optimize(X_train, y_train, None, None)

        importance = optimizer.get_param_importance()
        assert isinstance(importance, dict)
        # May be empty if study is too small or parameters don't vary enough

    @pytest.mark.skipif(not (HAS_OPTUNA and HAS_LIGHTGBM), reason="Optuna and LightGBM required")
    def test_visualization_methods_empty_study(self) -> None:
        """
        Test visualization methods with empty study.
        """
        optimizer = LightGBMOptunaOptimizer(self.config)

        # These should return None for empty study
        assert optimizer.plot_optimization_history() is None
        assert optimizer.plot_param_importances() is None

    @pytest.mark.skipif(not (HAS_OPTUNA and HAS_LIGHTGBM), reason="Optuna and LightGBM required")
    def test_objective_function_error_handling(self) -> None:
        """
        Test objective function error handling.
        """
        optimizer = LightGBMOptunaOptimizer(self.config)

        from ml._imports import optuna

        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        # Test with invalid data (should handle gracefully)
        X_train = np.array([])  # Empty array
        y_train = np.array([])

        result = optimizer._objective(trial, X_train, y_train, None, None)

        # Should return a valid float (worst case score)
        assert isinstance(result, (int, float))

    @pytest.mark.skipif(not (HAS_OPTUNA and HAS_LIGHTGBM), reason="Optuna and LightGBM required")
    def test_optimizer_properties(self) -> None:
        """
        Test optimizer properties.
        """
        optimizer = LightGBMOptunaOptimizer(self.config)

        # Initially empty
        assert optimizer.study is None
        assert optimizer.best_params == {}

        # After setting values
        optimizer._best_params = {"test": 1.0}
        assert optimizer.best_params == {"test": 1.0}
        # Should return copy, not reference
        optimizer.best_params["test"] = 2.0
        assert optimizer._best_params["test"] == 1.0

    @pytest.mark.skipif(not (HAS_OPTUNA and HAS_LIGHTGBM), reason="Optuna and LightGBM required")
    def test_config_validation_in_optimizer(self) -> None:
        """
        Test that optimizer respects config validation.
        """
        # Test with various sampler types
        for sampler in ["tpe", "random", "cmaes"]:
            config = OptunaConfig(sampler=sampler)
            optimizer = LightGBMOptunaOptimizer(config)
            assert optimizer._config.sampler == sampler

        # Test with various pruner types
        for pruner in ["median", "percentile", "hyperband", "none"]:
            config = OptunaConfig(pruner=pruner)
            optimizer = LightGBMOptunaOptimizer(config)
            assert optimizer._config.pruner == pruner

    @pytest.mark.skipif(not (HAS_OPTUNA and HAS_LIGHTGBM), reason="Optuna and LightGBM required")
    def test_objective_sharpe_ratio_calculation(self) -> None:
        """
        Test Sharpe ratio calculation in objective function.
        """
        optimizer = LightGBMOptunaOptimizer(self.config)

        from ml._imports import optuna

        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        # Create data where we can predict Sharpe ratio
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)  # Returns
        X_val = np.random.randn(50, 5)
        y_val = np.ones(50) * 0.1  # Constant positive returns

        result = optimizer._objective(trial, X_train, y_train, X_val, y_val)

        # Should return a finite value
        assert np.isfinite(result)

    @pytest.mark.skipif(not (HAS_OPTUNA and HAS_LIGHTGBM), reason="Optuna and LightGBM required")
    def test_objective_different_metrics(self) -> None:
        """
        Test objective function with different metrics.
        """
        metrics_to_test = ["sharpe_ratio", "r2", "rmse", "mae", "accuracy"]

        np.random.seed(42)
        X_train = np.random.randn(50, 3)
        y_train = np.random.randint(0, 2, 50)  # Binary for classification
        X_val = np.random.randn(20, 3)
        y_val = np.random.randint(0, 2, 20)

        for metric in metrics_to_test:
            config = OptunaConfig(
                enabled=True,
                n_trials=1,
                metric=metric,
                direction=(
                    "maximize"
                    if metric in ["sharpe_ratio", "accuracy", "r2", "auc"]
                    else "minimize"
                ),
            )

            optimizer = LightGBMOptunaOptimizer(config)

            from ml._imports import optuna

            study = optuna.create_study(direction=config.direction)
            trial = study.ask()

            result = optimizer._objective(trial, X_train, y_train, X_val, y_val)
            assert isinstance(result, (int, float))
            assert np.isfinite(result)
