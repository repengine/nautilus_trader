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
Optuna hyperparameter optimization for LightGBM models.

This module provides hyperparameter optimization capabilities for LightGBM models using
Optuna, with LightGBM-specific parameter search spaces and pruning strategies.

"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ml._imports import HAS_LIGHTGBM
from ml._imports import HAS_OPTUNA
from ml._imports import check_ml_dependencies
from ml._imports import lgb
from ml._imports import optuna
from ml.config.lightgbm_unified import OptunaConfig


class LightGBMOptunaOptimizer:
    """
    Optuna hyperparameter optimizer for LightGBM models.

    This optimizer provides comprehensive hyperparameter search for LightGBM
    with intelligent search spaces, pruning, and early stopping integration.

    Parameters
    ----------
    config : OptunaConfig
        Configuration for Optuna optimization.

    """

    def __init__(self, config: OptunaConfig) -> None:
        """
        Initialize LightGBM Optuna optimizer.

        Parameters
        ----------
        config : OptunaConfig
            Configuration for hyperparameter optimization.

        Raises
        ------
        ImportError
            If required dependencies are not installed.

        """
        self._config = config

        # Validate dependencies
        if not HAS_OPTUNA:
            check_ml_dependencies(["optuna"])
        if not HAS_LIGHTGBM:
            check_ml_dependencies(["lightgbm"])

        self._study: optuna.Study | None = None
        self._best_params: dict[str, Any] = {}

    def _create_study(self) -> optuna.Study:
        """
        Create Optuna study with configured sampler and pruner.

        Returns
        -------
        optuna.Study
            Configured Optuna study.

        """
        # Configure sampler
        if self._config.sampler == "tpe":
            sampler = optuna.samplers.TPESampler()
        elif self._config.sampler == "random":
            sampler = optuna.samplers.RandomSampler()
        elif self._config.sampler == "cmaes":
            sampler = optuna.samplers.CmaEsSampler()
        else:
            sampler = optuna.samplers.TPESampler()  # Default fallback

        # Configure pruner
        if self._config.pruner == "median":
            pruner = optuna.pruners.MedianPruner()
        elif self._config.pruner == "percentile":
            pruner = optuna.pruners.PercentilePruner(percentile=25)
        elif self._config.pruner == "hyperband":
            pruner = optuna.pruners.HyperbandPruner()
        elif self._config.pruner == "none":
            pruner = optuna.pruners.NopPruner()
        else:
            pruner = optuna.pruners.MedianPruner()  # Default fallback

        # Create study
        if self._config.storage_url and self._config.study_name:
            study = optuna.create_study(
                direction=self._config.direction,
                sampler=sampler,
                pruner=pruner,
                study_name=self._config.study_name,
                storage=self._config.storage_url,
                load_if_exists=True,
            )
        else:
            study = optuna.create_study(
                direction=self._config.direction,
                sampler=sampler,
                pruner=pruner,
            )

        return study

    def _suggest_lgb_parameters(self, trial: optuna.Trial) -> dict[str, Any]:
        """
        Suggest LightGBM hyperparameters for optimization.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object for parameter suggestion.

        Returns
        -------
        dict[str, Any]
            Dictionary of suggested LightGBM parameters.

        """
        params = {
            # Core boosting parameters
            "num_iterations": trial.suggest_int("num_iterations", 50, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 10, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            # Sampling parameters
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 0, 7),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            # Regularization parameters
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            # Tree structure
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-5, 1e-1, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            # Other parameters
            "verbosity": -1,  # Silent
            "random_state": 42,
            "n_jobs": -1,
        }

        # Suggest boosting type (with GOSS and DART options)
        boosting_type = trial.suggest_categorical(
            "boosting_type",
            ["gbdt", "goss", "dart", "rf"],
        )
        params["boosting_type"] = boosting_type

        # Add GOSS-specific parameters
        if boosting_type == "goss":
            params["top_rate"] = trial.suggest_float("top_rate", 0.1, 0.5)
            params["other_rate"] = trial.suggest_float("other_rate", 0.05, 0.3)

        # Add DART-specific parameters
        elif boosting_type == "dart":
            params["drop_rate"] = trial.suggest_float("drop_rate", 0.05, 0.5)
            params["max_drop"] = trial.suggest_int("max_drop", 10, 100)
            params["skip_drop"] = trial.suggest_float("skip_drop", 0.3, 0.8)
            params["uniform_drop"] = trial.suggest_categorical("uniform_drop", [True, False])

        # Add RF-specific parameters
        elif boosting_type == "rf":
            params["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.4, 1.0)
            params["bagging_freq"] = trial.suggest_int("bagging_freq", 1, 7)
            params["feature_fraction"] = trial.suggest_float("feature_fraction", 0.4, 1.0)

        # Ensure num_leaves is reasonable for max_depth
        if params["num_leaves"] >= 2 ** params["max_depth"]:
            params["num_leaves"] = 2 ** params["max_depth"] - 1

        return params

    def _objective(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
    ) -> float:
        """
        Objective function for Optuna optimization.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object.
        X_train : np.ndarray
            Training feature matrix.
        y_train : np.ndarray
            Training target values.
        X_val : np.ndarray | None
            Validation feature matrix.
        y_val : np.ndarray | None
            Validation target values.

        Returns
        -------
        float
            Objective value to optimize.

        """
        # Suggest parameters
        params = self._suggest_lgb_parameters(trial)

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)

        valid_sets = []
        valid_names = []
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("validation")

        # Setup pruning callback
        pruning_callback = optuna.integration.LightGBMPruningCallback(
            trial,
            self._config.metric,
            valid_name="validation",
        )

        # Train model
        try:
            model = lgb.train(
                params,
                train_data,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=[pruning_callback] if valid_sets else None,
            )

            # Calculate objective value
            if self._config.metric == "sharpe_ratio":
                # Custom Sharpe ratio calculation
                if X_val is not None and y_val is not None:
                    predictions = model.predict(X_val, num_iteration=model.best_iteration)
                    returns = predictions * y_val  # Assuming y_val represents returns
                    if np.std(returns) > 0:
                        objective_value = np.mean(returns) / np.std(returns)
                    else:
                        objective_value = 0.0
                else:
                    objective_value = 0.0
            else:
                # Use validation score from model
                if valid_sets and model.best_score:
                    objective_value = model.best_score["validation"][self._config.metric]
                else:
                    # Fallback to training score or zero
                    objective_value = 0.0

                # Handle direction for standard metrics
                if self._config.direction == "minimize" and self._config.metric in [
                    "auc",
                    "accuracy",
                    "r2",
                ]:
                    objective_value = -objective_value
                elif self._config.direction == "maximize" and self._config.metric in [
                    "rmse",
                    "mae",
                ]:
                    objective_value = -objective_value

            return objective_value

        except optuna.TrialPruned:
            # Trial was pruned
            raise
        except Exception:
            # Return worst possible score for failed trials
            return float("-inf") if self._config.direction == "maximize" else float("inf")

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Run hyperparameter optimization.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix.
        y_train : np.ndarray
            Training target values.
        X_val : np.ndarray | None
            Validation feature matrix.
        y_val : np.ndarray | None
            Validation target values.

        Returns
        -------
        dict[str, Any]
            Best hyperparameters found.

        """
        self._study = self._create_study()

        # Define objective function with training data
        def objective_with_data(trial: optuna.Trial) -> float:
            return self._objective(trial, X_train, y_train, X_val, y_val)

        # Run optimization
        start_time = time.time()
        self._study.optimize(
            objective_with_data,
            n_trials=self._config.n_trials,
            timeout=self._config.timeout,
            show_progress_bar=True,
        )
        optimization_time = time.time() - start_time

        self._best_params = self._study.best_params.copy()

        # Log optimization results
        print(f"Optimization completed in {optimization_time:.2f}s")
        print(f"Best trial: {self._study.best_trial.number}")
        print(f"Best value: {self._study.best_value:.6f}")
        print(f"Best parameters: {self._best_params}")

        return self._best_params

    @property
    def study(self) -> optuna.Study | None:
        """
        Return the Optuna study object.

        Returns
        -------
        optuna.Study | None
            The Optuna study object if optimization has been run.

        """
        return self._study

    @property
    def best_params(self) -> dict[str, Any]:
        """
        Return the best parameters found.

        Returns
        -------
        dict[str, Any]
            Dictionary of best hyperparameters.

        """
        return self._best_params.copy()

    def get_param_importance(self) -> dict[str, float]:
        """
        Get parameter importance from completed study.

        Returns
        -------
        dict[str, float]
            Dictionary mapping parameter names to importance scores.

        """
        if self._study is None:
            return {}

        try:
            importance = optuna.importance.get_param_importances(self._study)
            return importance
        except Exception:
            return {}

    def plot_optimization_history(self) -> Any:
        """
        Plot optimization history.

        Returns
        -------
        Any
            Plotly figure object if available.

        """
        if self._study is None:
            return None

        try:
            return optuna.visualization.plot_optimization_history(self._study)
        except Exception:
            return None

    def plot_param_importances(self) -> Any:
        """
        Plot parameter importances.

        Returns
        -------
        Any
            Plotly figure object if available.

        """
        if self._study is None:
            return None

        try:
            return optuna.visualization.plot_param_importances(self._study)
        except Exception:
            return None


# Explicit exports
__all__ = [
    "LightGBMOptunaOptimizer",
]
