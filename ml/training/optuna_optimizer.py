"""
Optuna-based hyperparameter optimizer for XGBoost models.

This module provides sophisticated hyperparameter optimization capabilities using
Optuna, with support for various sampling strategies, pruning algorithms, and custom
objective functions tailored for financial machine learning applications.

"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ml._imports import HAS_OPTUNA
from ml._imports import HAS_XGBOOST
from ml._imports import check_ml_dependencies
from ml._imports import optuna
from ml._imports import xgb
from ml.config.shared import OptunaConfig


if TYPE_CHECKING:
    import optuna

# Configure module logger
logger = logging.getLogger(__name__)


class XGBoostOptunaOptimizer:
    """
    Optuna-based hyperparameter optimizer for XGBoost models.

    This optimizer provides intelligent hyperparameter search for XGBoost models
    with support for different sampling strategies, pruning techniques, and
    financial-specific objective functions.

    Features:
    - Multiple sampling algorithms (TPE, Random, CMA-ES)
    - Advanced pruning strategies (Median, Percentile, Hyperband)
    - Custom objective functions for financial metrics
    - GPU-aware optimization
    - Study persistence for long-running optimizations
    - Early stopping and timeout handling

    Parameters
    ----------
    config : OptunaConfig
        Configuration for Optuna optimization.

    """

    def __init__(self, config: OptunaConfig) -> None:
        """
        Initialize Optuna optimizer.

        Parameters
        ----------
        config : OptunaConfig
            Configuration for optimization parameters.

        """
        self.config = config
        self._optuna: Any = None
        self._study: Any = None

    def _ensure_optuna(self) -> None:
        """
        Ensure Optuna is available and initialize if needed.
        """
        if not HAS_OPTUNA:
            check_ml_dependencies(["optuna"])

        if self._optuna is None:
            self._optuna = optuna

    def create_study(self) -> Any:
        """
        Create or load Optuna study.

        Returns
        -------
        optuna.Study
            The study object for optimization.

        """
        self._ensure_optuna()

        # Configure study storage
        storage = None
        if self.config.storage_url is not None:
            storage = self._optuna.storages.RDBStorage(
                url=self.config.storage_url,
                engine_kwargs={"pool_pre_ping": True, "pool_recycle": 300},
            )

        # Create or load study
        study = self._optuna.create_study(
            study_name=self.config.study_name,
            storage=storage,
            direction=self.config.direction,
            pruner=self._create_pruner(),
            sampler=self._create_sampler(),
            load_if_exists=True,  # Resume existing study if available
        )

        self._study = study
        return study

    def _create_sampler(self) -> Any:
        """
        Create Optuna sampler based on configuration.

        Returns
        -------
        optuna.samplers.BaseSampler
            Configured sampler for hyperparameter optimization.

        """
        if self.config.sampler == "tpe":
            return self._optuna.samplers.TPESampler(
                n_startup_trials=10,
                n_ei_candidates=24,
                multivariate=True,
                constant_liar=True,
            )
        elif self.config.sampler == "random":
            return self._optuna.samplers.RandomSampler()
        elif self.config.sampler == "cmaes":
            return self._optuna.samplers.CmaEsSampler(
                n_startup_trials=10,
                restart_strategy="ipop",
            )
        elif self.config.sampler == "grid":
            # Note: Grid sampler requires predefined search space
            return self._optuna.samplers.GridSampler()
        else:
            # Default to TPE
            return self._optuna.samplers.TPESampler()

    def _create_pruner(self) -> Any:
        """
        Create Optuna pruner based on configuration.

        Returns
        -------
        optuna.pruners.BasePruner | None
            Configured pruner for early stopping of unpromising trials.

        """
        if self.config.pruner == "none":
            return None
        elif self.config.pruner == "median":
            return self._optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1,
            )
        elif self.config.pruner == "percentile":
            return self._optuna.pruners.PercentilePruner(
                percentile=25.0,
                n_startup_trials=5,
                n_warmup_steps=10,
            )
        elif self.config.pruner == "hyperband":
            return self._optuna.pruners.HyperbandPruner(
                min_resource=10,
                max_resource=1000,
                reduction_factor=3,
            )
        else:
            # Default to median pruner
            return self._optuna.pruners.MedianPruner()

    def sample_xgboost_params(
        self,
        trial: optuna.Trial,
        base_params: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Sample XGBoost hyperparameters for a trial.

        This method defines the search space for XGBoost hyperparameters,
        with ranges optimized for financial time series data.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object for parameter sampling.
        base_params : dict[str, Any]
            Base parameters that should not be optimized.

        Returns
        -------
        dict[str, Any]
            Dictionary of sampled XGBoost parameters.

        """
        # Core hyperparameters with financial-optimized ranges
        sampled_params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0, step=0.1),
            "gamma": trial.suggest_float("gamma", 0.0, 10.0, step=0.1),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 50.0, step=0.1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50.0, step=0.1),
        }

        # Advanced parameters for specific cases
        if base_params.get("objective") in ["reg:squarederror", "reg:logistic"]:
            # Additional parameters for regression
            sampled_params["huber_slope"] = trial.suggest_float("huber_slope", 0.1, 10.0)

        # Scale parameters based on tree method
        tree_method = base_params.get("tree_method", "hist")
        if tree_method == "gpu_hist":
            # GPU-specific optimizations
            sampled_params["max_bin"] = trial.suggest_int("max_bin", 64, 512, step=64)
            # Reduce some parameters for GPU memory efficiency
            sampled_params["max_depth"] = min(sampled_params["max_depth"], 10)

        # Combine with base parameters
        final_params = {**base_params, **sampled_params}

        return final_params

    def create_objective_function(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        X_val: npt.NDArray[np.float64],
        y_val: npt.NDArray[np.float64],
        base_params: dict[str, Any],
        metric_function: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float],
        early_stopping_rounds: int = 50,
    ) -> Callable[[Any], float]:
        """
        Create Optuna objective function for XGBoost optimization.

        Parameters
        ----------
        X_train : npt.NDArray[np.float64]
            Training features.
        y_train : npt.NDArray[np.float64]
            Training targets.
        X_val : npt.NDArray[np.float64]
            Validation features.
        y_val : npt.NDArray[np.float64]
            Validation targets.
        base_params : dict[str, Any]
            Base XGBoost parameters.
        metric_function : Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float]
            Function to calculate optimization metric.
        early_stopping_rounds : int, default 50
            Early stopping rounds for XGBoost training.

        Returns
        -------
        Callable
            Objective function for Optuna optimization.

        """
        if not HAS_XGBOOST:
            check_ml_dependencies(["xgboost"])

        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            params = self.sample_xgboost_params(trial, base_params)

            try:
                # Create model based on objective
                if params.get("objective") == "binary:logistic":
                    model = xgb.XGBClassifier(**params)
                else:
                    model = xgb.XGBRegressor(**params)

                # Set up pruning callback
                pruning_callback = None
                if self.config.pruner != "none":
                    pruning_callback = self._optuna.integration.XGBoostPruningCallback(
                        trial,
                        observation_key="validation_0-" + params.get("eval_metric", "rmse"),
                    )

                # Train model with early stopping
                callbacks = []
                if pruning_callback is not None:
                    callbacks.append(pruning_callback)

                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False,
                    callbacks=callbacks if callbacks else None,
                )

                # Generate predictions
                if hasattr(model, "predict_proba"):
                    # Classification: use probability of positive class
                    predictions = model.predict_proba(X_val)[:, 1]
                else:
                    # Regression: use direct predictions
                    predictions = model.predict(X_val)

                # Calculate metric
                metric_value = metric_function(y_val, predictions)

                # Handle NaN or infinite values
                if not np.isfinite(metric_value):
                    return float("-inf") if self.config.direction == "maximize" else float("inf")

                return metric_value

            except Exception as e:
                # Log error and return worst possible score
                logger.info(f"Trial {trial.number} failed: {e}")
                return float("-inf") if self.config.direction == "maximize" else float("inf")

        return objective

    def optimize(
        self,
        objective: Callable[[Any], float],
        n_trials: int | None = None,
        timeout: int | None = None,
        callbacks: list[Callable[[Any, Any], None]] | None = None,
    ) -> dict[str, Any]:
        """
        Run Optuna optimization.

        Parameters
        ----------
        objective : Callable
            Objective function to optimize.
        n_trials : int | None, optional
            Number of trials to run. Uses config value if None.
        timeout : int | None, optional
            Timeout in seconds. Uses config value if None.
        callbacks : list[Callable] | None, optional
            Optional callbacks for monitoring optimization progress.

        Returns
        -------
        dict[str, Any]
            Optimization results including best parameters and study statistics.

        """
        study = self.create_study()

        # Use config values if not specified
        n_trials = n_trials if n_trials is not None else self.config.n_trials
        timeout = timeout if timeout is not None else self.config.timeout

        # Add progress callback if enabled
        optimization_callbacks = callbacks or []

        try:
            # Run optimization
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                callbacks=optimization_callbacks,
                n_jobs=1,  # XGBoost with GPU doesn't support parallel trials
                show_progress_bar=True,
            )

            # Prepare results
            results = {
                "best_params": study.best_params,
                "best_value": study.best_value,
                "best_trial": study.best_trial,
                "n_trials": len(study.trials),
                "study": study,
                "optimization_history": [
                    {"trial": i, "value": trial.value}
                    for i, trial in enumerate(study.trials)
                    if trial.value is not None
                ],
            }

            # Add study statistics
            completed_trials = [
                t for t in study.trials if t.state == self._optuna.trial.TrialState.COMPLETE
            ]
            failed_trials = [
                t for t in study.trials if t.state == self._optuna.trial.TrialState.FAIL
            ]
            pruned_trials = [
                t for t in study.trials if t.state == self._optuna.trial.TrialState.PRUNED
            ]

            results["statistics"] = {
                "n_completed": len(completed_trials),
                "n_failed": len(failed_trials),
                "n_pruned": len(pruned_trials),
                "success_rate": len(completed_trials) / max(len(study.trials), 1),
            }

            return results

        except KeyboardInterrupt:
            logger.info("\nOptimization interrupted by user")
            if len(study.trials) > 0:
                return {
                    "best_params": study.best_params,
                    "best_value": study.best_value,
                    "n_trials": len(study.trials),
                    "interrupted": True,
                }
            else:
                raise

    def get_study_summary(self, study: Any | None = None) -> dict[str, Any]:
        """
        Get comprehensive summary of optimization study.

        Parameters
        ----------
        study : optuna.Study | None, optional
            Study to summarize. Uses internal study if None.

        Returns
        -------
        dict[str, Any]
            Study summary with statistics and insights.

        """
        if study is None:
            study = self._study

        if study is None:
            raise ValueError("No study available. Run optimization first.")

        # Basic statistics
        summary = {
            "study_name": study.study_name,
            "direction": study.direction,
            "n_trials": len(study.trials),
            "best_value": study.best_value,
            "best_params": study.best_params,
        }

        # Trial state distribution
        states: dict[str, int] = {}
        for trial in study.trials:
            state = trial.state.name
            states[state] = states.get(state, 0) + 1

        summary["trial_states"] = states

        # Parameter importance (if available)
        try:
            importance = self._optuna.importance.get_param_importances(study)
            summary["param_importance"] = dict(
                sorted(importance.items(), key=lambda x: x[1], reverse=True),
            )
        except Exception:
            summary["param_importance"] = {}

        # Optimization history
        if len(study.trials) > 0:
            values = [t.value for t in study.trials if t.value is not None]
            if values:
                summary["value_statistics"] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "median": np.median(values),
                }

        return summary


# Explicit exports
__all__ = [
    "LightGBMOptunaOptimizer",
    "XGBoostOptunaOptimizer",
]


class LightGBMOptunaOptimizer:
    """Thin wrapper reusing the XGBoost optimizer for LightGBM parameter sweeps."""

    def __init__(self, config: OptunaConfig) -> None:
        self._delegate = XGBoostOptunaOptimizer(config)

    def optimize(
        self,
        objective: Callable[[Any], float],
        n_trials: int | None = None,
        timeout: int | None = None,
        callbacks: list[Callable[[Any, Any], None]] | None = None,
    ) -> dict[str, Any]:
        return self._delegate.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=callbacks,
        )

    def get_study_summary(self, study: Any | None = None) -> dict[str, Any]:
        return self._delegate.get_study_summary(study)
