"""
Hyperparameter optimization component for BaseMLTrainer decomposition.

This module provides the HyperparameterComponent which encapsulates hyperparameter
optimization logic from BaseMLTrainer (lines 493-502 and 514-816), including:
- Optuna enablement check (_should_use_optuna)
- Hyperparameter optimization (_optimize_hyperparameters)
- Metric resolution and calculation
- Sampler and pruner factory methods
- Sharpe ratio metric calculation for trading models

Following Universal ML Architecture Pattern 2: Protocol-First Interface Design.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt

from ml._imports import HAS_OPTUNA
from ml._imports import HAS_SKLEARN
from ml._imports import check_ml_dependencies
from ml._imports import optuna


if TYPE_CHECKING:
    from ml.config.base import MLTrainingConfig
    from ml.config.shared import OptunaConfig


logger = logging.getLogger(__name__)


class HyperparameterTrainerProtocol(Protocol):
    """
    Protocol for trainer interaction with hyperparameter component.

    Defines the interface that any trainer must implement to work with
    the HyperparameterComponent. This follows Protocol-First design
    (Pattern 2) to enable structural typing without inheritance coupling.

    Attributes
    ----------
    _config : MLTrainingConfig
        Training configuration with Optuna settings.
    _optuna_study : optuna.Study | None
        Optuna study instance after optimization.

    """

    _config: MLTrainingConfig
    _optuna_study: optuna.Study | None

    def _log_info(self, message: str, *args: object, **kwargs: Any) -> None:
        """Log info message."""
        ...

    def _log_warning(self, message: str, *args: object, **kwargs: Any) -> None:
        """Log warning message."""
        ...

    def _create_model(self, params: dict[str, Any]) -> Any:
        """Create a new model instance with given parameters."""
        ...

    def _suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest hyperparameters for Optuna trial."""
        ...

    def _train_model(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        X_val: npt.NDArray[np.float64],
        y_val: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Train model and return results dict with 'model' key."""
        ...

    def predict(
        self,
        model: Any,
        X: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> npt.NDArray[np.float32] | npt.NDArray[np.float64]:
        """Make predictions with trained model."""
        ...


class HyperparameterComponent:
    """
    Component responsible for hyperparameter optimization operations.

    This component encapsulates the hyperparameter optimization logic from
    BaseMLTrainer (lines 493-502 and 514-816), implementing:
    - Optuna optimization with configurable metrics
    - Sampler and pruner factory methods (TPE, Random, CMA-ES, etc.)
    - Metric calculation (accuracy, RMSE, MAE, R2, AUC, Sharpe ratio)
    - Classification problem detection

    The component delegates model creation and training to the trainer instance
    through the HyperparameterTrainerProtocol interface, following Protocol-First design.

    Parameters
    ----------
    trainer : HyperparameterTrainerProtocol
        The trainer instance that implements the HyperparameterTrainerProtocol.

    Example
    -------
    >>> from ml.training.common import HyperparameterComponent
    >>> # trainer is an instance implementing HyperparameterTrainerProtocol
    >>> hp_component = HyperparameterComponent(trainer)
    >>> if hp_component._should_use_optuna():
    ...     best_params = hp_component._optimize_hyperparameters(
    ...         X_train, y_train, X_val, y_val
    ...     )
    ...     print(f"Best params: {best_params}")

    """

    def __init__(self, trainer: HyperparameterTrainerProtocol) -> None:
        """
        Initialize the hyperparameter optimization component with a trainer reference.

        Parameters
        ----------
        trainer : HyperparameterTrainerProtocol
            The trainer instance for delegation.

        """
        self._trainer = trainer

    def _should_use_optuna(self) -> bool:
        """
        Check if Optuna hyperparameter optimization should be used.

        Optuna is enabled when:
        1. HAS_OPTUNA flag is True (optuna is installed)
        2. Config has optuna_config attribute
        3. optuna_config is not None
        4. optuna_config.enabled is True (default)

        Returns
        -------
        bool
            True if Optuna optimization should be performed, False otherwise.

        Example
        -------
        >>> hp_component = HyperparameterComponent(trainer)
        >>> if hp_component._should_use_optuna():
        ...     best_params = hp_component._optimize_hyperparameters(X, y, X_val, y_val)

        """
        return (
            HAS_OPTUNA
            and hasattr(self._trainer._config, "optuna_config")
            and self._trainer._config.optuna_config is not None
            and getattr(self._trainer._config.optuna_config, "enabled", True)
        )

    def _optimize_hyperparameters(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        X_val: npt.NDArray[np.float64],
        y_val: npt.NDArray[np.float64],
        *,
        validation_returns: npt.NDArray[np.float64] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.

        Creates an Optuna study with configurable sampler and pruner,
        runs n_trials optimization iterations, and returns the best parameters.

        Parameters
        ----------
        X_train : npt.NDArray[np.float64]
            Training features array of shape (n_samples, n_features).
        y_train : npt.NDArray[np.float64]
            Training targets array of shape (n_samples,).
        X_val : npt.NDArray[np.float64]
            Validation features array of shape (n_val_samples, n_features).
        y_val : npt.NDArray[np.float64]
            Validation targets array of shape (n_val_samples,).
        validation_returns : npt.NDArray[np.float64] | None, optional
            Returns array for Sharpe ratio calculation.
        **kwargs : Any
            Additional parameters passed to model training.

        Returns
        -------
        dict[str, Any]
            Best hyperparameters found during optimization.

        Raises
        ------
        ImportError
            If Optuna is not installed (via check_ml_dependencies).

        Example
        -------
        >>> best_params = hp_component._optimize_hyperparameters(
        ...     X_train, y_train, X_val, y_val,
        ...     validation_returns=returns_array
        ... )
        >>> print(f"Best learning rate: {best_params.get('learning_rate')}")

        """
        if not HAS_OPTUNA:
            check_ml_dependencies(["optuna"])

        self._trainer._log_info("Starting hyperparameter optimization with Optuna")

        optuna_kwargs = dict(kwargs)
        optuna_returns = optuna_kwargs.pop("validation_returns", validation_returns)
        optuna_cfg: OptunaConfig | None = getattr(self._trainer._config, "optuna_config", None)
        metric_name = self._resolve_optuna_metric_name(y_train, optuna_cfg)
        direction = self._resolve_optuna_direction(metric_name, optuna_cfg)

        sampler: optuna.samplers.BaseSampler = optuna.samplers.TPESampler(seed=42)
        pruner: optuna.pruners.BasePruner | None = None
        if optuna_cfg is not None:
            try:
                sampler = self._build_optuna_sampler(optuna_cfg)
            except Exception as exc:  # pragma: no cover - defensive fallback
                self._trainer._log_warning(
                    "Optuna sampler '%s' unavailable (%s); falling back to TPE",
                    optuna_cfg.sampler,
                    exc,
                )
            try:
                pruner = self._build_optuna_pruner(optuna_cfg)
            except Exception as exc:  # pragma: no cover - defensive fallback
                self._trainer._log_warning(
                    "Optuna pruner '%s' unavailable (%s); disabling pruning",
                    optuna_cfg.pruner,
                    exc,
                )

        def objective(trial: optuna.Trial) -> float:
            params = self._trainer._suggest_hyperparameters(trial)
            params.update(optuna_kwargs)

            model = self._trainer._create_model(params)
            if hasattr(model, "fit"):
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
            else:
                model = self._train_with_params(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    params,
                )

            predictions = self._trainer.predict(model, X_val)
            score = self._calculate_optuna_metric(
                metric_name,
                y_val,
                predictions,
                validation_returns=optuna_returns,
            )
            return score

        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
        )

        n_trials = getattr(optuna_cfg, "n_trials", 100)
        timeout = getattr(optuna_cfg, "timeout", None)
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        self._trainer._optuna_study = study
        best_params = study.best_params
        self._trainer._log_info(f"Best parameters found: {best_params}")

        return dict(best_params)

    def _resolve_optuna_metric_name(
        self,
        y_train: npt.NDArray[np.float64],
        optuna_cfg: OptunaConfig | None,
    ) -> str:
        """
        Resolve the metric name for Optuna optimization.

        Uses the metric from config if specified, otherwise defaults to
        'accuracy' for classification problems or 'rmse' for regression.

        Parameters
        ----------
        y_train : npt.NDArray[np.float64]
            Training targets to determine problem type.
        optuna_cfg : OptunaConfig | None
            Optuna configuration with optional metric setting.

        Returns
        -------
        str
            The metric name to optimize (e.g., 'accuracy', 'rmse', 'sharpe_ratio').

        """
        metric = (optuna_cfg.metric if optuna_cfg is not None else "") or ""
        if metric:
            return metric
        return "accuracy" if self._is_classification_problem(y_train) else "rmse"

    def _resolve_optuna_direction(
        self,
        metric_name: str,
        optuna_cfg: OptunaConfig | None,
    ) -> str:
        """
        Resolve the optimization direction for Optuna.

        Determines whether to maximize or minimize based on the metric.
        Logs a warning if config direction conflicts with metric's natural direction.

        Parameters
        ----------
        metric_name : str
            The metric being optimized.
        optuna_cfg : OptunaConfig | None
            Optuna configuration with optional direction setting.

        Returns
        -------
        str
            Either 'maximize' or 'minimize'.

        """
        metric_direction = self._optuna_direction_for_metric(metric_name)
        if optuna_cfg is not None and optuna_cfg.direction != metric_direction:
            self._trainer._log_warning(
                "Optuna direction '%s' mismatches metric '%s'; using '%s'",
                optuna_cfg.direction,
                metric_name,
                metric_direction,
            )
        return metric_direction

    @staticmethod
    def _optuna_direction_for_metric(metric_name: str) -> str:
        """
        Get the natural optimization direction for a metric.

        Error metrics (RMSE, MAE) should be minimized.
        Performance metrics (accuracy, AUC, Sharpe, R2) should be maximized.

        Parameters
        ----------
        metric_name : str
            The metric name (case-insensitive).

        Returns
        -------
        str
            'minimize' for error metrics, 'maximize' for others.

        Example
        -------
        >>> HyperparameterComponent._optuna_direction_for_metric("rmse")
        'minimize'
        >>> HyperparameterComponent._optuna_direction_for_metric("accuracy")
        'maximize'

        """
        metric = metric_name.lower()
        if metric in {"rmse", "mae"}:
            return "minimize"
        return "maximize"

    def _calculate_optuna_metric(
        self,
        metric_name: str,
        y_true: npt.NDArray[np.float64],
        y_pred: npt.NDArray[np.float32] | npt.NDArray[np.float64],
        *,
        validation_returns: npt.NDArray[np.float64] | None = None,
    ) -> float:
        """
        Calculate the specified optimization metric.

        Supports multiple metrics:
        - 'accuracy': Classification accuracy after thresholding
        - 'rmse': Root Mean Square Error
        - 'mae': Mean Absolute Error
        - 'r2': R-squared coefficient
        - 'auc': Area Under ROC Curve (requires sklearn)
        - 'sharpe_ratio': Annualized Sharpe ratio (requires validation_returns)

        Parameters
        ----------
        metric_name : str
            The metric to calculate (case-insensitive).
        y_true : npt.NDArray[np.float64]
            Ground truth labels/values.
        y_pred : npt.NDArray[np.float32] | npt.NDArray[np.float64]
            Model predictions (probabilities or values).
        validation_returns : npt.NDArray[np.float64] | None, optional
            Returns array for Sharpe ratio calculation.

        Returns
        -------
        float
            The calculated metric value.

        Notes
        -----
        Falls back to accuracy if:
        - 'sharpe_ratio' requested without validation_returns
        - 'auc' requested without sklearn
        - Unknown metric for classification problem

        """
        metric = metric_name.lower()
        targets = np.asarray(y_true, dtype=np.float64).reshape(-1)
        predictions = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        is_classification = self._is_classification_problem(targets)

        if metric == "sharpe_ratio":
            if validation_returns is None:
                self._trainer._log_warning(
                    "Optuna metric '%s' requested without validation returns; using accuracy",
                    metric_name,
                )
                return self._calculate_optuna_metric("accuracy", targets, predictions)
            return self._calculate_sharpe_metric(predictions, targets, validation_returns)

        if metric == "auc":
            if not HAS_SKLEARN:
                self._trainer._log_warning(
                    "sklearn unavailable; falling back to accuracy for AUC metric"
                )
                metric = "accuracy"
            else:
                from sklearn.metrics import roc_auc_score

                try:
                    return float(roc_auc_score(targets, predictions))
                except ValueError as exc:
                    self._trainer._log_warning(
                        "roc_auc_score failed (%s); falling back to accuracy",
                        exc,
                    )
                    metric = "accuracy"

        if metric == "accuracy":
            labels = self._probabilities_to_labels(predictions)
            return float(np.mean(labels == targets.astype(int)))

        if metric == "rmse":
            return float(np.sqrt(np.mean((targets - predictions) ** 2)))

        if metric == "mae":
            return float(np.mean(np.abs(targets - predictions)))

        if metric == "r2":
            ss_res = np.sum((targets - predictions) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            return float(1.0 - (ss_res / ss_tot if ss_tot != 0 else 0.0))

        # Default fallback for unknown metrics
        if is_classification:
            labels = self._probabilities_to_labels(predictions)
            return float(np.mean(labels == targets.astype(int)))

        return float(-np.mean((targets - predictions) ** 2))

    @staticmethod
    def _probabilities_to_labels(
        predictions: npt.NDArray[np.float64],
        threshold: float = 0.5,
    ) -> npt.NDArray[np.int64]:
        """
        Convert probability predictions to binary labels.

        Parameters
        ----------
        predictions : npt.NDArray[np.float64]
            Probability predictions in range [0, 1].
        threshold : float, default 0.5
            Decision threshold for binary classification.

        Returns
        -------
        npt.NDArray[np.int64]
            Binary labels (0 or 1).

        Example
        -------
        >>> probs = np.array([0.3, 0.7, 0.5, 0.9])
        >>> labels = HyperparameterComponent._probabilities_to_labels(probs)
        >>> # labels = [0, 1, 1, 1]  (0.5 is at boundary, rounds to 1)

        """
        return (predictions >= threshold).astype(np.int64)

    def _calculate_sharpe_metric(
        self,
        predictions: npt.NDArray[np.float64],
        targets: npt.NDArray[np.float64],
        validation_returns: npt.NDArray[np.float64],
    ) -> float:
        """
        Calculate annualized Sharpe ratio from predictions and returns.

        Converts predictions to trading signals, applies them to returns,
        and computes the annualized Sharpe ratio assuming 252 trading days.

        Parameters
        ----------
        predictions : npt.NDArray[np.float64]
            Model predictions (probabilities or values).
        targets : npt.NDArray[np.float64]
            Ground truth labels/values (used for classification detection).
        validation_returns : npt.NDArray[np.float64]
            Actual market returns for computing strategy performance.

        Returns
        -------
        float
            Annualized Sharpe ratio. Returns 0.0 if:
            - No valid samples
            - Zero standard deviation (constant returns)
            - All non-finite values

        Example
        -------
        >>> sharpe = hp_component._calculate_sharpe_metric(
        ...     predictions, targets, returns
        ... )
        >>> print(f"Sharpe Ratio: {sharpe:.2f}")

        """
        signals = (
            self._probabilities_to_labels(predictions)
            if self._is_classification_problem(targets)
            else np.sign(predictions)
        )
        returns = np.asarray(validation_returns, dtype=np.float64).reshape(-1)
        n = min(len(signals), len(returns))
        if n == 0:
            return 0.0
        strategy_returns = returns[:n] * signals[:n]
        strategy_returns = strategy_returns[np.isfinite(strategy_returns)]
        if strategy_returns.size == 0:
            return 0.0
        std = np.std(strategy_returns)
        if std == 0.0:
            return 0.0
        sharpe = np.sqrt(252.0) * np.mean(strategy_returns) / std
        return float(sharpe)

    def _build_optuna_sampler(
        self,
        cfg: OptunaConfig,
    ) -> optuna.samplers.BaseSampler:
        """
        Build Optuna sampler from configuration.

        Parameters
        ----------
        cfg : OptunaConfig
            Configuration with sampler setting.

        Returns
        -------
        optuna.samplers.BaseSampler
            The configured sampler instance.

        Notes
        -----
        Supported samplers:
        - 'tpe': Tree-structured Parzen Estimator (default)
        - 'random': Random sampling
        - 'cmaes': Covariance Matrix Adaptation Evolution Strategy
        - 'grid': Falls back to TPE with warning (requires explicit search space)

        """
        sampler_name = cfg.sampler.lower()
        if sampler_name == "random":
            return optuna.samplers.RandomSampler()
        if sampler_name == "cmaes":
            return optuna.samplers.CmaEsSampler()
        if sampler_name == "grid":
            self._trainer._log_warning(
                "Grid sampler requires explicit search space; using TPE instead"
            )
            return optuna.samplers.TPESampler(seed=42)
        return optuna.samplers.TPESampler(seed=42)

    def _build_optuna_pruner(
        self,
        cfg: OptunaConfig,
    ) -> optuna.pruners.BasePruner | None:
        """
        Build Optuna pruner from configuration.

        Parameters
        ----------
        cfg : OptunaConfig
            Configuration with pruner setting.

        Returns
        -------
        optuna.pruners.BasePruner | None
            The configured pruner instance, or None if disabled.

        Notes
        -----
        Supported pruners:
        - 'none': No pruning (returns None)
        - 'median': Median pruner (default)
        - 'hyperband': Hyperband pruner
        - 'percentile': Percentile pruner (25th percentile)

        """
        pruner_name = cfg.pruner.lower()
        if pruner_name == "none":
            return None
        if pruner_name == "hyperband":
            return optuna.pruners.HyperbandPruner()
        if pruner_name == "percentile":
            return optuna.pruners.PercentilePruner(25.0)
        return optuna.pruners.MedianPruner()

    def _train_with_params(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        X_val: npt.NDArray[np.float64],
        y_val: npt.NDArray[np.float64],
        params: dict[str, Any],
    ) -> Any:
        """
        Train model with specific parameters (for Optuna objective).

        Delegates to trainer's _train_model and extracts the model from results.

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
        params : dict[str, Any]
            Model parameters to use for training.

        Returns
        -------
        Any
            The trained model instance.

        """
        results = self._trainer._train_model(X_train, y_train, X_val, y_val, **params)
        return results["model"]

    def _is_classification_problem(
        self,
        y: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    ) -> bool:
        """
        Determine if this is a classification problem based on target values.

        Parameters
        ----------
        y : npt.NDArray[np.float32] | npt.NDArray[np.float64]
            Target array to analyze.

        Returns
        -------
        bool
            True if targets suggest classification (few unique integer values),
            False if regression (many unique or continuous values).

        Notes
        -----
        Classification is assumed if:
        - All values are integers and <= 10 unique values
        - All values in [0, 1] range and <= 10 unique values

        """
        unique_values = np.unique(y)

        # If all values are integers and there are few unique values, assume classification
        if np.all(y == y.astype(int)) and len(unique_values) <= 10:
            return True

        # Check if values are in [0, 1] range (common for binary classification)
        if np.all((y >= 0) & (y <= 1)) and len(unique_values) <= 10:
            return True

        return False


__all__ = [
    "HyperparameterComponent",
    "HyperparameterTrainerProtocol",
]
