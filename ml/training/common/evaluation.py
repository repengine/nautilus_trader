"""
Evaluation component for BaseMLTrainer decomposition.

This module provides the EvaluationComponent which encapsulates evaluation
logic from BaseMLTrainer (lines 1163-1263 and 1519-1607), including:
- Model evaluation (evaluate)
- Trading metrics calculation (calculate_trading_metrics)
- Problem type detection (_is_classification_problem, _is_classifier_objective)
- Classification metrics (_calculate_classification_metrics)
- Regression metrics (_calculate_regression_metrics)

Following Universal ML Architecture Pattern 2: Protocol-First Interface Design.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt

from ml._imports import HAS_SKLEARN


if TYPE_CHECKING:
    from ml.config.base import MLTrainingConfig


logger = logging.getLogger(__name__)


class EvaluationTrainerProtocol(Protocol):
    """
    Protocol for trainer interaction with evaluation component.

    Defines the interface that any trainer must implement to work with
    the EvaluationComponent. This follows Protocol-First design
    (Pattern 2) to enable structural typing without inheritance coupling.

    Attributes
    ----------
    _config : MLTrainingConfig
        Training configuration with objective settings.

    """

    _config: MLTrainingConfig

    def predict(
        self,
        model: Any,
        X: npt.NDArray[np.float64],
        *,
        return_labels: bool = False,
    ) -> npt.NDArray[np.float32] | npt.NDArray[np.float64]:
        """
        Generate predictions from the model.

        Parameters
        ----------
        model : Any
            The trained model.
        X : npt.NDArray[np.float64]
            Feature array for prediction.
        return_labels : bool, optional
            If True, return class labels instead of probabilities.
            Default is False.

        Returns
        -------
        npt.NDArray[np.float32] | npt.NDArray[np.float64]
            Model predictions.

        """
        ...


class EvaluationComponent:
    """
    Component responsible for model evaluation operations.

    This component encapsulates the evaluation logic from BaseMLTrainer
    (lines 1163-1263 and 1519-1607), implementing:
    - Model evaluation with classification/regression metric selection
    - Trading-specific metrics (Sharpe ratio, max drawdown, etc.)
    - Problem type detection for automatic metric selection

    The component delegates prediction to the trainer instance through the
    EvaluationTrainerProtocol interface, following Protocol-First design.

    Parameters
    ----------
    trainer : EvaluationTrainerProtocol
        The trainer instance that implements the EvaluationTrainerProtocol.

    Example
    -------
    >>> from ml.training.common import EvaluationComponent
    >>> # trainer is an instance implementing EvaluationTrainerProtocol
    >>> eval_component = EvaluationComponent(trainer)
    >>> metrics = eval_component.evaluate(model, X_test, y_test)
    >>> print(f"Model accuracy: {metrics.get('accuracy', 'N/A')}")

    """

    def __init__(self, trainer: EvaluationTrainerProtocol) -> None:
        """
        Initialize the evaluation component with a trainer reference.

        Parameters
        ----------
        trainer : EvaluationTrainerProtocol
            The trainer instance for delegation.

        """
        self._trainer = trainer

    def evaluate(
        self,
        model: Any,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> dict[str, float]:
        """
        Evaluate model performance using standard ML metrics.

        Automatically detects whether the problem is classification or
        regression based on target values and selects appropriate metrics.

        Parameters
        ----------
        model : Any
            The trained model to evaluate.
        X : npt.NDArray[np.float64]
            Feature array for evaluation.
        y : npt.NDArray[np.float64]
            Target array for evaluation.

        Returns
        -------
        dict[str, float]
            Dictionary of evaluation metrics. For classification:
            accuracy, precision, recall, f1_score. For regression:
            mse, rmse, mae, r2_score.

        Example
        -------
        >>> metrics = eval_component.evaluate(model, X_test, y_test)
        >>> # For classification
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
        >>> # For regression
        >>> print(f"RMSE: {metrics['rmse']:.4f}")

        """
        # For classification metrics, we need labels not probabilities
        if self._is_classification_problem(y):
            predictions = self._trainer.predict(model, X, return_labels=True)
            return self._calculate_classification_metrics(y, predictions)
        else:
            # For regression, we get raw values
            predictions = self._trainer.predict(model, X)
            return self._calculate_regression_metrics(y, predictions)

    def calculate_trading_metrics(
        self,
        returns: npt.NDArray[np.float64],
        predictions: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    ) -> dict[str, float]:
        """
        Calculate trading-specific performance metrics.

        Converts model predictions to trading signals and computes
        strategy performance metrics.

        Parameters
        ----------
        returns : npt.NDArray[np.float64]
            Asset returns for the prediction period.
        predictions : npt.NDArray[np.float32] | npt.NDArray[np.float64]
            Model predictions (signals).

        Returns
        -------
        dict[str, float]
            Dictionary of trading metrics including:
            - total_return: Cumulative strategy return
            - sharpe_ratio: Annualized Sharpe ratio (assuming daily returns)
            - max_drawdown: Maximum peak-to-trough drawdown
            - win_rate: Proportion of positive strategy returns
            - information_ratio: Risk-adjusted excess return vs benchmark

        Example
        -------
        >>> trading_metrics = eval_component.calculate_trading_metrics(
        ...     returns=asset_returns,
        ...     predictions=model_predictions,
        ... )
        >>> print(f"Sharpe Ratio: {trading_metrics.get('sharpe_ratio', 0):.2f}")
        >>> print(f"Max Drawdown: {trading_metrics.get('max_drawdown', 0):.2%}")

        """
        # Convert predictions to trading signals
        if self._is_classifier_objective():
            signals = np.where(predictions >= 0.5, 1.0, -1.0)
        else:
            signals = np.sign(predictions)

        # Calculate strategy returns
        strategy_returns = returns * signals

        # Remove any NaN or infinite values
        strategy_returns = strategy_returns[np.isfinite(strategy_returns)]
        if len(strategy_returns) == 0:
            return {}

        # Calculate metrics
        metrics: dict[str, float] = {}

        # Total return
        total_return = float(np.prod(1 + strategy_returns) - 1)
        metrics["total_return"] = total_return

        # Sharpe ratio (annualized, assuming daily returns)
        std_returns = float(np.std(strategy_returns))
        if std_returns > 0:
            sharpe_ratio = float(
                np.sqrt(252) * np.mean(strategy_returns) / std_returns
            )
            metrics["sharpe_ratio"] = sharpe_ratio

        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = float(np.min(drawdown))
        metrics["max_drawdown"] = abs(max_drawdown)

        # Win rate
        winning_trades = strategy_returns > 0
        if len(winning_trades) > 0:
            win_rate = float(np.mean(winning_trades))
            metrics["win_rate"] = win_rate

        # Information ratio (excess return / tracking error)
        benchmark_return = float(np.mean(returns))
        excess_returns = strategy_returns - benchmark_return
        std_excess = float(np.std(excess_returns))
        if std_excess > 0:
            information_ratio = float(np.mean(excess_returns) / std_excess)
            metrics["information_ratio"] = information_ratio

        return metrics

    def _is_classification_problem(
        self,
        y: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    ) -> bool:
        """
        Determine if this is a classification problem based on target values.

        Uses heuristics to detect classification:
        1. All values are integers and <= 10 unique values
        2. All values in [0, 1] range with <= 10 unique values

        Parameters
        ----------
        y : npt.NDArray[np.float32] | npt.NDArray[np.float64]
            Target array.

        Returns
        -------
        bool
            True if classification, False if regression.

        Example
        -------
        >>> # Binary classification targets
        >>> y_binary = np.array([0, 1, 1, 0, 1])
        >>> eval_component._is_classification_problem(y_binary)
        True
        >>> # Continuous regression targets
        >>> y_regression = np.array([1.23, 4.56, 7.89])
        >>> eval_component._is_classification_problem(y_regression)
        False

        """
        unique_values = np.unique(y)

        # If all values are integers and there are few unique values, assume classification
        if np.all(y == y.astype(int)) and len(unique_values) <= 10:
            return True

        # Check if values are in [0, 1] range (common for binary classification)
        if np.all((y >= 0) & (y <= 1)) and len(unique_values) <= 10:
            return True

        return False

    def _is_classifier_objective(self) -> bool:
        """
        Check if the config objective indicates a classification task.

        Examines the objective string for classification-related keywords:
        'binary', 'class', 'logit'.

        Returns
        -------
        bool
            True if the objective indicates classification.

        Example
        -------
        >>> # With config.objective = "binary:logistic"
        >>> eval_component._is_classifier_objective()
        True

        """
        objective = str(getattr(self._trainer._config, "objective", "")).lower()
        if not objective:
            return False
        return any(token in objective for token in ("binary", "class", "logit"))

    def _calculate_classification_metrics(
        self,
        y_true: npt.NDArray[np.float64],
        y_pred: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    ) -> dict[str, float]:
        """
        Calculate classification metrics.

        Uses scikit-learn metrics if available, otherwise falls back to
        simple accuracy calculation.

        Parameters
        ----------
        y_true : npt.NDArray[np.float64]
            True labels.
        y_pred : npt.NDArray[np.float32] | npt.NDArray[np.float64]
            Predicted labels.

        Returns
        -------
        dict[str, float]
            Classification metrics including accuracy, precision,
            recall, and f1_score (if sklearn available), or just
            accuracy if sklearn is unavailable.

        Example
        -------
        >>> metrics = eval_component._calculate_classification_metrics(
        ...     y_true=np.array([0, 1, 1, 0]),
        ...     y_pred=np.array([0, 1, 0, 0]),
        ... )
        >>> print(f"Accuracy: {metrics['accuracy']:.2%}")

        """
        try:
            if not HAS_SKLEARN:
                raise ImportError("sklearn not available")
            from sklearn.metrics import accuracy_score
            from sklearn.metrics import f1_score
            from sklearn.metrics import precision_score
            from sklearn.metrics import recall_score

            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(
                    precision_score(y_true, y_pred, average="weighted", zero_division=0)
                ),
                "recall": float(
                    recall_score(y_true, y_pred, average="weighted", zero_division=0)
                ),
                "f1_score": float(
                    f1_score(y_true, y_pred, average="weighted", zero_division=0)
                ),
            }
        except ImportError:
            # Fallback to simple accuracy if sklearn not available
            if len(y_true) == 0 or len(y_pred) == 0:
                return {"accuracy": 0.0}
            return {
                "accuracy": float(np.mean(y_true == y_pred)),
            }

    def _calculate_regression_metrics(
        self,
        y_true: npt.NDArray[np.float64],
        y_pred: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    ) -> dict[str, float]:
        """
        Calculate regression metrics.

        Computes standard regression metrics using numpy.

        Parameters
        ----------
        y_true : npt.NDArray[np.float64]
            True target values.
        y_pred : npt.NDArray[np.float32] | npt.NDArray[np.float64]
            Predicted values.

        Returns
        -------
        dict[str, float]
            Regression metrics including:
            - mse: Mean Squared Error
            - rmse: Root Mean Squared Error
            - mae: Mean Absolute Error
            - r2_score: Coefficient of determination

        Example
        -------
        >>> metrics = eval_component._calculate_regression_metrics(
        ...     y_true=np.array([1.0, 2.0, 3.0]),
        ...     y_pred=np.array([1.1, 2.2, 2.9]),
        ... )
        >>> print(f"RMSE: {metrics['rmse']:.4f}")

        """
        mse = float(np.mean((y_true - y_pred) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y_true - y_pred)))

        # R-squared
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2,
        }


__all__ = [
    "EvaluationComponent",
    "EvaluationTrainerProtocol",
]
