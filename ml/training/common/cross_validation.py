"""
Cross-validation component for BaseMLTrainer decomposition.

This module provides the CrossValidationComponent which encapsulates cross-validation
logic from BaseMLTrainer (lines 504-512 and 818-1069), including:
- CV check (_should_use_cv)
- CV orchestration (_cross_validate)
- Time series CV with expanding window (_time_series_cv)
- Deprecated standard CV (_standard_cv)
- Purged walk-forward CV (_purged_cv)

Following Universal ML Architecture Pattern 2: Protocol-First Interface Design.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt


if TYPE_CHECKING:
    from ml.config.base import MLTrainingConfig


logger = logging.getLogger(__name__)


class CVTrainerProtocol(Protocol):
    """
    Protocol for trainer interaction with cross-validation component.

    Defines the interface that any trainer must implement to work with
    the CrossValidationComponent. This follows Protocol-First design
    (Pattern 2) to enable structural typing without inheritance coupling.

    Attributes
    ----------
    _config : MLTrainingConfig
        Training configuration with CV settings.

    """

    _config: MLTrainingConfig

    def _log_info(self, message: str, *args: object, **kwargs: Any) -> None:
        """Log info message."""
        ...

    def _log_warning(self, message: str, *args: object, **kwargs: Any) -> None:
        """Log warning message."""
        ...

    def _create_model(self, params: dict[str, Any]) -> Any:
        """Create a new model instance with given parameters."""
        ...

    def _get_model_params(self) -> dict[str, Any]:
        """Get default model parameters."""
        ...

    def _train_with_params(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        X_val: npt.NDArray[np.float64],
        y_val: npt.NDArray[np.float64],
        params: dict[str, Any],
    ) -> Any:
        """Train model with specific parameters."""
        ...

    def evaluate(
        self,
        model: Any,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> dict[str, float]:
        """Evaluate model performance."""
        ...


class CrossValidationComponent:
    """
    Component responsible for cross-validation operations.

    This component encapsulates the cross-validation logic from BaseMLTrainer
    (lines 504-512 and 818-1069), implementing time-series safe CV strategies:
    - Time series CV (expanding window)
    - Purged walk-forward CV (with embargo, using PurgedCrossValidator)
    - Standard CV (deprecated, forwards to time_series)

    The component delegates model creation and evaluation to the trainer instance
    through the CVTrainerProtocol interface, following Protocol-First design.

    Parameters
    ----------
    trainer : CVTrainerProtocol
        The trainer instance that implements the CVTrainerProtocol.

    Example
    -------
    >>> from ml.training.common import CrossValidationComponent
    >>> # trainer is an instance implementing CVTrainerProtocol
    >>> cv_component = CrossValidationComponent(trainer)
    >>> if cv_component._should_use_cv():
    ...     results = cv_component._cross_validate(X, y)
    ...     print(f"CV completed with {len(results)} folds")

    """

    def __init__(self, trainer: CVTrainerProtocol) -> None:
        """
        Initialize the cross-validation component with a trainer reference.

        Parameters
        ----------
        trainer : CVTrainerProtocol
            The trainer instance for delegation.

        """
        self._trainer = trainer

    def _should_use_cv(self) -> bool:
        """
        Check if cross-validation should be used.

        Cross-validation is enabled when the config has cv_folds > 1.

        Returns
        -------
        bool
            True if cross-validation should be performed, False otherwise.

        Example
        -------
        >>> cv_component = CrossValidationComponent(trainer)
        >>> if cv_component._should_use_cv():
        ...     results = cv_component._cross_validate(X, y)

        """
        return (
            hasattr(self._trainer._config, "cv_folds")
            and self._trainer._config.cv_folds is not None
            and self._trainer._config.cv_folds > 1
        )

    def _cross_validate(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        """
        Perform cross-validation.

        Routes to the appropriate CV strategy based on config:
        - "time_series": Expanding window time-series CV
        - "purged": Purged walk-forward CV with embargo
        - "standard"/"blocked": Deprecated, forwards to time_series

        Parameters
        ----------
        X : npt.NDArray[np.float64]
            Features array of shape (n_samples, n_features).
        y : npt.NDArray[np.float64]
            Target array of shape (n_samples,).
        **kwargs : Any
            Additional parameters passed to CV methods.

        Returns
        -------
        list[dict[str, float]]
            Cross-validation results for each fold. Each dict contains
            evaluation metrics for that fold.

        Example
        -------
        >>> results = cv_component._cross_validate(X, y)
        >>> for i, fold_metrics in enumerate(results):
        ...     print(f"Fold {i}: {fold_metrics}")

        """
        n_folds: int = getattr(self._trainer._config, "cv_folds", 5)
        cv_strategy = getattr(self._trainer._config, "cv_strategy", "time_series")
        n_samples: int = len(X)

        # Guard against too few samples for requested folds
        if n_folds > n_samples:
            self._trainer._log_warning(
                f"cv_folds ({n_folds}) > samples ({n_samples}); reducing folds to {n_samples}",
            )
            n_folds = n_samples
        if n_folds < 2 or n_samples < 2:
            self._trainer._log_warning("Insufficient samples for cross-validation; skipping CV")
            return []

        self._trainer._log_info(f"Starting {n_folds}-fold {cv_strategy} cross-validation")

        # Time-series safe strategies only
        if cv_strategy in ("time_series", "standard", "blocked"):
            # Map any non-purged strategy to time_series for safety
            if cv_strategy == "standard":
                self._trainer._log_warning(
                    "cv_strategy 'standard' is deprecated; using time_series CV",
                )
            if cv_strategy == "blocked":
                self._trainer._log_warning(
                    "cv_strategy 'blocked' not implemented; using time_series CV",
                )
            return self._time_series_cv(X, y, n_folds, **kwargs)
        if cv_strategy == "purged":
            return self._purged_cv(X, y, n_folds, **kwargs)

        # Unknown strategies default to time-series safe behavior
        self._trainer._log_warning(
            f"Unknown cv_strategy '{cv_strategy}'; using time_series CV",
        )
        return self._time_series_cv(X, y, n_folds, **kwargs)

    def _time_series_cv(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        n_folds: int,
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        """
        Time series cross-validation with expanding window.

        Uses an expanding window approach where each fold uses all
        previous data for training and a fixed-size window for validation.
        This preserves temporal ordering and prevents data leakage.

        Parameters
        ----------
        X : npt.NDArray[np.float64]
            Features array of shape (n_samples, n_features).
        y : npt.NDArray[np.float64]
            Target array of shape (n_samples,).
        n_folds : int
            Number of cross-validation folds.
        **kwargs : Any
            Additional parameters passed to model training.

        Returns
        -------
        list[dict[str, float]]
            CV results for each fold.

        Example
        -------
        >>> results = cv_component._time_series_cv(X, y, n_folds=5)
        >>> assert len(results) == 5

        """
        n_samples: int = len(X)
        fold_size: int = int(n_samples // (n_folds + 1))
        results: list[dict[str, float]] = []

        if fold_size < 1:
            self._trainer._log_warning(
                f"Time-series CV fold_size={fold_size} too small for {n_folds} folds; skipping CV",
            )
            return results

        for i in range(n_folds):
            train_end = (i + 1) * fold_size
            val_start = train_end
            val_end = min(val_start + fold_size, n_samples)

            X_train_cv = X[:train_end]
            y_train_cv = y[:train_end]
            X_val_cv = X[val_start:val_end]
            y_val_cv = y[val_start:val_end]

            # Train model for this fold
            model = self._trainer._create_model(self._trainer._get_model_params())
            if hasattr(model, "fit"):
                model.fit(X_train_cv, y_train_cv, eval_set=[(X_val_cv, y_val_cv)], verbose=False)
            else:
                model = self._trainer._train_with_params(
                    X_train_cv,
                    y_train_cv,
                    X_val_cv,
                    y_val_cv,
                    kwargs,
                )

            # Evaluate
            fold_metrics = self._trainer.evaluate(model, X_val_cv, y_val_cv)
            results.append(fold_metrics)

        # Calculate average metrics
        if results:
            avg_metrics: dict[str, float] = {}
            for key in results[0].keys():
                avg_metrics[f"cv_{key}_mean"] = float(np.mean([r[key] for r in results]))
                avg_metrics[f"cv_{key}_std"] = float(np.std([r[key] for r in results]))

            self._trainer._log_info(f"CV results: {avg_metrics}")

        return results

    def _standard_cv(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        n_folds: int,
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        """
        Deprecated: Use time-series aware CV.

        This method forwards to time-series CV to avoid unsafe shuffling
        on temporal data. Standard k-fold CV with shuffling can cause
        data leakage in financial time series.

        Parameters
        ----------
        X : npt.NDArray[np.float64]
            Features array of shape (n_samples, n_features).
        y : npt.NDArray[np.float64]
            Target array of shape (n_samples,).
        n_folds : int
            Number of folds.
        **kwargs : Any
            Additional parameters.

        Returns
        -------
        list[dict[str, float]]
            CV results (from time_series_cv).

        .. deprecated::
            Standard CV is deprecated for financial data. Use time_series
            or purged CV strategy instead.

        """
        self._trainer._log_warning(
            "cv_strategy 'standard' is deprecated; using time_series CV for safety",
        )
        return self._time_series_cv(X, y, n_folds, **kwargs)

    def _purged_cv(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        n_folds: int,
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        """
        Purged/embargoed walk-forward cross-validation (time-series safe).

        Uses ml.preprocessing.stationarity.PurgedCrossValidator to generate
        train/test indices which respect a purge gap and optional embargo
        window to prevent temporal leakage.

        Parameters
        ----------
        X : npt.NDArray[np.float64]
            Features array of shape (n_samples, n_features).
        y : npt.NDArray[np.float64]
            Target array of shape (n_samples,).
        n_folds : int
            Number of folds.
        **kwargs : Any
            Additional parameters passed to model training.

        Returns
        -------
        list[dict[str, float]]
            CV results for each fold.

        Notes
        -----
        If PurgedCrossValidator is unavailable (import error), falls back
        to time_series CV with a warning logged.

        Example
        -------
        >>> # With purge_gap=10 and embargo_pct=0.05 in config
        >>> results = cv_component._purged_cv(X, y, n_folds=5)
        >>> # Each fold has a gap to prevent information leakage

        """
        try:
            from ml.preprocessing.stationarity import PurgedCrossValidator
        except Exception as exc:  # pragma: no cover - defensive
            self._trainer._log_warning(
                f"PurgedCrossValidator unavailable ({exc}); falling back to time_series CV",
            )
            return self._time_series_cv(X, y, n_folds, **kwargs)

        n_samples: int = len(X)
        if n_folds > n_samples:
            n_folds = n_samples
        if n_folds < 2 or n_samples < 2:
            self._trainer._log_warning("Insufficient samples for purged CV; skipping CV")
            return []

        purge_gap: int = int(getattr(self._trainer._config, "purge_gap", 0) or 0)
        embargo_pct: float = float(getattr(self._trainer._config, "embargo_pct", 0.0) or 0.0)

        cv = PurgedCrossValidator(
            n_splits=int(n_folds),
            purge_gap=purge_gap,
            embargo_pct=embargo_pct,
        )

        results: list[dict[str, float]] = []
        for train_idx, val_idx in cv.split(X, y):
            X_train_cv = X[train_idx]
            y_train_cv = y[train_idx]
            X_val_cv = X[val_idx]
            y_val_cv = y[val_idx]

            model = self._trainer._create_model(self._trainer._get_model_params())
            if hasattr(model, "fit"):
                model.fit(
                    X_train_cv,
                    y_train_cv,
                    eval_set=[(X_val_cv, y_val_cv)],
                    verbose=False,
                )
            else:
                model = self._trainer._train_with_params(
                    X_train_cv,
                    y_train_cv,
                    X_val_cv,
                    y_val_cv,
                    kwargs,
                )

            fold_metrics = self._trainer.evaluate(model, X_val_cv, y_val_cv)
            results.append(fold_metrics)

        # Optional: log aggregate
        if results:
            avg_metrics: dict[str, float] = {}
            for key in results[0].keys():
                avg_metrics[f"cv_{key}_mean"] = float(np.mean([r[key] for r in results]))
                avg_metrics[f"cv_{key}_std"] = float(np.std([r[key] for r in results]))
            self._trainer._log_info(f"Purged CV results: {avg_metrics}")

        return results


__all__ = [
    "CVTrainerProtocol",
    "CrossValidationComponent",
]
