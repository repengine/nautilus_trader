"""
Configuration for XGBoost model training.

This module provides the comprehensive msgspec-based configuration for XGBoost training,
including all advanced features like GPU acceleration, Optuna optimization, MLflow
tracking, and feature monitoring.

"""

from __future__ import annotations

from typing import Any

from ml.config.base import MLTrainingConfig
from ml.config.shared import AdvancedTrainingConfig
from ml.config.shared import MLflowConfig
from ml.config.shared import OptunaConfig
from ml.config.shared import XGBoostGPUConfig
from nautilus_trader.common.config import NonNegativeFloat
from nautilus_trader.common.config import NonNegativeInt
from nautilus_trader.common.config import PositiveFloat
from nautilus_trader.common.config import PositiveInt


class XGBoostTrainingConfig(MLTrainingConfig, kw_only=True, frozen=True):
    """
    Comprehensive configuration for XGBoost model training.

    This configuration extends the base MLTrainingConfig with XGBoost-specific
    parameters for tree-based gradient boosting, including advanced features for
    GPU acceleration, hyperparameter optimization, and experiment tracking.

    Parameters
    ----------
    n_estimators : PositiveInt, default 100
        Number of gradient boosted trees. Equivalent to number of boosting rounds.
    max_depth : PositiveInt, default 6
        Maximum depth of a tree. Increasing this value makes the model more complex
        and more likely to overfit.
    learning_rate : PositiveFloat, default 0.3
        Boosting learning rate (xgb's "eta"). Step size shrinkage used in update
        to prevent overfitting.
    min_child_weight : NonNegativeFloat, default 1.0
        Minimum sum of instance weight (hessian) needed in a child.
    subsample : PositiveFloat, default 1.0
        Subsample ratio of the training instances. Setting it to 0.5 means that
        XGBoost would randomly sample half of the training data prior to growing trees.
    colsample_bytree : PositiveFloat, default 1.0
        Subsample ratio of columns when constructing each tree.
    colsample_bylevel : PositiveFloat, default 1.0
        Subsample ratio of columns for each level.
    gamma : NonNegativeFloat, default 0.0
        Minimum loss reduction required to make a further partition on a leaf node.
    reg_alpha : NonNegativeFloat, default 0.0
        L1 regularization term on weights.
    reg_lambda : NonNegativeFloat, default 1.0
        L2 regularization term on weights.
    tree_method : str, default "hist"
        Tree construction algorithm. Options: "hist", "gpu_hist", "exact", "approx".
    gpu_id : NonNegativeInt, default 0
        GPU device ID (only relevant when tree_method="gpu_hist").
    objective : str, default "binary:logistic"
        Learning objective. Options: "binary:logistic", "reg:squarederror", "multi:softprob".
    eval_metric : str, default "auc"
        Evaluation metric for validation data.
    enable_shap : bool, default False
        Whether to compute SHAP values for feature importance analysis.
    monotonic_constraints : dict[str, int] | None, optional
        Monotonic constraints for features. Keys are feature names, values are
        -1 (decreasing), 0 (no constraint), or 1 (increasing).
    multi_asset : bool, default False
        Whether to train on multiple assets with cross-sectional features.
    sector_map : dict[str, str] | None, optional
        Mapping from asset symbols to sector names for multi-asset training.
    cross_sectional_features : bool, default True
        Whether to include cross-sectional ranking features for multi-asset models.
    optimize_hyperparams : bool, default False
        Whether to optimize hyperparameters using Optuna.
    n_trials : PositiveInt, default 100
        Number of optimization trials when optimize_hyperparams is True.
    optimization_metric : str, default "sharpe_ratio"
        Metric to optimize during hyperparameter tuning.
    gpu_config : XGBoostGPUConfig, optional
        GPU acceleration settings.
    optuna_config : OptunaConfig, optional
        Optuna hyperparameter optimization settings.
    mlflow_config : MLflowConfig, optional
        MLflow experiment tracking settings.
    advanced_config : AdvancedTrainingConfig, optional
        Advanced training features like cross-validation and ONNX export.

    """

    # Core XGBoost parameters
    n_estimators: PositiveInt = 100
    max_depth: PositiveInt = 6
    learning_rate: PositiveFloat = 0.3
    min_child_weight: NonNegativeFloat = 1.0
    subsample: PositiveFloat = 1.0
    colsample_bytree: PositiveFloat = 1.0
    colsample_bylevel: PositiveFloat = 1.0
    gamma: NonNegativeFloat = 0.0
    reg_alpha: NonNegativeFloat = 0.0
    reg_lambda: NonNegativeFloat = 1.0

    # Missing value handling
    handle_missing: bool = True
    missing_value: float = float("nan")
    scale_pos_weight: PositiveFloat | None = None

    # Hardware settings
    tree_method: str = "hist"  # "hist" for CPU, "gpu_hist" for GPU
    gpu_id: NonNegativeInt = 0

    # Training objective
    objective: str = "binary:logistic"
    eval_metric: str = "auc"

    # Advanced features
    enable_shap: bool = False
    monotonic_constraints: dict[str, int] | None = None

    # Multi-asset configuration
    multi_asset: bool = False
    sector_map: dict[str, str] | None = None
    cross_sectional_features: bool = True

    # Legacy hyperparameter optimization (kept for backward compatibility)
    optimize_hyperparams: bool = False
    n_trials: PositiveInt = 100
    optimization_metric: str = "sharpe_ratio"

    # Advanced configuration components
    gpu_config: XGBoostGPUConfig | None = None
    optuna_config: OptunaConfig | None = None
    mlflow_config: MLflowConfig | None = None
    advanced_config: AdvancedTrainingConfig | None = None

    # Convenience properties for backward compatibility
    @property
    def track_feature_decay(self) -> bool:
        """
        Whether to track feature importance decay.
        """
        return self.advanced_config.track_feature_decay if self.advanced_config else False

    @property
    def feature_decay_threshold(self) -> float:
        """
        Threshold for feature importance decay alerts.
        """
        return self.advanced_config.feature_decay_threshold if self.advanced_config else 0.3

    @property
    def feature_history_window(self) -> int:
        """
        Number of training runs to keep in feature importance history.
        """
        return self.advanced_config.feature_history_window if self.advanced_config else 10

    @property
    def cv_strategy(self) -> str:
        """
        Cross-validation strategy.
        """
        return self.advanced_config.cv_strategy if self.advanced_config else "time_series"

    @property
    def cv_folds(self) -> int:
        """
        Number of cross-validation folds.
        """
        return self.advanced_config.cv_folds if self.advanced_config else 5

    @property
    def purge_gap(self) -> int:
        """
        Gap between train/test in purged cross-validation.
        """
        return self.advanced_config.purge_gap if self.advanced_config else 10

    @property
    def export_onnx(self) -> bool:
        """
        Whether to export model to ONNX format.
        """
        return self.advanced_config.export_onnx if self.advanced_config else False

    @property
    def onnx_output_path(self) -> str:
        """
        Path for ONNX model export.
        """
        if self.advanced_config and self.advanced_config.onnx_output_path:
            return self.advanced_config.onnx_output_path
        return "./models/xgboost.onnx"

    @property
    def enable_monitoring(self) -> bool:
        """
        Whether to enable Prometheus metrics collection.
        """
        return self.advanced_config.enable_monitoring if self.advanced_config else True

    def __post_init__(self) -> None:
        """
        Post-initialization validation.
        """
        # Validate subsample and colsample ratios
        if not (0.0 < self.subsample <= 1.0):
            msg = f"subsample must be in (0.0, 1.0], got {self.subsample}"
            raise ValueError(msg)

        if not (0.0 < self.colsample_bytree <= 1.0):
            msg = f"colsample_bytree must be in (0.0, 1.0], got {self.colsample_bytree}"
            raise ValueError(msg)

        if not (0.0 < self.colsample_bylevel <= 1.0):
            msg = f"colsample_bylevel must be in (0.0, 1.0], got {self.colsample_bylevel}"
            raise ValueError(msg)

        # Validate tree method
        valid_tree_methods = ["hist", "gpu_hist", "exact", "approx"]
        if self.tree_method not in valid_tree_methods:
            msg = f"tree_method must be one of {valid_tree_methods}, got {self.tree_method}"
            raise ValueError(msg)

        # Validate objective
        valid_objectives = ["binary:logistic", "reg:squarederror", "multi:softprob", "reg:logistic"]
        if self.objective not in valid_objectives:
            msg = f"objective must be one of {valid_objectives}, got {self.objective}"
            raise ValueError(msg)

        # Validate multi-asset settings
        if self.multi_asset and self.sector_map is None:
            msg = "sector_map is required when multi_asset=True"
            raise ValueError(msg)

        # Validate monotonic constraints
        if self.monotonic_constraints is not None:
            for feature, constraint in self.monotonic_constraints.items():
                if constraint not in [-1, 0, 1]:
                    msg = (
                        f"monotonic constraint for {feature} must be -1, 0, or 1, got {constraint}"
                    )
                    raise ValueError(msg)

        # Handle legacy optimize_hyperparams flag
        if self.optimize_hyperparams and not self.optuna_config:
            # Create default Optuna config from legacy settings
            object.__setattr__(
                self,
                "optuna_config",
                OptunaConfig(
                    enabled=True,
                    n_trials=self.n_trials,
                    metric=self.optimization_metric,
                ),
            )

        # Handle GPU settings
        if self.tree_method == "gpu_hist" and not self.gpu_config:
            # Create default GPU config
            object.__setattr__(
                self,
                "gpu_config",
                XGBoostGPUConfig(
                    enabled=True,
                    device_id=self.gpu_id,
                ),
            )

    def get_xgb_params(self) -> dict[str, Any]:
        """
        Get XGBoost parameters as a dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary of XGBoost parameters suitable for model initialization.

        """
        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "colsample_bylevel": self.colsample_bylevel,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "tree_method": self.tree_method,
            "random_state": self.random_seed,
            "n_jobs": -1,
            "verbosity": 0,
        }

        # Add GPU parameters if using GPU
        if self.gpu_config and self.gpu_config.enabled:
            params["tree_method"] = "gpu_hist"
            params["gpu_id"] = self.gpu_config.device_id
            params["predictor"] = self.gpu_config.predictor
            params["max_bin"] = self.gpu_config.max_bin
        elif self.tree_method == "gpu_hist":
            # Legacy GPU settings
            params["gpu_id"] = self.gpu_id
            params["predictor"] = "gpu_predictor"

        return params

    def validate_environment(self) -> list[str]:
        """
        Validate that the environment supports the configured features.

        Returns
        -------
        list[str]
            List of warning messages for unsupported features.

        """
        warnings = []

        # Check GPU availability
        if self.gpu_config and self.gpu_config.enabled and self.gpu_config.validate_gpu:
            try:
                # Try to create a simple GPU-based DMatrix to test GPU availability
                import numpy as np

                from ml._imports import xgb as _xgb

                if _xgb is None:
                    raise ImportError("xgboost not installed")
                rng = np.random.default_rng(42)
                test_data = rng.standard_normal((10, 5))
                test_dtrain = _xgb.DMatrix(test_data, enable_categorical=False)
                # If this doesn't raise, GPU should be available
                del test_dtrain, test_data
            except Exception as e:
                warnings.append(f"GPU acceleration requested but not available: {e}")

        # Check Optuna availability
        if self.optuna_config and self.optuna_config.enabled:
            from ml._imports import HAS_OPTUNA

            if not HAS_OPTUNA:
                warnings.append(
                    "Optuna optimization requested but optuna not installed. Install with: pip install 'nautilus-trader[ml]'",
                )

        # Check MLflow availability
        if self.mlflow_config and self.mlflow_config.enabled:
            from ml._imports import HAS_MLFLOW

            if not HAS_MLFLOW:
                warnings.append(
                    "MLflow tracking requested but mlflow not installed. Install with: pip install 'nautilus-trader[ml]'",
                )

        # Check ONNX availability
        if self.export_onnx:
            from ml._imports import HAS_ONNX_EXPORT

            if not HAS_ONNX_EXPORT:
                warnings.append(
                    "ONNX export requested but onnx tools not installed. Install with: pip install onnxmltools skl2onnx",
                )

        return warnings


# Backward compatibility alias
UnifiedXGBoostConfig = XGBoostTrainingConfig


# Explicit exports
__all__ = [
    "UnifiedXGBoostConfig",  # Backward compatibility
    "XGBoostTrainingConfig",
]
