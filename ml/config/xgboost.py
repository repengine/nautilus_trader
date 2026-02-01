"""
Configuration for XGBoost model training.

This module provides the comprehensive msgspec-based configuration for XGBoost training,
including all advanced features like GPU acceleration, Optuna optimization, MLflow
tracking, and feature monitoring.

"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ml.config._env_utils import ensure_env as _ensure_env
from ml.config._env_utils import env_non_negative_int as _env_non_negative_int
from ml.config._env_utils import env_positive_float as _env_positive_float
from ml.config._env_utils import env_positive_int as _env_positive_int
from ml.config._env_utils import env_truthy as _env_truthy
from ml.config.base import MLTrainingConfig
from ml.config.shared import AdvancedTrainingConfig
from ml.config.shared import OptunaConfig
from ml.config.shared import XGBoostGPUConfig
from ml.config.targets import TargetSemanticsConfig
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
    mlflow_config : deprecated, use ModelRegistry instead
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
    mlflow_config: None = None  # deprecated
    advanced_config: AdvancedTrainingConfig | None = None

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
    ) -> XGBoostTrainingConfig:
        """
        Build XGBoost training configuration from environment variables.
        """
        source = _ensure_env(env)

        data_source_value = source.get("ML_XGB_DATA_SOURCE") or source.get("ML_TRAIN_DATA_SOURCE")
        if not data_source_value:
            raise ValueError("ML_XGB_DATA_SOURCE or ML_TRAIN_DATA_SOURCE must be set")

        target_column = (
            source.get("ML_XGB_TARGET_COLUMN")
            or source.get("ML_TRAIN_TARGET_COLUMN")
            or "target"
        )

        target_semantics_raw = (
            source.get("ML_XGB_TARGET_SEMANTICS") or source.get("ML_TRAIN_TARGET_SEMANTICS")
        )
        if not target_semantics_raw:
            raise ValueError("ML_XGB_TARGET_SEMANTICS or ML_TRAIN_TARGET_SEMANTICS must be set")
        target_semantics = TargetSemanticsConfig.from_json(target_semantics_raw)

        kwargs: dict[str, Any] = {
            "data_source": data_source_value,
            "target_column": target_column,
            "target_semantics": target_semantics,
        }

        if "ML_TRAIN_TRAIN_TEST_SPLIT" in source or "ML_XGB_TRAIN_TEST_SPLIT" in source:
            split = _env_positive_float(
                source,
                "ML_XGB_TRAIN_TEST_SPLIT"
                if "ML_XGB_TRAIN_TEST_SPLIT" in source
                else "ML_TRAIN_TRAIN_TEST_SPLIT",
                0.8,
            )
            if not (0.0 < split < 1.0):
                split = 0.8
            kwargs["train_test_split"] = split

        if "ML_TRAIN_RANDOM_SEED" in source or "ML_XGB_RANDOM_SEED" in source:
            kwargs["random_seed"] = _env_non_negative_int(
                source,
                "ML_XGB_RANDOM_SEED"
                if "ML_XGB_RANDOM_SEED" in source
                else "ML_TRAIN_RANDOM_SEED",
                42,
            )

        if "ML_TRAIN_EARLY_STOPPING_ROUNDS" in source or "ML_XGB_EARLY_STOPPING_ROUNDS" in source:
            kwargs["early_stopping_rounds"] = _env_positive_int(
                source,
                "ML_XGB_EARLY_STOPPING_ROUNDS"
                if "ML_XGB_EARLY_STOPPING_ROUNDS" in source
                else "ML_TRAIN_EARLY_STOPPING_ROUNDS",
                50,
            )

        if "ML_TRAIN_VALIDATION_METRIC" in source or "ML_XGB_VALIDATION_METRIC" in source:
            kwargs["validation_metric"] = (
                source.get("ML_XGB_VALIDATION_METRIC")
                or source.get("ML_TRAIN_VALIDATION_METRIC")
                or "accuracy"
            ).strip()

        save_model = source.get("ML_XGB_SAVE_MODEL_PATH") or source.get("ML_TRAIN_SAVE_MODEL_PATH")
        if save_model:
            kwargs["save_model_path"] = save_model.strip()

        def _set_pos_int(env_key: str, attr: str, default: int) -> None:
            if env_key in source:
                kwargs[attr] = _env_positive_int(source, env_key, default)

        def _set_pos_float(env_key: str, attr: str, default: float) -> None:
            if env_key in source:
                kwargs[attr] = _env_positive_float(source, env_key, default)

        _set_pos_int("ML_XGB_N_ESTIMATORS", "n_estimators", cls.n_estimators)
        _set_pos_int("ML_XGB_MAX_DEPTH", "max_depth", cls.max_depth)
        _set_pos_float("ML_XGB_LEARNING_RATE", "learning_rate", cls.learning_rate)
        _set_pos_float("ML_XGB_MIN_CHILD_WEIGHT", "min_child_weight", cls.min_child_weight)
        _set_pos_float("ML_XGB_SUBSAMPLE", "subsample", cls.subsample)
        _set_pos_float("ML_XGB_COLSAMPLE_BYTREE", "colsample_bytree", cls.colsample_bytree)
        _set_pos_float("ML_XGB_COLSAMPLE_BYLEVEL", "colsample_bylevel", cls.colsample_bylevel)
        _set_pos_float("ML_XGB_GAMMA", "gamma", cls.gamma)
        _set_pos_float("ML_XGB_REG_ALPHA", "reg_alpha", cls.reg_alpha)
        _set_pos_float("ML_XGB_REG_LAMBDA", "reg_lambda", cls.reg_lambda)

        if "ML_XGB_TREE_METHOD" in source:
            kwargs["tree_method"] = source["ML_XGB_TREE_METHOD"].strip()
        if "ML_XGB_GPU_ID" in source:
            kwargs["gpu_id"] = _env_non_negative_int(source, "ML_XGB_GPU_ID", cls.gpu_id)
        if "ML_XGB_OBJECTIVE" in source:
            kwargs["objective"] = source["ML_XGB_OBJECTIVE"].strip()
        if "ML_XGB_EVAL_METRIC" in source:
            kwargs["eval_metric"] = source["ML_XGB_EVAL_METRIC"].strip()
        if "ML_XGB_ENABLE_SHAP" in source:
            kwargs["enable_shap"] = _env_truthy(source, "ML_XGB_ENABLE_SHAP", False)
        if "ML_XGB_MULTI_ASSET" in source:
            kwargs["multi_asset"] = _env_truthy(source, "ML_XGB_MULTI_ASSET", False)
        if "ML_XGB_CROSS_SECTIONAL_FEATURES" in source:
            kwargs["cross_sectional_features"] = _env_truthy(
                source,
                "ML_XGB_CROSS_SECTIONAL_FEATURES",
                True,
            )

        sector_map_raw = source.get("ML_XGB_SECTOR_MAP")
        if sector_map_raw:
            try:
                kwargs["sector_map"] = json.loads(sector_map_raw)
            except json.JSONDecodeError:
                # Leave unset on invalid JSON
                pass

        monotonic_raw = source.get("ML_XGB_MONOTONIC_CONSTRAINTS")
        if monotonic_raw:
            try:
                parsed_constraints = json.loads(monotonic_raw)
            except json.JSONDecodeError:
                parsed_constraints = None
            if isinstance(parsed_constraints, dict):
                kwargs["monotonic_constraints"] = {
                    str(feature): int(direction)
                    for feature, direction in parsed_constraints.items()
                }

        if "ML_XGB_OPTIMIZE_HYPERPARAMS" in source:
            kwargs["optimize_hyperparams"] = _env_truthy(
                source,
                "ML_XGB_OPTIMIZE_HYPERPARAMS",
                False,
            )
        if "ML_XGB_N_TRIALS" in source:
            kwargs["n_trials"] = _env_positive_int(source, "ML_XGB_N_TRIALS", cls.n_trials)
        if "ML_XGB_OPTIMIZATION_METRIC" in source:
            kwargs["optimization_metric"] = source["ML_XGB_OPTIMIZATION_METRIC"].strip()

        if any(key.startswith("ML_XGB_GPU_") for key in source):
            kwargs["gpu_config"] = XGBoostGPUConfig.from_env(env=source)
        if any(key.startswith("ML_OPTUNA_") for key in source):
            kwargs["optuna_config"] = OptunaConfig.from_env(env=source)

        advanced_keys = (
            "ML_TRAIN_TRACK_FEATURE_DECAY",
            "ML_TRAIN_FEATURE_DECAY_THRESHOLD",
            "ML_TRAIN_FEATURE_HISTORY_WINDOW",
            "ML_TRAIN_CV_STRATEGY",
            "ML_TRAIN_CV_FOLDS",
            "ML_TRAIN_PURGE_GAP",
            "ML_TRAIN_EXPORT_ONNX",
            "ML_TRAIN_ONNX_OUTPUT_PATH",
            "ML_TRAIN_ENABLE_MONITORING",
        )
        if any(key in source for key in advanced_keys):
            kwargs["advanced_config"] = AdvancedTrainingConfig.from_env(env=source)

        return cls(**kwargs)

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
        super().__post_init__()
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
