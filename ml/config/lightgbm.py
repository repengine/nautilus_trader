"""
Configuration for LightGBM model training.

This module provides the comprehensive msgspec-based configuration for LightGBM
training, including all advanced features like GOSS, DART, GPU acceleration, Optuna
optimization, MLflow tracking, and feature monitoring.

"""

from __future__ import annotations

from typing import Any

import msgspec

from ml.config.base import MLTrainingConfig
from ml.config.shared import AdvancedTrainingConfig
from ml.config.shared import LightGBMGPUConfig
# MLflowConfig deprecated - use ModelRegistry
from ml.config.shared import OptunaConfig
from nautilus_trader.common.config import NonNegativeFloat
from nautilus_trader.common.config import NonNegativeInt
from nautilus_trader.common.config import PositiveFloat
from nautilus_trader.common.config import PositiveInt


class GOSSConfig(msgspec.Struct, kw_only=True, frozen=True):
    """
    Gradient-based One-Side Sampling (GOSS) configuration for LightGBM.

    GOSS retains all instances with large gradients and performs random sampling
    on instances with small gradients to reduce dataset size while maintaining accuracy.

    Parameters
    ----------
    enabled : bool, default False
        Enable GOSS sampling strategy.
    top_rate : float, default 0.2
        The retain ratio of large gradient data (0.0, 1.0).
    other_rate : float, default 0.1
        The retain ratio of small gradient data (0.0, 1.0).

    """

    enabled: bool = False
    top_rate: float = 0.2
    other_rate: float = 0.1

    def __post_init__(self) -> None:
        """
        Validate GOSS configuration.
        """
        if not (0.0 < self.top_rate < 1.0):
            msg = f"top_rate must be in (0.0, 1.0), got {self.top_rate}"
            raise ValueError(msg)

        if not (0.0 < self.other_rate < 1.0):
            msg = f"other_rate must be in (0.0, 1.0), got {self.other_rate}"
            raise ValueError(msg)

        if self.top_rate + self.other_rate >= 1.0:
            msg = f"top_rate + other_rate must be < 1.0, got {self.top_rate + self.other_rate}"
            raise ValueError(msg)


class DARTConfig(msgspec.Struct, kw_only=True, frozen=True):
    """
    Dropouts meet Multiple Additive Regression Trees (DART) configuration.

    DART applies dropout to trees to reduce overfitting by randomly dropping trees
    during the boosting process and normalizing tree predictions.

    Parameters
    ----------
    enabled : bool, default False
        Enable DART mode for LightGBM training.
    drop_rate : float, default 0.1
        Dropout rate for trees (0.0, 1.0).
    max_drop : int, default 50
        Maximum number of dropped trees in one iteration.
    skip_drop : float, default 0.5
        Probability of skipping the dropout procedure in a iteration.
    uniform_drop : bool, default False
        If True, use uniform dropout; otherwise use binomial dropout.
    xgboost_dart_mode : bool, default False
        If True, use XGBoost-compatible DART mode.

    """

    enabled: bool = False
    drop_rate: float = 0.1
    max_drop: int = 50
    skip_drop: float = 0.5
    uniform_drop: bool = False
    xgboost_dart_mode: bool = False

    def __post_init__(self) -> None:
        """
        Validate DART configuration.
        """
        if not (0.0 <= self.drop_rate <= 1.0):
            msg = f"drop_rate must be in [0.0, 1.0], got {self.drop_rate}"
            raise ValueError(msg)

        if self.max_drop <= 0:
            msg = f"max_drop must be positive, got {self.max_drop}"
            raise ValueError(msg)

        if not (0.0 <= self.skip_drop <= 1.0):
            msg = f"skip_drop must be in [0.0, 1.0], got {self.skip_drop}"
            raise ValueError(msg)


class EFBConfig(msgspec.Struct, kw_only=True, frozen=True):
    """
    Exclusive Feature Bundling (EFB) configuration for LightGBM.

    EFB bundles exclusive features together to reduce the number of features
    and speed up training while maintaining accuracy.

    Parameters
    ----------
    enabled : bool, default True
        Enable EFB feature bundling (LightGBM default).
    max_conflict_rate : float, default 0.0
        Maximum conflict rate for feature bundling (0.0, 1.0).
        0.0 means no conflicts allowed, higher values allow more conflicts.
    bundle_size : int, default 0
        Maximum number of features in one bundle. 0 means no limit.

    """

    enabled: bool = True
    max_conflict_rate: float = 0.0
    bundle_size: int = 0

    def __post_init__(self) -> None:
        """
        Validate EFB configuration.
        """
        if not (0.0 <= self.max_conflict_rate < 1.0):
            msg = f"max_conflict_rate must be in [0.0, 1.0), got {self.max_conflict_rate}"
            raise ValueError(msg)

        if self.bundle_size < 0:
            msg = f"bundle_size must be non-negative, got {self.bundle_size}"
            raise ValueError(msg)


class LightGBMTrainingConfig(MLTrainingConfig, kw_only=True, frozen=True):
    """
    Comprehensive configuration for LightGBM model training.

    This configuration extends the base MLTrainingConfig with LightGBM-specific
    parameters for gradient boosting with tree-based learners, including advanced
    features like GOSS, DART, EFB, GPU acceleration, Optuna optimization, and
    MLflow tracking.

    Parameters
    ----------
    n_estimators : PositiveInt, default 100
        Number of boosting iterations. Equivalent to number of trees to build.
    max_depth : PositiveInt, default 6
        Maximum depth of a tree. A deeper tree might increase accuracy but also
        lead to overfitting. -1 means no limit.
    learning_rate : PositiveFloat, default 0.1
        Boosting learning rate. Shrinkage rate for the contribution of each tree.
    num_leaves : PositiveInt, default 31
        Maximum number of leaves in one tree. Should be less than 2^max_depth.
    min_child_samples : PositiveInt, default 20
        Minimum number of data points in a leaf. This is a crucial parameter
        to control overfitting.
    min_child_weight : NonNegativeFloat, default 1e-3
        Minimum sum of instance weight (hessian) needed in a child (leaf).
    min_split_gain : NonNegativeFloat, default 0.0
        Minimum loss reduction required to make a further partition on a leaf node.
    subsample : PositiveFloat, default 1.0
        Subsample ratio of the training instances. 0.5 means half of training data.
        Ranges from (0.0, 1.0]. Used to prevent overfitting.
    subsample_freq : NonNegativeInt, default 0
        Frequency of subsample. <=0 means no enable. If > 0, will perform subsample
        at every k iteration.
    colsample_bytree : PositiveFloat, default 1.0
        Subsample ratio of columns when constructing each tree. Ranges from (0.0, 1.0].
    reg_alpha : NonNegativeFloat, default 0.0
        L1 regularization term on weights. Helps prevent overfitting.
    reg_lambda : NonNegativeFloat, default 0.0
        L2 regularization term on weights. Helps prevent overfitting.
    scale_pos_weight : PositiveFloat, default 1.0
        Balance of positive and negative weights. Used for imbalanced datasets.
    objective : str, default "regression"
        Learning objective. Options: "regression", "binary", "multiclass", "lambdarank".
    metric : str, default "rmse"
        Metric for evaluation. Auto-selected based on objective if None.
    boosting_type : str, default "gbdt"
        Boosting type. Options: "gbdt", "dart", "goss", "rf".
    early_stopping_rounds : NonNegativeInt, default 10
        Activates early stopping. Training will stop if metric doesn't improve
        for this many rounds. 0 disables early stopping.
    n_jobs : int, default -1
        Number of parallel threads. -1 means use all available cores.
    random_state : NonNegativeInt, default 42
        Random seed for reproducible results.
    verbosity : int, default -1
        Controls the level of LightGBM's verbosity. -1 = silent.
    force_col_wise : bool, default False
        Force column-wise construction of histograms.
    force_row_wise : bool, default False
        Force row-wise construction of histograms.
    categorical_features : list[str], default []
        List of categorical feature names for native categorical support.
    goss_config : GOSSConfig, optional
        Gradient-based One-Side Sampling configuration.
    dart_config : DARTConfig, optional
        DART (Dropouts meet Multiple Additive Regression Trees) configuration.
    efb_config : EFBConfig, optional
        Exclusive Feature Bundling configuration.
    gpu_config : LightGBMGPUConfig, optional
        GPU acceleration settings.
    optuna_config : OptunaConfig, optional
        Optuna hyperparameter optimization settings.
    mlflow_config : deprecated, use ModelRegistry instead
        MLflow experiment tracking settings.
    advanced_config : AdvancedTrainingConfig, optional
        Advanced training features like cross-validation and ONNX export.

    """

    # Core boosting parameters
    n_estimators: PositiveInt = 100
    max_depth: PositiveInt = 6
    learning_rate: PositiveFloat = 0.1
    num_leaves: PositiveInt = 31
    min_child_samples: PositiveInt = 20
    min_child_weight: NonNegativeFloat = 1e-3
    min_split_gain: NonNegativeFloat = 0.0

    # Sampling parameters
    subsample: PositiveFloat = 1.0
    subsample_freq: NonNegativeInt = 0
    colsample_bytree: PositiveFloat = 1.0
    feature_fraction: PositiveFloat = 1.0
    bagging_fraction: PositiveFloat = 1.0
    bagging_freq: NonNegativeInt = 0
    bagging_seed: NonNegativeInt = 3

    # Regularization parameters
    reg_alpha: NonNegativeFloat = 0.0
    reg_lambda: NonNegativeFloat = 0.0
    scale_pos_weight: PositiveFloat = 1.0

    # Learning objective
    objective: str = "regression"
    metric: str = "rmse"
    boosting_type: str = "gbdt"

    # Training control
    early_stopping_rounds: NonNegativeInt = 10
    n_jobs: int = -1
    random_state: NonNegativeInt = 42
    verbosity: int = -1

    # Memory optimization
    force_col_wise: bool = False
    force_row_wise: bool = False

    # Categorical features
    categorical_features: list[str] = msgspec.field(default_factory=list)

    # Advanced configuration components
    goss_config: GOSSConfig | None = None
    dart_config: DARTConfig | None = None
    efb_config: EFBConfig | None = None
    gpu_config: LightGBMGPUConfig | None = None
    optuna_config: OptunaConfig | None = None
    mlflow_config: None = None  # deprecated
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
        return "./models/lightgbm.onnx"

    @property
    def enable_monitoring(self) -> bool:
        """
        Whether to enable Prometheus metrics collection.
        """
        return self.advanced_config.enable_monitoring if self.advanced_config else True

    def __post_init__(self) -> None:
        """
        Validate configuration after initialization.
        """
        # Validate LightGBM-specific parameters
        if not (0.0 < self.learning_rate <= 1.0):
            msg = f"learning_rate must be in (0.0, 1.0], got {self.learning_rate}"
            raise ValueError(msg)

        if not (0.0 < self.subsample <= 1.0):
            msg = f"subsample must be in (0.0, 1.0], got {self.subsample}"
            raise ValueError(msg)

        if not (0.0 < self.colsample_bytree <= 1.0):
            msg = f"colsample_bytree must be in (0.0, 1.0], got {self.colsample_bytree}"
            raise ValueError(msg)

        # Check num_leaves vs max_depth relationship
        if self.max_depth > 0 and self.num_leaves >= 2**self.max_depth:
            msg = f"num_leaves ({self.num_leaves}) should be less than 2^max_depth ({2**self.max_depth}) to avoid overfitting"
            raise ValueError(msg)

        # Validate boosting type
        valid_boosting_types = ["gbdt", "dart", "goss", "rf"]
        if self.boosting_type not in valid_boosting_types:
            msg = f"boosting_type must be one of {valid_boosting_types}, got {self.boosting_type}"
            raise ValueError(msg)

        # Validate objective
        valid_objectives = ["regression", "binary", "multiclass", "lambdarank"]
        if self.objective not in valid_objectives:
            msg = f"objective must be one of {valid_objectives}, got {self.objective}"
            raise ValueError(msg)

        # GOSS and DART are mutually exclusive
        if (
            self.goss_config
            and self.goss_config.enabled
            and self.dart_config
            and self.dart_config.enabled
        ):
            msg = "GOSS and DART cannot be enabled simultaneously"
            raise ValueError(msg)

        # Handle boosting type from configs
        if self.goss_config and self.goss_config.enabled:
            object.__setattr__(self, "boosting_type", "goss")
        elif self.dart_config and self.dart_config.enabled:
            object.__setattr__(self, "boosting_type", "dart")

        # Ensure force_col_wise and force_row_wise are not both True
        if self.force_col_wise and self.force_row_wise:
            msg = "force_col_wise and force_row_wise cannot both be True"
            raise ValueError(msg)

    def get_lgb_params(self) -> dict[str, Any]:
        """
        Get LightGBM parameters as a dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary of LightGBM parameters suitable for model initialization.

        """
        params = {
            "boosting_type": self.boosting_type,
            "objective": self.objective,
            "metric": self.metric,
            "num_iterations": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "min_child_weight": self.min_child_weight,
            "min_split_gain": self.min_split_gain,
            "subsample": self.subsample,
            "subsample_freq": self.subsample_freq,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "scale_pos_weight": self.scale_pos_weight,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbosity": self.verbosity,
        }

        # Apply memory optimization flags
        if self.force_col_wise:
            params["force_col_wise"] = True
        if self.force_row_wise:
            params["force_row_wise"] = True

        # Apply GOSS settings if enabled
        if self.goss_config and self.goss_config.enabled:
            params.update(
                {
                    "boosting_type": "goss",
                    "top_rate": self.goss_config.top_rate,
                    "other_rate": self.goss_config.other_rate,
                },
            )

        # Apply DART settings if enabled
        elif self.dart_config and self.dart_config.enabled:
            params.update(
                {
                    "boosting_type": "dart",
                    "drop_rate": self.dart_config.drop_rate,
                    "max_drop": self.dart_config.max_drop,
                    "skip_drop": self.dart_config.skip_drop,
                    "uniform_drop": self.dart_config.uniform_drop,
                    "xgboost_dart_mode": self.dart_config.xgboost_dart_mode,
                },
            )

        # Apply EFB settings
        if self.efb_config:
            params.update(
                {
                    "enable_bundle": self.efb_config.enabled,
                    "max_conflict_rate": self.efb_config.max_conflict_rate,
                },
            )
            if self.efb_config.bundle_size > 0:
                params["max_bundle"] = self.efb_config.bundle_size

        # Apply GPU settings if enabled
        if self.gpu_config and self.gpu_config.enabled:
            params.update(
                {
                    "device_type": "gpu",
                    "gpu_platform_id": self.gpu_config.platform_id,
                    "gpu_device_id": self.gpu_config.device_id,
                    "gpu_use_dp": self.gpu_config.gpu_use_dp,
                },
            )

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
        if self.gpu_config and self.gpu_config.enabled:
            try:
                from ml._imports import HAS_LIGHTGBM
                from ml._imports import lgb as _lgb

                if not HAS_LIGHTGBM or _lgb is None:
                    raise ImportError("lightgbm not installed")

                # Try to create a simple dataset to test availability
                import numpy as np

                rng = np.random.default_rng(42)
                test_data = rng.standard_normal((10, 5))
                test_labels = rng.integers(0, 2, 10)
                test_dataset = _lgb.Dataset(test_data, label=test_labels)
                del test_dataset, test_data, test_labels
            except Exception as e:
                warnings.append(f"GPU acceleration requested but may not be available: {e}")

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
UnifiedLightGBMConfig = LightGBMTrainingConfig


# Explicit exports
__all__ = [
    "DARTConfig",
    "EFBConfig",
    "GOSSConfig",
    "LightGBMTrainingConfig",
    "UnifiedLightGBMConfig",  # Backward compatibility
]
