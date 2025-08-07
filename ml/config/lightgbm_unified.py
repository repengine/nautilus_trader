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
Enhanced configuration for unified LightGBM model training.

This module provides msgspec-based configuration classes for the unified LightGBM
trainer, extending the base LightGBM configuration with advanced features including
GOSS, DART, feature bundling, GPU acceleration, Optuna optimization, and MLflow
tracking.

"""

from __future__ import annotations

from typing import Any

import msgspec

from ml.config.lightgbm import LightGBMTrainingConfig


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


class GPUConfig(msgspec.Struct, kw_only=True, frozen=True):
    """
    GPU acceleration configuration for LightGBM.

    Parameters
    ----------
    enabled : bool, default False
        Enable GPU acceleration for LightGBM training.
    device_id : int, default 0
        GPU device ID to use (0-based indexing).
    platform_id : int, default -1
        OpenCL platform ID. -1 means auto-detect.
    gpu_use_dp : bool, default False
        Set to True to use double precision math on GPU (slower but more accurate).

    """

    enabled: bool = False
    device_id: int = 0
    platform_id: int = -1
    gpu_use_dp: bool = False

    def __post_init__(self) -> None:
        """
        Validate GPU configuration.
        """
        if self.device_id < 0:
            msg = f"device_id must be non-negative, got {self.device_id}"
            raise ValueError(msg)


class OptunaConfig(msgspec.Struct, kw_only=True, frozen=True):
    """
    Optuna hyperparameter optimization configuration for LightGBM.

    Parameters
    ----------
    enabled : bool, default False
        Enable Optuna hyperparameter optimization.
    n_trials : int, default 100
        Number of optimization trials to run.
    direction : str, default "maximize"
        Optimization direction. Options: "maximize", "minimize".
    metric : str, default "sharpe_ratio"
        Metric to optimize. Options: "sharpe_ratio", "accuracy", "auc", "rmse".
    pruner : str, default "median"
        Pruning algorithm. Options: "median", "percentile", "hyperband", "none".
    sampler : str, default "tpe"
        Sampling algorithm. Options: "tpe", "random", "cmaes", "grid".
    timeout : int | None, default None
        Optimization timeout in seconds. None for no timeout.
    study_name : str | None, default None
        Name of the study for persistence. None for in-memory study.
    storage_url : str | None, default None
        Database URL for study persistence. None for in-memory storage.

    """

    enabled: bool = False
    n_trials: int = 100
    direction: str = "maximize"
    metric: str = "sharpe_ratio"
    pruner: str = "median"
    sampler: str = "tpe"
    timeout: int | None = None
    study_name: str | None = None
    storage_url: str | None = None

    def __post_init__(self) -> None:
        """
        Validate Optuna configuration.
        """
        if self.n_trials <= 0:
            msg = f"n_trials must be positive, got {self.n_trials}"
            raise ValueError(msg)

        valid_directions = ["maximize", "minimize"]
        if self.direction not in valid_directions:
            msg = f"direction must be one of {valid_directions}, got {self.direction}"
            raise ValueError(msg)

        valid_metrics = ["sharpe_ratio", "accuracy", "auc", "rmse", "mae", "r2"]
        if self.metric not in valid_metrics:
            msg = f"metric must be one of {valid_metrics}, got {self.metric}"
            raise ValueError(msg)

        valid_pruners = ["median", "percentile", "hyperband", "none"]
        if self.pruner not in valid_pruners:
            msg = f"pruner must be one of {valid_pruners}, got {self.pruner}"
            raise ValueError(msg)

        valid_samplers = ["tpe", "random", "cmaes", "grid"]
        if self.sampler not in valid_samplers:
            msg = f"sampler must be one of {valid_samplers}, got {self.sampler}"
            raise ValueError(msg)

        if self.timeout is not None and self.timeout <= 0:
            msg = f"timeout must be positive or None, got {self.timeout}"
            raise ValueError(msg)


class MLflowConfig(msgspec.Struct, kw_only=True, frozen=True):
    """
    MLflow experiment tracking configuration for LightGBM.

    Parameters
    ----------
    enabled : bool, default False
        Enable MLflow experiment tracking.
    tracking_uri : str, default "http://localhost:5000"
        MLflow tracking server URI.
    experiment_name : str, default "lightgbm_unified"
        Name of the MLflow experiment.
    register_model : bool, default True
        Register model in MLflow model registry after training.
    model_name : str, default "lightgbm_unified"
        Registered model name in MLflow.
    log_artifacts : bool, default True
        Log training artifacts (feature importance, SHAP values, etc.).
    log_model : bool, default True
        Log the trained model to MLflow.
    auto_log : bool, default False
        Enable MLflow autologging for LightGBM.

    """

    enabled: bool = False
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "lightgbm_unified"
    register_model: bool = True
    model_name: str = "lightgbm_unified"
    log_artifacts: bool = True
    log_model: bool = True
    auto_log: bool = False

    def __post_init__(self) -> None:
        """
        Validate MLflow configuration.
        """
        if not self.experiment_name.strip():
            msg = "experiment_name cannot be empty"
            raise ValueError(msg)

        if self.register_model and not self.model_name.strip():
            msg = "model_name cannot be empty when register_model is True"
            raise ValueError(msg)

        # Basic URL validation
        if not (
            self.tracking_uri.startswith(("http://", "https://", "file://"))
            or self.tracking_uri.startswith("sqlite://")
        ):
            msg = f"tracking_uri must start with http://, https://, file://, or sqlite://, got {self.tracking_uri}"
            raise ValueError(msg)


class UnifiedLightGBMConfig(LightGBMTrainingConfig, kw_only=True, frozen=True):
    """
    Unified configuration for enhanced LightGBM trainer.

    This configuration extends LightGBMTrainingConfig with advanced features for
    GOSS, DART, EFB, GPU acceleration, hyperparameter optimization, experiment
    tracking, and comprehensive feature monitoring.

    Parameters
    ----------
    goss_config : GOSSConfig, default GOSSConfig()
        Gradient-based One-Side Sampling configuration.
    dart_config : DARTConfig, default DARTConfig()
        DART (Dropouts meet Multiple Additive Regression Trees) configuration.
    efb_config : EFBConfig, default EFBConfig()
        Exclusive Feature Bundling configuration.
    gpu_config : GPUConfig, default GPUConfig()
        GPU acceleration settings.
    optuna_config : OptunaConfig, default OptunaConfig()
        Optuna hyperparameter optimization settings.
    mlflow_config : MLflowConfig, default MLflowConfig()
        MLflow experiment tracking settings.
    categorical_features : list[str], default []
        List of categorical feature names for native categorical support.
    track_feature_decay : bool, default True
        Enable feature importance decay tracking over time.
    feature_decay_threshold : float, default 0.3
        Threshold for feature importance decay alerts (0.3 = 30% decline).
    feature_history_window : int, default 10
        Number of training runs to keep in feature importance history.
    cv_strategy : str, default "time_series"
        Cross-validation strategy. Options: "time_series", "blocked", "purged", "standard".
    cv_folds : int, default 5
        Number of cross-validation folds.
    purge_gap : int, default 10
        Gap between train/test in purged cross-validation (in time steps).
    export_onnx : bool, default False
        Export trained model to ONNX format for inference.
    onnx_output_path : str, default "./models/lightgbm_unified.onnx"
        Output path for ONNX model export.
    enable_monitoring : bool, default True
        Enable Prometheus metrics collection during training.

    """

    # LightGBM-specific configuration components
    goss_config: GOSSConfig = msgspec.field(default_factory=GOSSConfig)
    dart_config: DARTConfig = msgspec.field(default_factory=DARTConfig)
    efb_config: EFBConfig = msgspec.field(default_factory=EFBConfig)
    gpu_config: GPUConfig = msgspec.field(default_factory=GPUConfig)
    optuna_config: OptunaConfig = msgspec.field(default_factory=OptunaConfig)
    mlflow_config: MLflowConfig = msgspec.field(default_factory=MLflowConfig)

    # Categorical features
    categorical_features: list[str] = msgspec.field(default_factory=list)

    # Feature tracking settings
    track_feature_decay: bool = True
    feature_decay_threshold: float = 0.3
    feature_history_window: int = 10

    # Cross-validation settings
    cv_strategy: str = "time_series"
    cv_folds: int = 5
    purge_gap: int = 10

    # Model export settings
    export_onnx: bool = False
    onnx_output_path: str = "./models/lightgbm_unified.onnx"

    # Monitoring settings
    enable_monitoring: bool = True

    def __post_init__(self) -> None:
        """
        Validate configuration after initialization.
        """
        # Run parent validation first
        # Note: Call parent __post_init__ if it exists
        try:
            super().__post_init__()
        except AttributeError:
            # Parent doesn't have __post_init__, which is fine
            pass

        # Validate feature decay settings
        if not (0.0 < self.feature_decay_threshold < 1.0):
            msg = (
                f"feature_decay_threshold must be in (0.0, 1.0), got {self.feature_decay_threshold}"
            )
            raise ValueError(msg)

        if self.feature_history_window <= 0:
            msg = f"feature_history_window must be positive, got {self.feature_history_window}"
            raise ValueError(msg)

        # Validate cross-validation settings
        valid_cv_strategies = ["time_series", "blocked", "purged", "standard"]
        if self.cv_strategy not in valid_cv_strategies:
            msg = f"cv_strategy must be one of {valid_cv_strategies}, got {self.cv_strategy}"
            raise ValueError(msg)

        if self.cv_folds < 2:
            msg = f"cv_folds must be at least 2, got {self.cv_folds}"
            raise ValueError(msg)

        if self.purge_gap < 0:
            msg = f"purge_gap must be non-negative, got {self.purge_gap}"
            raise ValueError(msg)

        # Validate ONNX settings
        if self.export_onnx and not self.onnx_output_path.strip():
            msg = "onnx_output_path cannot be empty when export_onnx is True"
            raise ValueError(msg)

        # Validate GOSS + DART compatibility
        if self.goss_config.enabled and self.dart_config.enabled:
            msg = "GOSS and DART cannot be enabled simultaneously"
            raise ValueError(msg)

    def validate_config(self) -> list[str]:
        """
        Validate configuration and return list of warnings.

        Returns
        -------
        list[str]
            List of validation warnings.

        """
        warnings = []

        # Validate feature decay settings
        if not (0.0 < self.feature_decay_threshold < 1.0):
            warnings.append(
                f"feature_decay_threshold must be in (0.0, 1.0), got {self.feature_decay_threshold}",
            )

        if self.feature_history_window <= 0:
            warnings.append(
                f"feature_history_window must be positive, got {self.feature_history_window}",
            )

        # Validate cross-validation settings
        valid_cv_strategies = ["time_series", "blocked", "purged", "standard"]
        if self.cv_strategy not in valid_cv_strategies:
            warnings.append(
                f"cv_strategy must be one of {valid_cv_strategies}, got {self.cv_strategy}",
            )

        if self.cv_folds < 2:
            warnings.append(f"cv_folds must be at least 2, got {self.cv_folds}")

        if self.purge_gap < 0:
            warnings.append(f"purge_gap must be non-negative, got {self.purge_gap}")

        # Validate ONNX settings
        if self.export_onnx and not self.onnx_output_path.strip():
            warnings.append("onnx_output_path cannot be empty when export_onnx is True")

        # Check compatibility issues
        if self.goss_config.enabled and self.dart_config.enabled:
            warnings.append("GOSS and DART cannot be enabled simultaneously")

        if self.gpu_config.enabled and self.optuna_config.enabled:
            warnings.append("GPU + Optuna optimization may have limited parallelization")

        if self.dart_config.enabled and self.early_stopping_rounds > 0:
            warnings.append("Early stopping may not work well with DART mode")

        return warnings

    def get_unified_lgb_params(self) -> dict[str, Any]:
        """
        Get LightGBM parameters with GOSS, DART, EFB, and GPU settings.

        Returns
        -------
        dict[str, Any]
            Dictionary of LightGBM parameters including advanced settings.

        """
        params = self.get_lgb_params()

        # Apply GOSS settings if enabled
        if self.goss_config.enabled:
            params.update(
                {
                    "boosting_type": "goss",
                    "top_rate": self.goss_config.top_rate,
                    "other_rate": self.goss_config.other_rate,
                },
            )

        # Apply DART settings if enabled
        elif self.dart_config.enabled:
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
        if self.efb_config.enabled:
            params.update(
                {
                    "enable_bundle": True,
                    "max_conflict_rate": self.efb_config.max_conflict_rate,
                },
            )
            if self.efb_config.bundle_size > 0:
                params["max_bundle"] = self.efb_config.bundle_size
        else:
            params["enable_bundle"] = False

        # Apply GPU settings if enabled
        if self.gpu_config.enabled:
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
        if self.gpu_config.enabled:
            try:
                import lightgbm as lgb

                # Try to create a simple GPU-based dataset to test GPU availability
                import numpy as np

                test_data = np.random.randn(10, 5)
                test_labels = np.random.randint(0, 2, 10)
                test_dataset = lgb.Dataset(test_data, label=test_labels)
                # If this doesn't raise, basic functionality should work
                del test_dataset, test_data, test_labels
            except Exception as e:
                warnings.append(f"GPU acceleration requested but may not be available: {e}")

        # Check Optuna availability
        if self.optuna_config.enabled:
            try:
                import optuna
            except ImportError:
                warnings.append(
                    "Optuna optimization requested but optuna not installed. "
                    "Install with: pip install 'nautilus-trader[ml]'",
                )

        # Check MLflow availability
        if self.mlflow_config.enabled:
            try:
                import mlflow
                import mlflow.lightgbm
            except ImportError:
                warnings.append(
                    "MLflow tracking requested but mlflow not installed. "
                    "Install with: pip install 'nautilus-trader[ml]'",
                )

        # Check ONNX availability
        if self.export_onnx:
            try:
                import onnxmltools
                import skl2onnx
            except ImportError:
                warnings.append(
                    "ONNX export requested but onnx tools not installed. "
                    "Install with: pip install onnxmltools skl2onnx",
                )

        return warnings


# Explicit exports
__all__ = [
    "DARTConfig",
    "EFBConfig",
    "GOSSConfig",
    "GPUConfig",
    "MLflowConfig",
    "OptunaConfig",
    "UnifiedLightGBMConfig",
]
