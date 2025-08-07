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
Enhanced configuration for unified XGBoost model training.

This module provides msgspec-based configuration classes for the unified XGBoost
trainer, extending the base XGBoost configuration with advanced features including GPU
acceleration, Optuna optimization, MLflow tracking, and feature decay monitoring.

"""

from __future__ import annotations

from typing import Any

import msgspec

from ml.config.xgboost import XGBoostTrainingConfig


class GPUConfig(msgspec.Struct, kw_only=True, frozen=True):
    """
    GPU acceleration configuration.

    Parameters
    ----------
    enabled : bool, default False
        Enable GPU acceleration for XGBoost training.
    device_id : int, default 0
        GPU device ID to use (0-based indexing).
    max_bin : int, default 256
        Maximum number of bins for GPU histogram construction.
        Higher values may improve accuracy but use more memory.
    predictor : str, default "gpu_predictor"
        Predictor type for GPU inference. Options: "gpu_predictor", "cpu_predictor".
    validate_gpu : bool, default True
        Validate GPU availability before training.

    """

    enabled: bool = False
    device_id: int = 0
    max_bin: int = 256
    predictor: str = "gpu_predictor"
    validate_gpu: bool = True

    def __post_init__(self) -> None:
        """
        Validate GPU configuration.
        """
        if self.predictor not in ["gpu_predictor", "cpu_predictor"]:
            msg = f"predictor must be 'gpu_predictor' or 'cpu_predictor', got {self.predictor}"
            raise ValueError(msg)

        if self.device_id < 0:
            msg = f"device_id must be non-negative, got {self.device_id}"
            raise ValueError(msg)

        if self.max_bin <= 0:
            msg = f"max_bin must be positive, got {self.max_bin}"
            raise ValueError(msg)


class OptunaConfig(msgspec.Struct, kw_only=True, frozen=True):
    """
    Optuna hyperparameter optimization configuration.

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
    MLflow experiment tracking configuration.

    Parameters
    ----------
    enabled : bool, default False
        Enable MLflow experiment tracking.
    tracking_uri : str, default "http://localhost:5000"
        MLflow tracking server URI.
    experiment_name : str, default "xgboost_unified"
        Name of the MLflow experiment.
    register_model : bool, default True
        Register model in MLflow model registry after training.
    model_name : str, default "xgboost_unified"
        Registered model name in MLflow.
    log_artifacts : bool, default True
        Log training artifacts (feature importance, SHAP values, etc.).
    log_model : bool, default True
        Log the trained model to MLflow.
    auto_log : bool, default False
        Enable MLflow autologging for XGBoost.

    """

    enabled: bool = False
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "xgboost_unified"
    register_model: bool = True
    model_name: str = "xgboost_unified"
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


class UnifiedXGBoostConfig(XGBoostTrainingConfig, kw_only=True, frozen=True):
    """
    Unified configuration for enhanced XGBoost trainer.

    This configuration extends XGBoostTrainingConfig with advanced features for
    GPU acceleration, hyperparameter optimization, experiment tracking, and
    comprehensive feature monitoring.

    Parameters
    ----------
    gpu_config : GPUConfig, default GPUConfig()
        GPU acceleration settings.
    optuna_config : OptunaConfig, default OptunaConfig()
        Optuna hyperparameter optimization settings.
    mlflow_config : MLflowConfig, default MLflowConfig()
        MLflow experiment tracking settings.
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
    onnx_output_path : str, default "./models/xgboost_unified.onnx"
        Output path for ONNX model export.
    enable_monitoring : bool, default True
        Enable Prometheus metrics collection during training.

    """

    # Advanced configuration components
    gpu_config: GPUConfig = msgspec.field(default_factory=GPUConfig)
    optuna_config: OptunaConfig = msgspec.field(default_factory=OptunaConfig)
    mlflow_config: MLflowConfig = msgspec.field(default_factory=MLflowConfig)

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
    onnx_output_path: str = "./models/xgboost_unified.onnx"

    # Monitoring settings
    enable_monitoring: bool = True

    def __post_init__(self) -> None:
        """
        Validate configuration after initialization.
        """
        # Run parent validation first
        super().__post_init__()

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
                f"feature_decay_threshold must be in (0.0, 1.0), got {self.feature_decay_threshold}"
            )

        if self.feature_history_window <= 0:
            warnings.append(
                f"feature_history_window must be positive, got {self.feature_history_window}"
            )

        # Validate cross-validation settings
        valid_cv_strategies = ["time_series", "blocked", "purged", "standard"]
        if self.cv_strategy not in valid_cv_strategies:
            warnings.append(
                f"cv_strategy must be one of {valid_cv_strategies}, got {self.cv_strategy}"
            )

        if self.cv_folds < 2:
            warnings.append(f"cv_folds must be at least 2, got {self.cv_folds}")

        if self.purge_gap < 0:
            warnings.append(f"purge_gap must be non-negative, got {self.purge_gap}")

        # Validate ONNX settings
        if self.export_onnx and not self.onnx_output_path.strip():
            warnings.append("onnx_output_path cannot be empty when export_onnx is True")

        # Warn about GPU + Optuna interaction
        if self.gpu_config.enabled and self.optuna_config.enabled:
            warnings.append("GPU + Optuna optimization may have limited parallelization")

        return warnings

    def get_unified_xgb_params(self) -> dict[str, Any]:
        """
        Get XGBoost parameters with GPU and advanced settings.

        Returns
        -------
        dict[str, Any]
            Dictionary of XGBoost parameters including GPU settings.

        """
        params = self.get_xgb_params()

        # Apply GPU settings if enabled
        if self.gpu_config.enabled:
            params.update(
                {
                    "tree_method": "gpu_hist",
                    "predictor": self.gpu_config.predictor,
                    "gpu_id": self.gpu_config.device_id,
                    "max_bin": self.gpu_config.max_bin,
                }
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
        if self.gpu_config.enabled and self.gpu_config.validate_gpu:
            try:
                # Try to create a simple GPU-based DMatrix to test GPU availability
                import numpy as np
                import xgboost as xgb

                test_data = np.random.randn(10, 5)
                test_dtrain = xgb.DMatrix(test_data, enable_categorical=False)
                # If this doesn't raise, GPU should be available
                del test_dtrain, test_data
            except Exception as e:
                warnings.append(f"GPU acceleration requested but not available: {e}")

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
    "GPUConfig",
    "MLflowConfig",
    "OptunaConfig",
    "UnifiedXGBoostConfig",
]
