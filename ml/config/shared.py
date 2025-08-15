"""
Shared configuration classes for ML model training.

This module provides shared msgspec-based configuration classes used across different ML
trainers including GPU acceleration, Optuna hyperparameter optimization, and MLflow
experiment tracking.

"""

from __future__ import annotations

import msgspec


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
        Metric to optimize. Options: "sharpe_ratio", "accuracy", "auc", "rmse", "mae", "r2".
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
    experiment_name : str, default "ml_experiment"
        Name of the MLflow experiment.
    register_model : bool, default True
        Register model in MLflow model registry after training.
    model_name : str, default "ml_model"
        Registered model name in MLflow.
    log_artifacts : bool, default True
        Log training artifacts (feature importance, SHAP values, etc.).
    log_model : bool, default True
        Log the trained model to MLflow.
    auto_log : bool, default False
        Enable MLflow autologging for the ML framework.

    """

    enabled: bool = False
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "ml_experiment"
    register_model: bool = True
    model_name: str = "ml_model"
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


class BaseGPUConfig(msgspec.Struct, kw_only=True, frozen=True):
    """
    Base GPU acceleration configuration.

    Parameters
    ----------
    enabled : bool, default False
        Enable GPU acceleration for model training.
    device_id : int, default 0
        GPU device ID to use (0-based indexing).
    validate_gpu : bool, default True
        Validate GPU availability before training.

    """

    enabled: bool = False
    device_id: int = 0
    validate_gpu: bool = True

    def __post_init__(self) -> None:
        """
        Validate base GPU configuration.
        """
        if self.device_id < 0:
            msg = f"device_id must be non-negative, got {self.device_id}"
            raise ValueError(msg)


class XGBoostGPUConfig(BaseGPUConfig, kw_only=True, frozen=True):
    """
    GPU acceleration configuration for XGBoost.

    Parameters
    ----------
    max_bin : int, default 256
        Maximum number of bins for GPU histogram construction.
        Higher values may improve accuracy but use more memory.
    predictor : str, default "gpu_predictor"
        Predictor type for GPU inference. Options: "gpu_predictor", "cpu_predictor".

    """

    max_bin: int = 256
    predictor: str = "gpu_predictor"

    def __post_init__(self) -> None:
        """
        Validate XGBoost GPU configuration.
        """
        super().__post_init__()

        if self.predictor not in ["gpu_predictor", "cpu_predictor"]:
            msg = f"predictor must be 'gpu_predictor' or 'cpu_predictor', got {self.predictor}"
            raise ValueError(msg)

        if self.max_bin <= 0:
            msg = f"max_bin must be positive, got {self.max_bin}"
            raise ValueError(msg)


class LightGBMGPUConfig(BaseGPUConfig, kw_only=True, frozen=True):
    """
    GPU acceleration configuration for LightGBM.

    Parameters
    ----------
    platform_id : int, default -1
        OpenCL platform ID. -1 means auto-detect.
    gpu_use_dp : bool, default False
        Set to True to use double precision math on GPU (slower but more accurate).

    """

    platform_id: int = -1
    gpu_use_dp: bool = False


class AdvancedTrainingConfig(msgspec.Struct, kw_only=True, frozen=True):
    """
    Advanced training features shared across ML frameworks.

    Parameters
    ----------
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
    onnx_output_path : str | None, default None
        Output path for ONNX model export. Auto-generated if None.
    enable_monitoring : bool, default True
        Enable Prometheus metrics collection during training.

    """

    track_feature_decay: bool = True
    feature_decay_threshold: float = 0.3
    feature_history_window: int = 10
    cv_strategy: str = "time_series"
    cv_folds: int = 5
    purge_gap: int = 10
    export_onnx: bool = False
    onnx_output_path: str | None = None
    enable_monitoring: bool = True

    def __post_init__(self) -> None:
        """
        Validate advanced training configuration.
        """
        if not (0.0 < self.feature_decay_threshold <= 1.0):
            msg = (
                f"feature_decay_threshold must be in (0.0, 1.0], got {self.feature_decay_threshold}"
            )
            raise ValueError(msg)

        if self.feature_history_window <= 0:
            msg = f"feature_history_window must be positive, got {self.feature_history_window}"
            raise ValueError(msg)

        valid_strategies = ["time_series", "blocked", "purged", "standard"]
        if self.cv_strategy not in valid_strategies:
            msg = f"cv_strategy must be one of {valid_strategies}, got {self.cv_strategy}"
            raise ValueError(msg)

        if self.cv_folds <= 1:
            msg = f"cv_folds must be > 1, got {self.cv_folds}"
            raise ValueError(msg)

        if self.purge_gap < 0:
            msg = f"purge_gap must be non-negative, got {self.purge_gap}"
            raise ValueError(msg)


# Explicit exports
__all__ = [
    "AdvancedTrainingConfig",
    "BaseGPUConfig",
    "LightGBMGPUConfig",
    "MLflowConfig",
    "OptunaConfig",
    "XGBoostGPUConfig",
]
