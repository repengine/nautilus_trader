"""
Shared configuration classes for ML model training.

This module provides shared msgspec-based configuration classes used across different ML
trainers including GPU acceleration, Optuna hyperparameter optimization, and MLflow
experiment tracking.

"""

from __future__ import annotations

from collections.abc import Mapping

import msgspec

from ml.common.validation_strategies import CV_STRATEGIES
from ml.common.validation_strategies import DEFAULT_CV_STRATEGY
from ml.config._env_utils import ensure_env as _ensure_env
from ml.config._env_utils import env_non_negative_int as _env_non_negative_int
from ml.config._env_utils import env_positive_float as _env_positive_float
from ml.config._env_utils import env_positive_int as _env_positive_int
from ml.config._env_utils import env_truthy as _env_truthy


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

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
    ) -> OptunaConfig:
        """
        Build configuration from environment variables.

        Recognised variables:
            ML_OPTUNA_ENABLED
            ML_OPTUNA_TRIALS
            ML_OPTUNA_DIRECTION
            ML_OPTUNA_METRIC
            ML_OPTUNA_PRUNER
            ML_OPTUNA_SAMPLER
            ML_OPTUNA_TIMEOUT
            ML_OPTUNA_STUDY_NAME
            ML_OPTUNA_STORAGE_URL
        """
        source = _ensure_env(env)

        enabled = _env_truthy(source, "ML_OPTUNA_ENABLED", False)
        n_trials = _env_positive_int(source, "ML_OPTUNA_TRIALS", 100)
        direction = source.get("ML_OPTUNA_DIRECTION", "maximize").strip().lower()
        metric = source.get("ML_OPTUNA_METRIC", "sharpe_ratio").strip().lower()
        pruner = source.get("ML_OPTUNA_PRUNER", "median").strip().lower()
        sampler = source.get("ML_OPTUNA_SAMPLER", "tpe").strip().lower()

        timeout_raw = source.get("ML_OPTUNA_TIMEOUT")
        timeout_val: int | None
        if timeout_raw is None or not timeout_raw.strip():
            timeout_val = None
        else:
            try:
                parsed = int(timeout_raw)
            except ValueError:
                parsed = -1
            timeout_val = parsed if parsed > 0 else None

        study_name = source.get("ML_OPTUNA_STUDY_NAME")
        if study_name is not None and not study_name.strip():
            study_name = None
        storage_url = source.get("ML_OPTUNA_STORAGE_URL")
        if storage_url is not None and not storage_url.strip():
            storage_url = None

        return cls(
            enabled=enabled,
            n_trials=n_trials,
            direction=direction,
            metric=metric,
            pruner=pruner,
            sampler=sampler,
            timeout=timeout_val,
            study_name=study_name,
            storage_url=storage_url,
        )


# MLflowConfig removed - deprecated in favor of ModelRegistry


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

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        prefix: str = "ML_GPU",
    ) -> BaseGPUConfig:
        """
        Build GPU configuration from environment variables.

        Args:
            env: Optional environment mapping; defaults to ``os.environ``.
            prefix: Namespace for variable names (defaults to ``ML_GPU``).
        """
        source = _ensure_env(env)
        enabled = _env_truthy(source, f"{prefix}_ENABLED", False)
        device_id = _env_non_negative_int(source, f"{prefix}_DEVICE_ID", 0)
        validate_gpu = _env_truthy(source, f"{prefix}_VALIDATE", True)

        return cls(
            enabled=enabled,
            device_id=device_id,
            validate_gpu=validate_gpu,
        )


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

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        prefix: str = "ML_XGB_GPU",
    ) -> XGBoostGPUConfig:
        """
        Build XGBoost GPU configuration from environment variables.
        """
        source = _ensure_env(env)
        base = BaseGPUConfig.from_env(env=source, prefix=prefix)
        max_bin = _env_positive_int(source, f"{prefix}_MAX_BIN", 256)
        predictor = source.get(f"{prefix}_PREDICTOR", "gpu_predictor").strip() or "gpu_predictor"

        return cls(
            enabled=base.enabled,
            device_id=base.device_id,
            validate_gpu=base.validate_gpu,
            max_bin=max_bin,
            predictor=predictor,
        )


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

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        prefix: str = "ML_LGBM_GPU",
    ) -> LightGBMGPUConfig:
        """
        Build LightGBM GPU configuration from environment variables.
        """
        source = _ensure_env(env)
        base = BaseGPUConfig.from_env(env=source, prefix=prefix)
        platform_id = _env_non_negative_int(source, f"{prefix}_PLATFORM_ID", -1)
        gpu_use_dp = _env_truthy(source, f"{prefix}_USE_DP", False)

        return cls(
            enabled=base.enabled,
            device_id=base.device_id,
            validate_gpu=base.validate_gpu,
            platform_id=platform_id,
            gpu_use_dp=gpu_use_dp,
        )


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
        Cross-validation strategy. Options: "time_series", "purged".
        Deprecated options ("blocked", "standard") are mapped to "time_series".
    cv_folds : int, default 5
        Number of cross-validation folds.
    purge_gap : int, default 10
        Gap between train/test in purged cross-validation (in time steps).
    embargo_pct : float, default 0.0
        Percentage of samples to embargo after each validation fold (0.0-1.0).
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
    cv_strategy: str = DEFAULT_CV_STRATEGY
    cv_folds: int = 5
    purge_gap: int = 10
    embargo_pct: float = 0.0
    export_onnx: bool = False
    onnx_output_path: str | None = None
    enable_monitoring: bool = True

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        prefix: str = "ML_TRAIN",
    ) -> AdvancedTrainingConfig:
        """
        Build advanced training configuration from environment variables.
        """
        source = _ensure_env(env)

        track_feature_decay = _env_truthy(source, f"{prefix}_TRACK_FEATURE_DECAY", True)
        feature_decay_threshold = _env_positive_float(
            source,
            f"{prefix}_FEATURE_DECAY_THRESHOLD",
            0.3,
        )
        feature_history_window = _env_positive_int(
            source,
            f"{prefix}_FEATURE_HISTORY_WINDOW",
            10,
        )
        cv_strategy = source.get(f"{prefix}_CV_STRATEGY", DEFAULT_CV_STRATEGY).strip().lower()
        cv_folds = _env_positive_int(source, f"{prefix}_CV_FOLDS", 5)
        purge_gap = _env_non_negative_int(source, f"{prefix}_PURGE_GAP", 10)
        embargo_pct = _env_positive_float(source, f"{prefix}_EMBARGO_PCT", 0.0)
        export_onnx = _env_truthy(source, f"{prefix}_EXPORT_ONNX", False)

        onnx_output_path = source.get(f"{prefix}_ONNX_OUTPUT_PATH")
        if onnx_output_path is not None and not onnx_output_path.strip():
            onnx_output_path = None

        enable_monitoring = _env_truthy(source, f"{prefix}_ENABLE_MONITORING", True)

        return cls(
            track_feature_decay=track_feature_decay,
            feature_decay_threshold=feature_decay_threshold,
            feature_history_window=feature_history_window,
            cv_strategy=cv_strategy,
            cv_folds=cv_folds,
            purge_gap=purge_gap,
            embargo_pct=embargo_pct,
            export_onnx=export_onnx,
            onnx_output_path=onnx_output_path,
            enable_monitoring=enable_monitoring,
        )

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

        valid_strategies = set(CV_STRATEGIES) | {"blocked", "standard"}
        if self.cv_strategy not in valid_strategies:
            msg = f"cv_strategy must be one of {sorted(valid_strategies)}, got {self.cv_strategy}"
            raise ValueError(msg)

        if self.cv_folds <= 1:
            msg = f"cv_folds must be > 1, got {self.cv_folds}"
            raise ValueError(msg)

        if self.purge_gap < 0:
            msg = f"purge_gap must be non-negative, got {self.purge_gap}"
            raise ValueError(msg)

        if not 0.0 <= float(self.embargo_pct) < 1.0:
            msg = f"embargo_pct must be in [0.0, 1.0), got {self.embargo_pct}"
            raise ValueError(msg)


# Explicit exports
__all__ = [
    "AdvancedTrainingConfig",
    "BaseGPUConfig",
    "LightGBMGPUConfig",
    "OptunaConfig",
    "XGBoostGPUConfig",
]
