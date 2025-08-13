
"""
Base configuration classes for ML components using msgspec.

These configuration classes provide type-safe configuration for ML actors, strategies,
and training components, following Nautilus conventions.

"""

from __future__ import annotations

from typing import Any, Literal

from msgspec import ValidationError

from ml.config.constants import Providers
from nautilus_trader.common.config import NautilusConfig
from nautilus_trader.common.config import NonNegativeFloat
from nautilus_trader.common.config import NonNegativeInt
from nautilus_trader.common.config import PositiveFloat
from nautilus_trader.common.config import PositiveInt
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import ComponentId
from nautilus_trader.model.identifiers import InstrumentId


class MLFeatureConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration for ML feature engineering.

    Parameters
    ----------
    lookback_window : PositiveInt, default 100
        The number of historical bars to consider for feature engineering.
    indicators : dict[str, dict[str, Any]], optional
        Dictionary of indicator configurations, where keys are indicator names
        and values are dictionaries of indicator parameters.
    feature_names : list[str], optional
        List of feature names to compute. If None, all available features are computed.
    normalize_features : bool, default True
        Whether to normalize features to [0, 1] range.
    fill_missing_with : float, default 0.0
        Value to use for missing data imputation.
    average_volume : PositiveFloat, default 1000000.0
        Average volume used for volume feature normalization.

    """

    lookback_window: PositiveInt = 100
    indicators: dict[str, dict[str, Any]] | None = None
    feature_names: list[str] | None = None
    normalize_features: bool = True
    fill_missing_with: float = 0.0
    average_volume: PositiveFloat = 1000000.0


class MLInferenceConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration for ML inference components.

    Parameters
    ----------
    model_path : str, optional
        Path to the trained model file (supports .pkl, .joblib, .onnx formats).
        Either model_path or model_id must be provided.
    model_id : str, optional
        Model ID to load from the unified registry.
        Either model_path or model_id must be provided.
    registry_path : str, optional
        Path to the model registry. Required if using model_id.
    prediction_threshold : NonNegativeFloat, default 0.5
        Minimum confidence threshold for predictions to be considered valid.
    max_inference_latency_ms : PositiveFloat, default 5.0
        Maximum allowed inference latency in milliseconds.
    feature_config : MLFeatureConfig, optional
        Configuration for feature engineering. If None, uses default configuration.
    batch_size : PositiveInt, default 1
        Batch size for model inference (for models that support batching).
    warm_up_period : NonNegativeInt, default 50
        Number of bars to process before starting predictions (for indicator initialization).
    use_manifest_features : bool, default True
        If True and using model_id, use feature schema from model manifest.
        If False, use feature_config even with manifest-based models.

    """

    model_path: str | None = None
    model_id: str | None = None
    registry_path: str | None = None
    prediction_threshold: NonNegativeFloat = 0.5
    max_inference_latency_ms: PositiveFloat = 5.0
    feature_config: MLFeatureConfig | None = None
    batch_size: PositiveInt = 1
    warm_up_period: NonNegativeInt = 50
    use_manifest_features: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.model_path and not self.model_id:
            raise ValidationError("Either model_path or model_id must be provided")
        if self.model_id and not self.registry_path:
            raise ValidationError("registry_path is required when using model_id")
        if self.model_path and self.model_id:
            raise ValidationError("Cannot specify both model_path and model_id")


class CircuitBreakerConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration for circuit breaker pattern.

    Parameters
    ----------
    failure_threshold : PositiveInt, default 5
        Number of consecutive failures before opening circuit.
    recovery_timeout : PositiveInt, default 60
        Time in seconds before attempting to close circuit.
    success_threshold : PositiveInt, default 3
        Number of consecutive successes required to close circuit.

    """

    failure_threshold: PositiveInt = 5
    recovery_timeout: PositiveInt = 60
    success_threshold: PositiveInt = 3


class MLActorConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration for ML actors with enhanced production features.

    Parameters
    ----------
    model_path : str
        Path to the trained model file (supports .pkl, .joblib, .onnx formats).
    model_id : str
        Unique identifier for the model (required for tracking).
    bar_type : BarType
        The bar type to subscribe to for ML inference.
    instrument_id : InstrumentId
        The instrument to perform ML inference on.
    prediction_threshold : NonNegativeFloat, default 0.5
        Minimum confidence threshold for predictions to be considered valid.
    max_inference_latency_ms : PositiveFloat, default 5.0
        Maximum allowed inference latency in milliseconds.
    feature_config : MLFeatureConfig, optional
        Configuration for feature engineering. If None, uses default configuration.
    batch_size : PositiveInt, default 1
        Batch size for model inference (for models that support batching).
    warm_up_period : NonNegativeInt, default 50
        Number of bars to process before starting predictions (for indicator initialization).
    publish_signals : bool, default True
        Whether to publish ML signals to the message bus.
    signal_data_type : str, default "MLSignal"
        The data type name for published ML signals.
    log_predictions : bool, default False
        Whether to log individual predictions (useful for debugging).
    enable_hot_reload : bool, default False
        Whether to enable model hot-reloading capability.
    model_check_interval : PositiveInt, default 300
        Interval in seconds to check for model updates (if hot reload enabled).
    preserve_state_on_reload : bool, default True
        Whether to preserve indicator state during model reloads.
    circuit_breaker_config : CircuitBreakerConfig, optional
        Configuration for circuit breaker fault tolerance.
    enable_health_monitoring : bool, default True
        Whether to enable health status monitoring and reporting.
    max_feature_latency_ms : PositiveFloat, default 0.5
        Maximum allowed feature computation latency in milliseconds.
    allow_pickle : bool, default False
        Whether to allow loading pickle models (security risk in production).
    component_id : ComponentId, optional
        The component ID. If None then the identifier will be taken from the actor class name.
    log_events : bool, default True
        If events should be logged by the actor.
    log_commands : bool, default True
        If commands should be logged by the actor.

    """

    model_path: str
    model_id: str  # NEW - required for tracking
    bar_type: BarType
    instrument_id: InstrumentId
    prediction_threshold: NonNegativeFloat = 0.5
    max_inference_latency_ms: PositiveFloat = 5.0
    feature_config: MLFeatureConfig | None = None
    batch_size: PositiveInt = 1
    warm_up_period: NonNegativeInt = 50
    publish_signals: bool = True
    signal_data_type: str = "MLSignal"
    log_predictions: bool = False
    enable_hot_reload: bool = False
    model_check_interval: PositiveInt = 300
    preserve_state_on_reload: bool = True
    circuit_breaker_config: CircuitBreakerConfig | None = None
    enable_health_monitoring: bool = True
    health_config: HealthMonitorConfig | None = None
    max_feature_latency_ms: PositiveFloat = 0.5
    allow_pickle: bool = False
    component_id: ComponentId | None = None
    log_events: bool = True
    log_commands: bool = True


class HealthMonitorConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration thresholds for the ML actor health monitor.

    Parameters
    ----------
    critical_consecutive_failures : PositiveInt, default 10
        Consecutive failures to mark actor as UNHEALTHY.
    degraded_success_rate_threshold : NonNegativeFloat, default 0.9
        Success rate threshold below which status is DEGRADED.
    degraded_consecutive_failures : PositiveInt, default 3
        Consecutive failures to mark status as DEGRADED.
    degraded_latency_violations : PositiveInt, default 100
        Total latency budget violations to mark status as DEGRADED.
    """

    critical_consecutive_failures: PositiveInt = 10
    degraded_success_rate_threshold: NonNegativeFloat = 0.9
    degraded_consecutive_failures: PositiveInt = 3
    degraded_latency_violations: PositiveInt = 100


class OnnxRuntimeConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    ONNX Runtime configuration used by inference loaders.

    Parameters
    ----------
    graph_optimization_level : str, default "all"
        One of: "disable", "basic", "extended", "all".
    execution_mode : str, default "sequential"
        One of: "sequential", "parallel".
    providers : list[str], default [Providers.CPU]
        Execution providers in priority order.
    intra_threads : int | None, default None
        Intra-op threads; None leaves the default.
    inter_threads : int | None, default None
        Inter-op threads; None leaves the default.
    """

    graph_optimization_level: str = "all"
    execution_mode: str = "sequential"
    providers: list[str] = [Providers.CPU]
    intra_threads: int | None = None
    inter_threads: int | None = None


class MLStrategyConfig(StrategyConfig, kw_only=True, frozen=True):
    """
    Configuration for ML-based trading strategies.

    Parameters
    ----------
    instrument_id : InstrumentId
        The instrument to trade.
    ml_signal_source : str
        The actor ID or data source for ML signals.
    position_size_pct : PositiveFloat, default 0.1
        Percentage of account balance to risk per trade (0.0 to 1.0).
    min_confidence : NonNegativeFloat, default 0.7
        Minimum ML signal confidence required to place trades.
    max_positions : PositiveInt, default 1
        Maximum number of concurrent positions allowed.
    stop_loss_pct : NonNegativeFloat, default 0.02
        Stop loss as percentage of entry price (0.0 to disable).
    take_profit_pct : NonNegativeFloat, default 0.04
        Take profit as percentage of entry price (0.0 to disable).

    """

    instrument_id: InstrumentId
    ml_signal_source: str
    position_size_pct: PositiveFloat = 0.1
    min_confidence: NonNegativeFloat = 0.7
    max_positions: PositiveInt = 1
    stop_loss_pct: NonNegativeFloat = 0.02
    take_profit_pct: NonNegativeFloat = 0.04


class MLTrainingConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration for ML model training.

    Parameters
    ----------
    data_source : str
        Path or identifier for training data source.
    target_column : str, default "target"
        Name of the target column in training data.
    feature_config : MLFeatureConfig, optional
        Configuration for feature engineering. If None, uses default configuration.
    train_test_split : PositiveFloat, default 0.8
        Fraction of data to use for training (remainder used for validation).
    random_seed : NonNegativeInt, default 42
        Random seed for reproducible training results.
    model_params : dict[str, Any], optional
        Model-specific hyperparameters.
    early_stopping_rounds : PositiveInt, default 50
        Number of rounds without improvement before stopping training.
    validation_metric : str, default "accuracy"
        Metric to use for model validation and early stopping.
    save_model_path : str, optional
        Path to save the trained model. If None, model is not saved.

    """

    data_source: str
    target_column: str = "target"
    feature_config: MLFeatureConfig | None = None
    train_test_split: PositiveFloat = 0.8
    random_seed: NonNegativeInt = 42
    model_params: dict[str, Any] | None = None
    early_stopping_rounds: PositiveInt = 50
    validation_metric: str = "accuracy"
    save_model_path: str | None = None


class ModelRegistryConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration for ML model registry.

    Parameters
    ----------
    registry_path : str, default "ml/registry"
        Base path for model registry storage.
    enable_mlflow : bool, default False
        Whether to enable MLflow integration for experiment tracking.
    mlflow_tracking_uri : str, optional
        MLflow tracking server URI. If None, uses local file storage.
    auto_versioning : bool, default True
        Whether to automatically version models on registration.
    max_versions_per_model : PositiveInt, default 10
        Maximum number of versions to keep per model.

    """

    registry_path: str = "ml/registry"
    enable_mlflow: bool = False
    mlflow_tracking_uri: str | None = None
    auto_versioning: bool = True
    max_versions_per_model: PositiveInt = 10


class MultiModelStrategyConfig(MLStrategyConfig, kw_only=True, frozen=True):
    """
    Configuration for strategies consuming multiple models.

    Parameters
    ----------
    target_model_ids : list[str]
        List of model IDs this strategy will consume signals from.
    aggregation_mode : str
        How to aggregate signals from multiple models: "voting", "weighted_average", or "best".
    model_weights : dict[str, float], optional
        Weights for each model ID when using weighted_average aggregation.
        Keys are model IDs, values are weights. If None, uses equal weights.
    required_models : PositiveInt, default 1
        Minimum number of models that must provide signals before trading.

    """

    target_model_ids: list[str]
    aggregation_mode: Literal["voting", "weighted_average", "best"]
    model_weights: dict[str, float] | None = None
    required_models: PositiveInt = 1


class ModelDeploymentConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration for model deployment.

    Parameters
    ----------
    deployment_target : str
        Where to deploy the model: "actor", "strategy", or "both".
    rollout_strategy : str
        How to roll out the model: "immediate", "gradual", or "canary".
    rollout_percentage : NonNegativeFloat, default 100.0
        Percentage of traffic to route to new model (0.0 to 100.0).
    health_check_interval : PositiveInt, default 60
        Interval in seconds between health checks.
    auto_rollback_on_error : bool, default True
        Whether to automatically rollback on deployment errors.

    """

    deployment_target: Literal["actor", "strategy", "both"]
    rollout_strategy: Literal["immediate", "gradual", "canary"]
    rollout_percentage: NonNegativeFloat = 100.0
    health_check_interval: PositiveInt = 60
    auto_rollback_on_error: bool = True

    def __post_init__(self) -> None:
        """Validate percentage is between 0 and 100."""
        if self.rollout_percentage > 100.0:
            raise ValidationError(
                f"rollout_percentage must be between 0.0 and 100.0, got {self.rollout_percentage}"
            )


class CanaryDeploymentConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration for canary deployments.

    Parameters
    ----------
    initial_traffic_percentage : NonNegativeFloat, default 10.0
        Initial percentage of traffic for canary deployment (0.0 to 100.0).
    increment_percentage : NonNegativeFloat, default 10.0
        Percentage to increase traffic by on each promotion (0.0 to 100.0).
    promotion_interval_seconds : PositiveInt, default 300
        Time between automatic promotions in seconds.
    error_threshold_percentage : NonNegativeFloat, default 5.0
        Error rate threshold that triggers automatic rollback (0.0 to 100.0).
    latency_threshold_ms : PositiveFloat, default 100.0
        Latency threshold in milliseconds that triggers rollback.
    auto_promote : bool, default True
        Whether to automatically promote canary based on metrics.
    auto_rollback : bool, default True
        Whether to automatically rollback on threshold violations.

    """

    initial_traffic_percentage: NonNegativeFloat = 10.0
    increment_percentage: NonNegativeFloat = 10.0
    promotion_interval_seconds: PositiveInt = 300
    error_threshold_percentage: NonNegativeFloat = 5.0
    latency_threshold_ms: PositiveFloat = 100.0
    auto_promote: bool = True
    auto_rollback: bool = True

    def __post_init__(self) -> None:
        """Validate all percentages are between 0 and 100."""
        for field_name, value in [
            ("initial_traffic_percentage", self.initial_traffic_percentage),
            ("increment_percentage", self.increment_percentage),
            ("error_threshold_percentage", self.error_threshold_percentage),
        ]:
            if value > 100.0:
                raise ValidationError(
                    f"{field_name} must be between 0.0 and 100.0, got {value}"
                )


class DataLoaderConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration for ML data loader cache and behavior.

    Parameters
    ----------
    cache_size : PositiveInt, default 1000
        Maximum number of cached DataFrames.
    enable_cache : bool, default True
        Whether to enable the internal cache.
    """

    cache_size: PositiveInt = 1000
    enable_cache: bool = True


class RegistryPolicyConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Policy settings for the model registry (SLOs, A/B defaults).

    Parameters
    ----------
    max_inference_latency_ms : PositiveFloat, default 5.0
        SLO for student inference latency.
    ab_models_required : PositiveInt, default 2
        Number of models required for A/B tests.
    """

    max_inference_latency_ms: PositiveFloat = 5.0
    ab_models_required: PositiveInt = 2


class StatsConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Statistical defaults for A/B testing and comparisons.

    Parameters
    ----------
    significance_level : NonNegativeFloat, default 0.05
        Alpha threshold for two-tailed tests.
    power : NonNegativeFloat, default 0.8
        Desired test power.
    small_sample_df_threshold : PositiveInt, default 30
        Degrees-of-freedom threshold to use conservative critical value.
    conservative_critical_value : PositiveFloat, default 2.0
        Critical value to use when df < threshold.
    z_alpha_default : PositiveFloat, default 1.96
        Default z critical for alpha=0.05 two-tailed.
    """

    significance_level: NonNegativeFloat = 0.05
    power: NonNegativeFloat = 0.8
    small_sample_df_threshold: PositiveInt = 30
    conservative_critical_value: PositiveFloat = 2.0
    z_alpha_default: PositiveFloat = 1.96
