"""
Base configuration classes for ML components using msgspec.

These configuration classes provide type-safe configuration for ML actors, strategies,
and training components, following Nautilus conventions.

"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from msgspec import ValidationError
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import ComponentId
from nautilus_trader.model.identifiers import InstrumentId

from ml.config._env_utils import ensure_env as _ensure_env
from ml.config._env_utils import env_non_negative_int as _env_non_negative_int
from ml.config._env_utils import env_positive_float as _env_positive_float
from ml.config._env_utils import env_positive_int as _env_positive_int
from ml.config._env_utils import env_truthy as _env_truthy
from ml.config._env_utils import resolve_db_connection as _resolve_db_connection
from ml.config.registry import ModelRegistryConfig as ModelRegistryConfig
from nautilus_trader.common.config import NautilusConfig
from nautilus_trader.common.config import NonNegativeFloat
from nautilus_trader.common.config import NonNegativeInt
from nautilus_trader.common.config import PositiveFloat
from nautilus_trader.common.config import PositiveInt
from nautilus_trader.config import StrategyConfig


LOGGER = logging.getLogger(__name__)


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
    use_dummy_stores : bool, default False
        If True, use DummyStore implementations that don't persist data (for testing).
        If False, require real store initialization (production mode).

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
    use_dummy_stores: bool = False

    def __post_init__(self) -> None:
        """
        Validate configuration.
        """
        if not self.model_path and not self.model_id:
            raise ValidationError("Either model_path or model_id must be provided")
        if self.model_id and not self.registry_path:
            raise ValidationError("registry_path is required when using model_id")
        if self.model_path and self.model_id:
            raise ValidationError("Cannot specify both model_path and model_id")

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
    ) -> MLInferenceConfig:
        """
        Build :class:`MLInferenceConfig` from environment variables.

        Environment overrides
        ---------------------
        MODEL_PATH / ML_MODEL_PATH
            Path to the on-disk model artifact.
        MODEL_ID / ML_MODEL_ID
            Registry model identifier; inferred from MODEL_PATH stem when unset.
        MODEL_REGISTRY_DIR / ML_MODEL_REGISTRY_DIR
            Path to registry root when using ``model_id``.
        ML_PREDICTION_THRESHOLD
            Prediction threshold within [0, 1].
        ML_MAX_INFERENCE_LATENCY_MS
            Maximum inference latency in milliseconds.
        ML_INFERENCE_BATCH_SIZE
            Batch size for inference.
        ML_WARM_UP_PERIOD
            Warm-up period in bars.
        ML_USE_MANIFEST_FEATURES
            Toggle manifest feature usage.
        USE_DUMMY_STORES / ML_USE_DUMMY_STORES
            Toggle dummy store usage (testing only).
        """
        source = _ensure_env(env)

        model_path_env = source.get("MODEL_PATH") or source.get("ML_MODEL_PATH")
        model_id_env = source.get("MODEL_ID") or source.get("ML_MODEL_ID")

        if not model_path_env and not model_id_env:
            raise ValueError("MODEL_PATH or MODEL_ID environment variables are required")

        registry_path = source.get("MODEL_REGISTRY_DIR") or source.get("ML_MODEL_REGISTRY_DIR")
        if model_id_env and not registry_path:
            raise ValueError("MODEL_REGISTRY_DIR environment variable is required when MODEL_ID is set")

        prediction_threshold = _env_positive_float(
            source,
            "ML_PREDICTION_THRESHOLD",
            0.5,
        )
        if prediction_threshold > 1.0:
            LOGGER.debug(
                "prediction_threshold_out_of_bounds",
                extra={"value": prediction_threshold},
            )
            prediction_threshold = 0.5

        max_latency_ms = _env_positive_float(
            source,
            "ML_MAX_INFERENCE_LATENCY_MS",
            5.0,
        )
        batch_size = _env_positive_int(source, "ML_INFERENCE_BATCH_SIZE", 1)
        warm_up_period = _env_non_negative_int(source, "ML_WARM_UP_PERIOD", 50)

        use_manifest = _env_truthy(source, "ML_USE_MANIFEST_FEATURES", True)
        use_dummy = _env_truthy(source, "ML_USE_DUMMY_STORES", False) or _env_truthy(
            source,
            "USE_DUMMY_STORES",
            False,
        )

        model_path_value = model_path_env if model_id_env is None else None
        inferred_model_id = model_id_env

        return cls(
            model_path=model_path_value,
            model_id=inferred_model_id,
            registry_path=registry_path,
            prediction_threshold=prediction_threshold,
            max_inference_latency_ms=max_latency_ms,
            feature_config=None,
            batch_size=batch_size,
            warm_up_period=warm_up_period,
            use_manifest_features=use_manifest,
            use_dummy_stores=use_dummy,
        )


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
    # Legacy alias (deprecated): allow tests/configs using `half_open_attempts`
    half_open_attempts: PositiveInt | None = None

    def __post_init__(self) -> None:
        """
        Normalize legacy aliases while preserving immutability.
        """
        # Map legacy `half_open_attempts` to `success_threshold` if provided.
        if (
            self.half_open_attempts is not None
            and self.half_open_attempts != self.success_threshold
        ):
            object.__setattr__(self, "success_threshold", int(self.half_open_attempts))


class MLActorConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration for ML actors with enhanced production features.

    Parameters
    ----------
    model_path : str
        Path to the trained model file (supports .joblib, .onnx, and native framework formats).
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
    component_id: ComponentId | None = None
    log_events: bool = True
    log_commands: bool = True
    # Security: enforce ONNX-only in production by default
    allow_non_onnx_in_dev: bool = False
    # Integration fields (enable automatic store initialization and testing fallbacks)
    db_connection: str | None = None
    use_dummy_stores: bool = False
    # Async persistence configuration (enabled by default for production performance)
    enable_async_persistence: bool = True
    persistence_queue_size: PositiveInt = 10000
    persistence_flush_interval: PositiveFloat = 1.0
    persistence_batch_size: PositiveInt = 100

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
    ) -> MLActorConfig:
        """
        Build :class:`MLActorConfig` from environment variables.

        Environment overrides
        ---------------------
        MODEL_PATH / ML_MODEL_PATH
            Model artifact path (required when MODEL_ID absent).
        MODEL_ID / ML_MODEL_ID
            Registry identifier; inferred from MODEL_PATH when missing.
        INSTRUMENT_ID
            Instrument identifier string (required).
        BAR_TYPE / ML_BAR_TYPE
            Subscription bar type for inference (required).
        DB_CONNECTION / ML_DB_CONNECTION / DATABASE_URL / NAUTILUS_DB
            Connection string overrides for persistence stores.
        SIGNAL_DATA_TYPE / ML_SIGNAL_DATA_TYPE
            Published signal data type (default ``"MLSignal"``).
        USE_DUMMY_STORES / ML_USE_DUMMY_STORES
            Toggle dummy store usage.
        ML_PUBLISH_SIGNALS, ML_LOG_PREDICTIONS, ML_ENABLE_HOT_RELOAD,
        ML_MODEL_CHECK_INTERVAL, ML_PRESERVE_STATE_ON_RELOAD,
        ML_ENABLE_HEALTH_MONITORING, ML_MAX_FEATURE_LATENCY_MS,
        ML_LOG_EVENTS, ML_LOG_COMMANDS, ML_ALLOW_NON_ONNX_IN_DEV,
        ML_ENABLE_ASYNC_PERSISTENCE, ML_PERSISTENCE_QUEUE_SIZE,
        ML_PERSISTENCE_FLUSH_INTERVAL_S, ML_PERSISTENCE_BATCH_SIZE
            Optional runtime tunables aligned with config fields.
        COMPONENT_ID / ML_COMPONENT_ID
            Optional component identifier.
        """
        source = _ensure_env(env)

        inference_cfg = MLInferenceConfig.from_env(env=source)

        instrument_str = source.get("INSTRUMENT_ID")
        if not instrument_str:
            raise ValueError("INSTRUMENT_ID environment variable is required")
        try:
            instrument_id = InstrumentId.from_str(instrument_str)
        except ValueError as exc:
            raise ValueError(f"Invalid INSTRUMENT_ID: {instrument_str}") from exc

        bar_type_str = source.get("BAR_TYPE") or source.get("ML_BAR_TYPE")
        if not bar_type_str:
            raise ValueError("BAR_TYPE environment variable is required")
        try:
            bar_type = BarType.from_str(bar_type_str)
        except ValueError as exc:
            raise ValueError(f"Invalid BAR_TYPE: {bar_type_str}") from exc

        signal_data_type = source.get("SIGNAL_DATA_TYPE") or source.get("ML_SIGNAL_DATA_TYPE")
        component_id_str = source.get("COMPONENT_ID") or source.get("ML_COMPONENT_ID")
        component_id: ComponentId | None = None
        if component_id_str:
            try:
                component_id = ComponentId(component_id_str)
            except Exception:
                LOGGER.debug(
                    "invalid_component_id_env_override",
                    extra={"value": component_id_str},
                )

        db_connection = _resolve_db_connection(source)

        publish_signals = _env_truthy(source, "ML_PUBLISH_SIGNALS", True)
        log_predictions = _env_truthy(source, "ML_LOG_PREDICTIONS", False)
        enable_hot_reload = _env_truthy(source, "ML_ENABLE_HOT_RELOAD", False)
        model_check_interval = _env_positive_int(source, "ML_MODEL_CHECK_INTERVAL", 300)
        preserve_state = _env_truthy(source, "ML_PRESERVE_STATE_ON_RELOAD", True)
        enable_health = _env_truthy(source, "ML_ENABLE_HEALTH_MONITORING", True)
        max_feature_latency = _env_positive_float(source, "ML_MAX_FEATURE_LATENCY_MS", 0.5)
        log_events = _env_truthy(source, "ML_LOG_EVENTS", True)
        log_commands = _env_truthy(source, "ML_LOG_COMMANDS", True)
        allow_non_onnx = _env_truthy(source, "ML_ALLOW_NON_ONNX_IN_DEV", False) or _env_truthy(
            source,
            "ALLOW_NON_ONNX_IN_DEV",
            False,
        )
        enable_async_persistence = _env_truthy(source, "ML_ENABLE_ASYNC_PERSISTENCE", True)
        persistence_queue_size = _env_positive_int(
            source,
            "ML_PERSISTENCE_QUEUE_SIZE",
            10000,
        )
        persistence_flush_interval = _env_positive_float(
            source,
            "ML_PERSISTENCE_FLUSH_INTERVAL_S",
            1.0,
        )
        persistence_batch_size = _env_positive_int(
            source,
            "ML_PERSISTENCE_BATCH_SIZE",
            100,
        )

        actor_model_id = inference_cfg.model_id
        if not actor_model_id:
            actor_model_id = Path(inference_cfg.model_path or "").stem or "ml_actor_model"

        return cls(
            model_path=inference_cfg.model_path or "",
            model_id=actor_model_id,
            bar_type=bar_type,
            instrument_id=instrument_id,
            prediction_threshold=inference_cfg.prediction_threshold,
            max_inference_latency_ms=inference_cfg.max_inference_latency_ms,
            feature_config=inference_cfg.feature_config,
            batch_size=inference_cfg.batch_size,
            warm_up_period=inference_cfg.warm_up_period,
            publish_signals=publish_signals,
            signal_data_type=signal_data_type or "MLSignal",
            log_predictions=log_predictions,
            enable_hot_reload=enable_hot_reload,
            model_check_interval=model_check_interval,
            preserve_state_on_reload=preserve_state,
            circuit_breaker_config=None,
            enable_health_monitoring=enable_health,
            health_config=None,
            max_feature_latency_ms=max_feature_latency,
            component_id=component_id,
            log_events=log_events,
            log_commands=log_commands,
            allow_non_onnx_in_dev=allow_non_onnx,
            db_connection=db_connection,
            use_dummy_stores=inference_cfg.use_dummy_stores,
            enable_async_persistence=enable_async_persistence,
            persistence_queue_size=persistence_queue_size,
            persistence_flush_interval=persistence_flush_interval,
            persistence_batch_size=persistence_batch_size,
        )


class DataCollectorConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration for the enhanced data collector.

    Attributes
    ----------
    data_dir : str
        Base directory to store collected data.
    storage_limit_gb : PositiveFloat
        Storage budget in gigabytes.
    end_date_iso : str | None
        Optional ISO8601 end date (YYYY-MM-DD). If None, uses current date.

    Environment overrides
    ----------------------
    ML_DATA_TIER1_DIR    -> data_dir
    (legacy) ML_DATA_ENHANCED_DIR -> data_dir
    ML_STORAGE_LIMIT_GB  -> storage_limit_gb
    ML_END_DATE          -> end_date_iso

    """

    data_dir: str = "./data/tier1"
    storage_limit_gb: PositiveFloat = 500.0
    end_date_iso: str | None = None

    _ENV_MAPPING: ClassVar[dict[str, str]] = {
        "data_dir": "ML_DATA_TIER1_DIR",
        "storage_limit_gb": "ML_STORAGE_LIMIT_GB",
        "end_date_iso": "ML_END_DATE",
    }
    # Backward-compatibility for older environments
    _LEGACY_ENV_MAPPING: ClassVar[dict[str, str]] = {
        "data_dir": "ML_DATA_ENHANCED_DIR",
    }

    def __post_init__(self) -> None:
        import os

        # Primary env overrides
        for field, env_var in self._ENV_MAPPING.items():
            if env_value := os.getenv(env_var):
                current = getattr(self, field)
                try:
                    casted = type(current)(env_value) if current is not None else env_value
                except Exception:
                    casted = env_value
                object.__setattr__(self, field, casted)

        # Legacy env overrides (only if primary not provided)
        for field, env_var in self._LEGACY_ENV_MAPPING.items():
            if os.getenv(self._ENV_MAPPING[field]):
                continue  # Prefer primary var when set
            if env_value := os.getenv(env_var):
                current = getattr(self, field)
                try:
                    casted = type(current)(env_value) if current is not None else env_value
                except Exception:
                    casted = env_value
                object.__setattr__(self, field, casted)


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

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.degraded_success_rate_threshold) <= 1.0):
            raise ValidationError("degraded_success_rate_threshold must be within [0, 1]")


class OnnxRuntimeConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    ONNX Runtime configuration placeholder for backward-compat imports.
    """


class OptimizationConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Optimization configuration placeholder for backward-compat imports.
    """


class MLSignalActorConfig(MLActorConfig, kw_only=True, frozen=True):
    """
    Signal actor configuration placeholder for backward-compat imports.
    """


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
    use_strategy_store : bool, default True
        Whether to persist strategy decisions to StrategyStore.
    strategy_store_config : dict[str, Any] | None, default None
        Configuration for StrategyStore (connection_string, batch_size, flush_interval_ms).
    persist_all_signals : bool, default False
        Whether to persist HOLD signals in addition to BUY/SELL.
    execute_trades : bool, default True
        Whether to execute actual trades. If False, the strategy will process signals,
        calculate decisions, persist to stores, and update metrics, but will not submit
        orders to the broker. Useful for testing in production without financial risk.

    """

    instrument_id: InstrumentId
    ml_signal_source: str
    position_size_pct: PositiveFloat = 0.1
    min_confidence: NonNegativeFloat = 0.7
    max_positions: PositiveInt = 1
    stop_loss_pct: NonNegativeFloat = 0.02
    take_profit_pct: NonNegativeFloat = 0.04
    use_strategy_store: bool = True
    strategy_store_config: dict[str, Any] | None = None
    persist_all_signals: bool = False
    execute_trades: bool = False
    # Optional sub-configs for strategy components (protocol-first)
    sizing_config: _SizingConfig | None = None
    risk_config: _RiskConfig | None = None
    execution_config: _ExecutionConfig | None = None
    portfolio_config: _PortfolioConfig | None = None
    analytics_config: _AnalyticsConfig | None = None
    # Optional circuit breaker for strategy-level operations (store writes/orders)
    circuit_breaker_config: CircuitBreakerConfig | None = None

    def __post_init__(self) -> None:
        try:
            super().__post_init__()  # type: ignore[misc]
        except AttributeError:  # pragma: no cover - upstream variant without super
            pass

        if not (0.0 < float(self.position_size_pct) <= 1.0):
            raise ValidationError("position_size_pct must be within (0, 1]")
        if not (0.0 <= float(self.min_confidence) <= 1.0):
            raise ValidationError("min_confidence must be within [0, 1]")
        if self.stop_loss_pct < 0.0:
            raise ValidationError("stop_loss_pct must be non-negative")
        if self.take_profit_pct < 0.0:
            raise ValidationError("take_profit_pct must be non-negative")

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
    ) -> MLStrategyConfig:
        """
        Build :class:`MLStrategyConfig` from environment variables.

        Environment overrides
        ---------------------
        STRATEGY_INSTRUMENT_ID / INSTRUMENT_ID
            Instrument identifier (required).
        ML_SIGNAL_SOURCE
            Upstream actor identifier (required).
        ML_POSITION_SIZE_PCT
            Fractional position sizing within (0, 1].
        ML_MIN_CONFIDENCE
            Minimum confidence threshold within [0, 1].
        ML_MAX_POSITIONS
            Maximum concurrent positions (> 0).
        ML_STOP_LOSS_PCT / ML_TAKE_PROFIT_PCT
            Stop loss / take profit percentages (>= 0).
        ML_USE_STRATEGY_STORE
            Toggle strategy store persistence.
        ML_PERSIST_ALL_SIGNALS
            Toggle persistence of HOLD/neutral signals.
        ML_EXECUTE_TRADES
            Toggle live order submission.
        """
        source = _ensure_env(env)

        instrument_str = source.get("STRATEGY_INSTRUMENT_ID") or source.get("INSTRUMENT_ID")
        if not instrument_str:
            raise ValueError("STRATEGY_INSTRUMENT_ID or INSTRUMENT_ID env variable is required")
        try:
            instrument_id = InstrumentId.from_str(instrument_str)
        except ValueError as exc:
            raise ValueError(f"Invalid strategy instrument identifier: {instrument_str}") from exc

        signal_source = source.get("ML_SIGNAL_SOURCE")
        if not signal_source:
            raise ValueError("ML_SIGNAL_SOURCE environment variable is required")

        position_size_pct = _env_positive_float(source, "ML_POSITION_SIZE_PCT", 0.1)
        if position_size_pct > 1.0:
            LOGGER.debug(
                "position_size_pct_clamped",
                extra={"value": position_size_pct},
            )
            position_size_pct = 1.0

        min_confidence = _env_positive_float(source, "ML_MIN_CONFIDENCE", 0.7)
        if min_confidence > 1.0:
            LOGGER.debug(
                "min_confidence_clamped",
                extra={"value": min_confidence},
            )
            min_confidence = 1.0

        max_positions = _env_positive_int(source, "ML_MAX_POSITIONS", 1)
        stop_loss_pct = _env_positive_float(source, "ML_STOP_LOSS_PCT", 0.02)
        take_profit_pct = _env_positive_float(source, "ML_TAKE_PROFIT_PCT", 0.04)

        use_strategy_store = _env_truthy(source, "ML_USE_STRATEGY_STORE", True)
        persist_all_signals = _env_truthy(source, "ML_PERSIST_ALL_SIGNALS", False)
        execute_trades = _env_truthy(source, "ML_EXECUTE_TRADES", False)

        return cls(
            instrument_id=instrument_id,
            ml_signal_source=signal_source,
            position_size_pct=position_size_pct,
            min_confidence=min_confidence,
            max_positions=max_positions,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            use_strategy_store=use_strategy_store,
            strategy_store_config=None,
            persist_all_signals=persist_all_signals,
            execute_trades=execute_trades,
            sizing_config=None,
            risk_config=None,
            execution_config=None,
            portfolio_config=None,
            analytics_config=None,
            circuit_breaker_config=None,
        )


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
    # FeatureStore integration
    db_connection: str | None = None
    pipeline_spec: Any | None = None


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
        """
        Validate percentage is between 0 and 100.
        """
        if self.rollout_percentage > 100.0:
            raise ValidationError(
                f"rollout_percentage must be between 0.0 and 100.0, got {self.rollout_percentage}",
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
        """
        Validate all percentages are between 0 and 100.
        """
        for field_name, value in [
            ("initial_traffic_percentage", self.initial_traffic_percentage),
            ("increment_percentage", self.increment_percentage),
            ("error_threshold_percentage", self.error_threshold_percentage),
        ]:
            if value > 100.0:
                raise ValidationError(
                    f"{field_name} must be between 0.0 and 100.0, got {value}",
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
    Compatibility shim; see ml.config.registry.RegistryPolicyConfig.
    """


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


if TYPE_CHECKING:  # typing-only to avoid runtime import cycles
    from ml.strategies.analytics import AnalyticsConfig as _AnalyticsConfig
    from ml.strategies.execution import ExecutionConfig as _ExecutionConfig
    from ml.strategies.portfolio import PortfolioConfig as _PortfolioConfig
    from ml.strategies.risk import RiskConfig as _RiskConfig
    from ml.strategies.sizing import SizingConfig as _SizingConfig
