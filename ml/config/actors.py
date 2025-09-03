"""
Actor-related configuration classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from ml.config.base import MLActorConfig
from ml.config.runtime import OnnxRuntimeConfig
from nautilus_trader.common.config import NautilusConfig
from nautilus_trader.common.config import NonNegativeFloat
from nautilus_trader.common.config import PositiveInt


if TYPE_CHECKING:
    # Import enum type for typing only to avoid runtime cycles
    from ml.actors.signal import OptimizationLevel as _OptimizationLevel
    from ml.actors.signal import SignalStrategy as _SignalStrategy
else:  # pragma: no cover
    _OptimizationLevel = object  # type: ignore[misc,assignment]
    _SignalStrategy = object  # type: ignore[misc,assignment]


class OptimizationConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Performance optimization configuration for signal actors.
    """

    level: Literal["standard", "optimized"] | _OptimizationLevel = "standard"
    enable_zero_copy: bool = False
    enable_model_warm_up: bool = False
    warm_up_iterations: PositiveInt = 100
    pre_allocate_buffers: bool = True
    use_lock_free_buffers: bool = False
    reservoir_sample_size: PositiveInt = 1000
    # Back-compat aliases (accepted, no behavioral effect unless used by callers/tests)
    feature_cache_size: PositiveInt | None = None
    enable_profiling: bool = False


class StrategyConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Strategy-specific configuration for signal generation.
    """

    extremes_top_pct: float = 0.1
    momentum_lookback: PositiveInt = 5
    ensemble_weights: dict[str, float] | None = None
    adaptive_volatility_factor: float = 2.0
    min_threshold: float = 0.1
    max_threshold: float = 0.95
    update_frequency: PositiveInt = 10
    # Back-compat aliases expected in some tests
    strategy_type: str | None = None
    threshold_long: float | None = None
    threshold_short: float | None = None


class MLSignalActorConfig(MLActorConfig, kw_only=True, frozen=True):
    """
    Unified configuration for ML Signal Actor with all features.
    """

    signal_strategy: (
        Literal["threshold", "extremes", "momentum", "ensemble", "adaptive"] | _SignalStrategy
    ) = "threshold"
    adaptive_window: PositiveInt = 20
    min_signal_separation_bars: PositiveInt = 3
    feature_importance_threshold: NonNegativeFloat = 0.01
    enable_regime_detection: bool = True
    optimization_config: OptimizationConfig | None = None
    strategy_config: StrategyConfig | None = None
    enable_hot_reload: bool = False
    hot_reload_interval: PositiveInt = 300
    custom_strategy: Any | None = None
    # Optional actor identifier for convenience in tests/logging
    actor_id: str | None = None
    # Feature registry integration
    feature_set_id: str | None = None
    registry_path: str | None = None
    use_registry_features: bool = False
    # ONNX runtime
    onnx_runtime_config: OnnxRuntimeConfig | None = None
    # FeatureStore integration
    use_feature_store: bool = False
    db_connection: str = "postgresql://postgres:postgres@localhost:5432/nautilus"
    persist_features: bool = True
    pipeline_spec: Any | None = None
    # Test mode
    use_dummy_stores: bool = False

    # Back-compat alias fields accepted by legacy tests (mapped in __post_init__)
    optimization: OptimizationConfig | None = None
    strategy: StrategyConfig | None = None

    def __post_init__(self) -> None:
        """Map backward-compat alias fields to canonical fields while frozen."""
        # Map optimization -> optimization_config
        if self.optimization is not None and getattr(self, "optimization_config", None) is None:
            try:
                object.__setattr__(self, "optimization_config", self.optimization)
            except Exception:
                # If the underlying object forbids __setattr__ (e.g., msgspec strict freeze), skip mapping
                pass

        # Map strategy -> strategy_config
        if self.strategy is not None and getattr(self, "strategy_config", None) is None:
            try:
                object.__setattr__(self, "strategy_config", self.strategy)
            except Exception:
                pass

        # Map legacy strategy parameters to canonical fields when provided
        legacy_strat: StrategyConfig | None = self.strategy if self.strategy is not None else None
        if legacy_strat is not None:
            # Strategy type
            if legacy_strat.strategy_type:
                try:
                    stype = legacy_strat.strategy_type
                    if isinstance(stype, str):
                        object.__setattr__(self, "signal_strategy", stype)
                except Exception:
                    # Ignore if not settable
                    pass

            # Thresholds → single prediction_threshold using a conservative merge
            if legacy_strat.threshold_long is not None or legacy_strat.threshold_short is not None:
                # Use the stricter of the two absolute thresholds
                thr_long = abs(legacy_strat.threshold_long or 0.0)
                thr_short = abs(legacy_strat.threshold_short or 0.0)
                merged: float = max(thr_long, thr_short)
                try:
                    object.__setattr__(self, "prediction_threshold", merged)
                except Exception:
                    pass


__all__ = [
    "MLSignalActorConfig",
    "OptimizationConfig",
    "StrategyConfig",
]
