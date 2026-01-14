"""
Signal Strategy Component.

This module implements signal generation strategies for MLSignalActor decomposition.

The component provides:
- 5 strategy implementations (Threshold, Extremes, Momentum, Ensemble, Adaptive)
- Strategy factory with 3-level priority system
- Atomic strategy swapping (prepare → execute pattern)
- Model-driven policy loading via adapter
- Hot path optimized signal generation (P99 <100μs)

Strategy Types:
- ThresholdSignalStrategy: Confidence-based thresholding
- ExtremesStrategy: Percentile-based extreme value detection
- MomentumStrategy: Prediction momentum analysis
- EnsembleStrategy: Weighted voting across multiple strategies
- AdaptiveStrategy: Dynamic threshold adaptation based on volatility

Strategy Factory Priority:
1. Custom strategy (config.custom_strategy) - use as-is
2. Model-driven policy (metadata.decision_policy) - load adapter
3. Built-in mapping (config.signal_strategy) - factory pattern

Hot Path Considerations:
- generate_signal() is called on every bar (hot path)
- Zero allocations in threshold strategy
- Ring buffers used in extremes/momentum for zero-copy
- Pre-computed thresholds and weights

Architecture Patterns (CLAUDE.md):
- Pattern 2: Protocol-First Interface Design (SignalGenerationStrategy ABC)
- Pattern 3: Hot/Cold Path Separation (strategies optimized for hot path)
- Pattern 4: Progressive Fallback Chains (custom → policy → built-in)

"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import MutableMapping
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

from ml.common.logging_utils import log_best_effort
from nautilus_trader.model.data import Bar


if TYPE_CHECKING:
    from logging import Logger

    from ml.actors.base import MLSignal
    from ml.config.actors import MLSignalActorConfig


def _get_ml_signal_class() -> type[MLSignal]:
    """
    Get MLSignal class with deferred import to avoid circular imports.

    Returns
    -------
    type
        The MLSignal class.

    """
    from ml.actors.base import MLSignal

    return MLSignal


# =================================================================================================
# Enums
# =================================================================================================


class SignalStrategy(Enum):
    """
    Signal generation strategy enumeration.

    Used in config to select which built-in strategy to use.

    """

    THRESHOLD = "threshold"
    EXTREMES = "extremes"
    MOMENTUM = "momentum"
    ENSEMBLE = "ensemble"
    ADAPTIVE = "adaptive"


class ThresholdStrategy(Enum):
    """
    Threshold strategy enumeration.

    Defines threshold adaptation modes.

    """

    STATIC = "static"
    REGIME_AWARE = "regime_aware"
    DYNAMIC = "dynamic"


# =================================================================================================
# Signal Generation Strategy Protocol
# =================================================================================================


class SignalGenerationStrategy(ABC):
    """
    Abstract base class for signal generation strategies.

    All strategies must implement generate_signal() with identical signature.
    This protocol enables hot path optimization with zero-allocation guarantees.

    Hot Path Requirements:
    - generate_signal() called on every bar (P99 <100μs overhead)
    - No DataFrame creation, file I/O, or network calls
    - Minimize allocations (pre-allocate and reuse buffers)
    - Use ring buffers from context for zero-copy operations

    """

    @abstractmethod
    def generate_signal(
        self,
        bar: Bar,
        prediction: float,
        confidence: float,
        features: npt.NDArray[np.float32],
        context: MutableMapping[str, Any],
    ) -> MLSignal | None:
        """
        Generate a signal based on the strategy logic.

        Parameters
        ----------
        bar : Bar
            The current bar with OHLCV data.
        prediction : float
            The model prediction value.
        confidence : float
            The confidence score (0.0 to 1.0).
        features : npt.NDArray[np.float32]
            The feature array used for prediction.
        context : MutableMapping[str, Any]
            Context dictionary containing:
            - timestamp_ns: Current timestamp in nanoseconds
            - model_id: Model identifier
            - log_predictions: Whether to include features in signal
            - prediction_history: Historical predictions (list)
            - _prediction_ring: Ring buffer (numpy array)
            - _prediction_ring_index: Ring buffer index (int)
            - _prediction_ring_count: Ring buffer count (int)
            - adaptive_threshold: Adaptive threshold value (float)
            - market_regime: Market regime label (str)

        Returns
        -------
        MLSignal | None
            The generated signal or None if conditions not met.

        Notes
        -----
        - Hot path: This method is called on every bar
        - Must complete in <100μs to stay within P99 budget
        - Should minimize allocations and reuse context buffers

        """
        ...


# Public alias to avoid confusion with trading strategies.
# A SignalPolicy is a decision policy that maps prediction context to an MLSignal.
SignalPolicy = SignalGenerationStrategy


# =================================================================================================
# Built-in Strategy Implementations
# =================================================================================================


class ThresholdSignalStrategy(SignalGenerationStrategy):
    """
    Simple threshold-based signal generation.

    Generates signal when confidence meets or exceeds threshold.
    This is the simplest and fastest strategy (no allocations).

    Hot Path: ✅ YES - Direct comparison only, zero allocations

    Example:
        >>> strategy = ThresholdSignalStrategy(threshold=0.7)
        >>> signal = strategy.generate_signal(bar, 0.8, 0.9, features, context)
        >>> assert signal is not None  # confidence=0.9 >= threshold=0.7

    """

    def __init__(self, threshold: float) -> None:
        """
        Initialize threshold signal strategy.

        Parameters
        ----------
        threshold : float
            The confidence threshold for generating signals.
            Must be in range [0.0, 1.0].

        """
        self.threshold = threshold

    def generate_signal(
        self,
        bar: Bar,
        prediction: float,
        confidence: float,
        features: npt.NDArray[np.float32],
        context: MutableMapping[str, Any],
    ) -> MLSignal | None:
        """
        Generate signal based on confidence threshold.

        Parameters
        ----------
        bar : Bar
            The current bar.
        prediction : float
            The model prediction.
        confidence : float
            The confidence score.
        features : npt.NDArray[np.float32]
            The feature array.
        context : MutableMapping[str, Any]
            Additional context information.

        Returns
        -------
        MLSignal | None
            The generated signal or None if threshold not met.

        Notes
        -----
        - Hot path optimized: single comparison, zero allocations
        - Signal allocated only when threshold met

        """
        if confidence >= self.threshold:
            MLSignal = _get_ml_signal_class()
            return MLSignal(
                instrument_id=bar.bar_type.instrument_id,
                model_id=context.get("model_id", "unknown"),
                prediction=prediction,
                confidence=confidence,
                features=features if context.get("log_predictions", False) else None,
                ts_event=bar.ts_event,
                ts_init=context["timestamp_ns"],
            )
        return None


class ExtremesStrategy(SignalGenerationStrategy):
    """
    Signal generation based on prediction extremes.

    Detects extreme predictions (top/bottom percentiles) using ring buffer
    for zero-copy operations. Uses np.partition instead of sort for efficiency.

    Hot Path: ✅ YES - Ring buffer + partition (no allocations after warm-up)

    Example:
        >>> strategy = ExtremesStrategy(top_pct=0.1, threshold=0.7, window_size=50)
        >>> # Generates signal when prediction in top 10% or bottom 10%
        >>> signal = strategy.generate_signal(bar, 0.95, 0.9, features, context)

    """

    def __init__(self, top_pct: float, threshold: float, window_size: int) -> None:
        """
        Initialize extremes strategy.

        Parameters
        ----------
        top_pct : float
            The percentile for extreme value detection (e.g., 0.1 for top/bottom 10%).
            Must be in range (0.0, 1.0).
        threshold : float
            The confidence threshold.
        window_size : int
            The window size for historical predictions.
            Must be > 0.

        Raises
        ------
        ValueError
            If top_pct is not in range (0.0, 1.0).

        """
        if top_pct <= 0.0 or top_pct >= 1.0:
            raise ValueError(f"top_pct must be in range (0.0, 1.0), got {top_pct}")
        self.top_pct = top_pct
        self.threshold = threshold
        self.window_size = window_size

    def generate_signal(
        self,
        bar: Bar,
        prediction: float,
        confidence: float,
        features: npt.NDArray[np.float32],
        context: MutableMapping[str, Any],
    ) -> MLSignal | None:
        """
        Generate signal based on prediction extremes.

        Parameters
        ----------
        bar : Bar
            The current bar.
        prediction : float
            The model prediction.
        confidence : float
            The confidence score.
        features : npt.NDArray[np.float32]
            The feature array.
        context : MutableMapping[str, Any]
            Additional context information.

        Returns
        -------
        MLSignal | None
            The generated signal or None if not extreme.

        Notes
        -----
        - Uses ring buffer for zero-copy operations
        - np.partition instead of sort for efficiency
        - Window must be full before signals generated

        """
        # Maintain a fixed-size ring buffer of recent predictions to avoid allocations
        ring: npt.NDArray[np.float32]
        scratch: npt.NDArray[np.float32]
        filled: int
        idx: int

        if "_pred_ring" not in context:
            context["_pred_ring"] = np.empty(self.window_size, dtype=np.float32)
            context["_pred_scratch"] = np.empty(self.window_size, dtype=np.float32)
            context["_pred_ring_filled"] = 0
            context["_pred_ring_idx"] = 0

        ring = context["_pred_ring"]
        scratch = context["_pred_scratch"]
        filled = int(context.get("_pred_ring_filled", 0))
        idx = int(context.get("_pred_ring_idx", 0))

        # Update ring buffer with the latest prediction
        ring[idx] = np.float32(prediction)
        idx = (idx + 1) % self.window_size
        filled = min(self.window_size, filled + 1)
        context["_pred_ring_idx"] = idx
        context["_pred_ring_filled"] = filled

        if filled < self.window_size:
            return None

        # Copy current window into scratch and compute thresholds
        # Using np.partition to avoid full sort; this keeps allocations bounded
        scratch[:filled] = ring[:filled]
        # Compute order statistics indices for bottom and top percentiles
        k_top = max(0, min(filled - 1, int(np.ceil((1.0 - self.top_pct) * filled)) - 1))
        k_bottom = max(0, min(filled - 1, int(np.floor(self.top_pct * filled)) - 1))
        top_threshold = float(np.partition(scratch[:filled], k_top)[k_top])
        bottom_threshold = float(np.partition(scratch[:filled], k_bottom)[k_bottom])

        if (
            prediction >= top_threshold or prediction <= bottom_threshold
        ) and confidence >= self.threshold:
            MLSignal = _get_ml_signal_class()
            return MLSignal(
                instrument_id=bar.bar_type.instrument_id,
                model_id=context.get("model_id", "unknown"),
                prediction=prediction,
                confidence=confidence,
                features=features if context.get("log_predictions", False) else None,
                ts_event=bar.ts_event,
                ts_init=context["timestamp_ns"],
            )
        return None


class MomentumStrategy(SignalGenerationStrategy):
    """
    Signal generation based on prediction momentum.

    Calculates momentum using telescoping difference (last - first) / (lookback - 1).
    Prefers ring buffer metadata for zero-allocation hot path.

    Hot Path: ✅ YES - Ring buffer access + difference calculation

    Example:
        >>> strategy = MomentumStrategy(lookback=10, threshold=0.7, momentum_threshold=0.01)
        >>> # Generates signal when abs(momentum) > 0.01 and confidence >= 0.7
        >>> signal = strategy.generate_signal(bar, 0.8, 0.9, features, context)

    """

    def __init__(self, lookback: int, threshold: float, momentum_threshold: float) -> None:
        """
        Initialize momentum strategy.

        Parameters
        ----------
        lookback : int
            The lookback period for momentum calculation.
            Must be > 1 for meaningful momentum.
        threshold : float
            The confidence threshold.
        momentum_threshold : float
            The momentum threshold for signal generation.

        """
        self.lookback = lookback
        self.threshold = threshold
        self.momentum_threshold = momentum_threshold

    def generate_signal(
        self,
        bar: Bar,
        prediction: float,
        confidence: float,
        features: npt.NDArray[np.float32],
        context: MutableMapping[str, Any],
    ) -> MLSignal | None:
        """
        Generate signal based on prediction momentum.

        Parameters
        ----------
        bar : Bar
            The current bar.
        prediction : float
            The model prediction.
        confidence : float
            The confidence score.
        features : npt.NDArray[np.float32]
            The feature array.
        context : MutableMapping[str, Any]
            Additional context information.

        Returns
        -------
        MLSignal | None
            The generated signal or None if momentum insufficient.

        Notes
        -----
        - Prefers ring buffer for hot path performance
        - Falls back to prediction_history list if ring unavailable
        - Momentum = (last - first) / (lookback - 1)

        """
        # Prefer ring-buffer history if provided for zero-allocation hot path
        ring = context.get("_prediction_ring")
        ring_idx = int(context.get("_prediction_ring_index", 0))
        ring_cnt = int(context.get("_prediction_ring_count", 0))
        look = int(self.lookback)
        if ring is not None and ring_cnt >= look:
            cap = int(ring.shape[0])
            # Oldest within the lookback window
            first_idx = (ring_idx - look) % cap
            last_idx = (ring_idx - 1) % cap
            first_val = float(ring[first_idx])
            last_val = float(ring[last_idx])
            # Telescoping sum of diffs => (last - first) / (lookback - 1)
            denom = max(1, look - 1)
            momentum = (last_val - first_val) / denom
        else:
            history = context.get("prediction_history", [])
            if len(history) < look:
                return None
            recent_predictions = history[-look:]
            momentum = np.mean(np.diff(recent_predictions))

        if abs(momentum) > self.momentum_threshold and confidence >= self.threshold:
            MLSignal = _get_ml_signal_class()
            return MLSignal(
                instrument_id=bar.bar_type.instrument_id,
                model_id=context.get("model_id", "unknown"),
                prediction=prediction * (1 + momentum),
                confidence=confidence,
                features=features if context.get("log_predictions", False) else None,
                ts_event=bar.ts_event,
                ts_init=context["timestamp_ns"],
            )
        return None


class EnsembleStrategy(SignalGenerationStrategy):
    """
    Ensemble of multiple strategies with weighted voting.

    Combines multiple sub-strategies using weighted average of confidences.
    Only strategies that vote (return signal) contribute to ensemble score.

    Hot Path: ✅ YES - Delegates to sub-strategies (inherits their performance)

    Example:
        >>> strategies = {
        ...     "threshold": ThresholdSignalStrategy(0.7),
        ...     "extremes": ExtremesStrategy(0.1, 0.7, 50),
        ... }
        >>> weights = {"threshold": 0.6, "extremes": 0.4}
        >>> ensemble = EnsembleStrategy(strategies, weights, threshold=0.7)

    """

    def __init__(
        self,
        strategies: dict[str, SignalGenerationStrategy],
        weights: dict[str, float],
        threshold: float,
    ) -> None:
        """
        Initialize ensemble strategy.

        Parameters
        ----------
        strategies : dict[str, SignalGenerationStrategy]
            Dictionary of named strategies.
        weights : dict[str, float]
            Weights for each strategy (should sum to ~1.0).
        threshold : float
            The ensemble confidence threshold.

        """
        self.strategies = strategies
        self.weights = weights
        self.threshold = threshold

    def generate_signal(
        self,
        bar: Bar,
        prediction: float,
        confidence: float,
        features: npt.NDArray[np.float32],
        context: MutableMapping[str, Any],
    ) -> MLSignal | None:
        """
        Generate signal using weighted ensemble voting.

        Parameters
        ----------
        bar : Bar
            The current bar.
        prediction : float
            The model prediction.
        confidence : float
            The confidence score.
        features : npt.NDArray[np.float32]
            The feature array.
        context : MutableMapping[str, Any]
            Additional context information.

        Returns
        -------
        MLSignal | None
            The ensemble signal or None if threshold not met.

        Notes
        -----
        - Only voting strategies (those that return signal) contribute
        - Ensemble confidence = weighted average of voting strategies
        - Original prediction preserved (not modified)

        """
        ensemble_score = 0.0
        total_weight = 0.0

        for name, strategy in self.strategies.items():
            signal = strategy.generate_signal(bar, prediction, confidence, features, context)
            if signal is not None:
                ensemble_score += self.weights.get(name, 0.0) * confidence
                total_weight += self.weights.get(name, 0.0)

        if total_weight > 0:
            ensemble_confidence = ensemble_score / total_weight
            if ensemble_confidence >= self.threshold:
                MLSignal = _get_ml_signal_class()
                return MLSignal(
                    instrument_id=bar.bar_type.instrument_id,
                    model_id=context.get("model_id", "unknown"),
                    prediction=prediction,
                    confidence=ensemble_confidence,
                    features=features if context.get("log_predictions", False) else None,
                    ts_event=bar.ts_event,
                    ts_init=context["timestamp_ns"],
                )
        return None


class AdaptiveStrategy(SignalGenerationStrategy):
    """
    Adaptive signal generation with dynamic thresholds.

    Uses adaptive_threshold from context (calculated by AdaptiveThresholdComponent).
    Signal strength = confidence / adaptive_threshold must be >= 1.0.

    Hot Path: ✅ YES - Context lookup + ratio calculation

    Example:
        >>> strategy = AdaptiveStrategy(
        ...     base_threshold=0.7,
        ...     volatility_factor=2.0,
        ...     min_threshold=0.1,
        ...     max_threshold=0.95,
        ... )
        >>> # Uses adaptive_threshold from context instead of base_threshold
        >>> signal = strategy.generate_signal(bar, 0.8, 0.9, features, context)

    """

    def __init__(
        self,
        base_threshold: float,
        volatility_factor: float,
        min_threshold: float,
        max_threshold: float,
    ) -> None:
        """
        Initialize the AdaptiveStrategy.

        Parameters
        ----------
        base_threshold : float
            The base confidence threshold for signal generation.
        volatility_factor : float
            Factor for adjusting threshold based on market volatility.
        min_threshold : float
            Minimum allowed threshold value.
        max_threshold : float
            Maximum allowed threshold value.

        """
        self.base_threshold = base_threshold
        self.volatility_factor = volatility_factor
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def generate_signal(
        self,
        bar: Bar,
        prediction: float,
        confidence: float,
        features: npt.NDArray[np.float32],
        context: MutableMapping[str, Any],
    ) -> MLSignal | None:
        """
        Generate adaptive signal based on dynamic thresholds.

        Parameters
        ----------
        bar : Bar
            The current bar data.
        prediction : float
            The model prediction value.
        confidence : float
            The confidence score of the prediction.
        features : npt.NDArray[np.float32]
            The computed feature array.
        context : MutableMapping[str, Any]
            Context dictionary containing adaptive threshold and timestamp.

        Returns
        -------
        MLSignal | None
            The generated signal if threshold is met, otherwise None.

        Notes
        -----
        - Uses adaptive_threshold from context (not base_threshold)
        - Signal strength = confidence / adaptive_threshold
        - Generates signal when signal_strength >= 1.0

        """
        adaptive_threshold = context.get("adaptive_threshold", self.base_threshold)
        signal_strength = confidence / adaptive_threshold if adaptive_threshold > 0 else 0.0

        if signal_strength >= 1.0:
            market_regime = context.get("market_regime", "unknown")
            MLSignal = _get_ml_signal_class()
            return MLSignal(
                instrument_id=bar.bar_type.instrument_id,
                model_id=context.get("model_id", "unknown"),
                prediction=prediction,
                confidence=confidence,
                features=features,
                metadata={
                    "adaptive_threshold": adaptive_threshold,
                    "signal_strength": signal_strength,
                    "market_regime": market_regime,
                },
                ts_event=bar.ts_event,
                ts_init=context["timestamp_ns"],
            )
        return None


# =================================================================================================
# Strategy Swapping Components
# =================================================================================================


class SignalPolicySwapper:
    """
    Atomic signal policy swapping for runtime updates.

    Mirrors the ModelSwapper pattern but for SignalGenerationStrategy
    (aka SignalPolicy) instances. All swap preparation happens off the hot path;
    reading the current policy on the hot path remains a single attribute dereference.

    Swap Pattern:
    1. prepare_swap() - Stage new strategy (off hot path)
    2. execute_swap() - Atomic pointer update (O(1), on hot path)

    Thread Safety:
    - prepare_swap() called off hot path (cold)
    - execute_swap() called on hot path (atomic assignment)

    Example:
        >>> swapper = SignalPolicySwapper()
        >>> swapper.set_current(ThresholdSignalStrategy(0.7))
        >>> # Later, off hot path:
        >>> swapper.prepare_swap(AdaptiveStrategy(...))
        >>> # On hot path:
        >>> if swapper.execute_swap():
        ...     print("Strategy swapped atomically")

    """

    def __init__(self) -> None:
        """
        Initialize SignalPolicySwapper.

        All swap preparation happens off hot path. Reading current_strategy is single
        attribute dereference.

        """
        self._current_strategy: SignalGenerationStrategy | None = None
        self._current_metadata: dict[str, Any] | None = None
        self._next_strategy: SignalGenerationStrategy | None = None
        self._next_metadata: dict[str, Any] | None = None
        self._swap_pending: bool = False
        self._load_error: Exception | None = None

    @property
    def current_strategy(self) -> SignalGenerationStrategy | None:
        """
        Return the current strategy instance, or None if unset.

        Returns
        -------
        SignalGenerationStrategy | None
            Current strategy instance

        """
        return self._current_strategy

    @property
    def current_metadata(self) -> dict[str, Any] | None:
        """
        Return metadata associated with the current strategy, if any.

        Returns
        -------
        dict[str, Any] | None
            Current strategy metadata

        """
        return self._current_metadata

    @property
    def swap_pending(self) -> bool:
        """
        True if a new strategy has been prepared and not yet applied.

        Returns
        -------
        bool
            Whether swap is pending

        """
        return self._swap_pending

    @property
    def load_error(self) -> Exception | None:
        """
        Any error encountered while preparing a swap (if applicable).

        Returns
        -------
        Exception | None
            Load error if any

        """
        return self._load_error

    def set_current(
        self,
        strategy: SignalGenerationStrategy,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Set the current strategy and clear any previous error state.

        Parameters
        ----------
        strategy : SignalGenerationStrategy
            Strategy to set as current
        metadata : dict[str, Any] | None
            Optional metadata to associate with strategy

        """
        self._current_strategy = strategy
        self._current_metadata = metadata or {}
        self._load_error = None

    def prepare_swap(
        self,
        strategy: SignalGenerationStrategy,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Prepare a swap by staging the next strategy instance and metadata.

        This method should be called off the hot path (cold path).

        Parameters
        ----------
        strategy : SignalGenerationStrategy
            Strategy to swap to
        metadata : dict[str, Any] | None
            Optional metadata to associate with strategy

        """
        self._next_strategy = strategy
        self._next_metadata = metadata or {}
        self._swap_pending = True
        self._load_error = None

    def prepare_swap_with_error(self, error: Exception) -> None:
        """
        Record an error during swap preparation and clear any pending swap.

        Parameters
        ----------
        error : Exception
            Error that occurred during preparation

        """
        self._load_error = error
        self._swap_pending = False

    def execute_swap(self) -> bool:
        """
        Atomically promote the prepared strategy to current, if pending.

        This method is called on the hot path and must be O(1).

        Returns
        -------
        bool
            True if a swap was applied; False otherwise.

        """
        if not self._swap_pending:
            return False

        old = self._current_strategy
        self._current_strategy = self._next_strategy
        self._current_metadata = self._next_metadata
        self._next_strategy = None
        self._next_metadata = None
        self._swap_pending = False
        del old
        return True


# Public alias for naming clarity (prefer this name going forward).
StrategySwapper = SignalPolicySwapper


# =================================================================================================
# Signal Strategy Component
# =================================================================================================


class SignalStrategyComponent:
    """
    Component for signal generation strategy management.

    Responsibilities:
    - Create strategies from config/metadata (3-level priority)
    - Manage atomic strategy swapping (prepare → execute pattern)
    - Provide current strategy access for hot path

    Strategy Creation Priority:
    1. Custom strategy (config.custom_strategy) - use as-is
    2. Model-driven policy (metadata.decision_policy) - load adapter
    3. Built-in mapping (config.signal_strategy) - factory

    Hot Path:
    - get_current_strategy() or current_strategy property (single attribute access)
    - apply_pending_swap() (O(1) pointer update if pending)

    Example:
        >>> component = SignalStrategyComponent(config, "actor_1", logger)
        >>> # Initialize with strategy from config
        >>> strategy = component.create_strategy(config, metadata)
        >>> component._swapper.set_current(strategy)
        >>> # Hot path usage
        >>> current = component.current_strategy
        >>> signal = current.generate_signal(bar, prediction, confidence, features, context)
        >>> # Off hot path: prepare swap
        >>> new_strategy = AdaptiveStrategy(...)
        >>> component.prepare_strategy_swap(new_strategy, metadata)
        >>> # On hot path: execute swap
        >>> if component.apply_pending_swap():
        ...     print("Strategy swapped")

    """

    def __init__(
        self,
        config: MLSignalActorConfig,
        actor_id: str,
        log: Logger,
    ) -> None:
        """
        Initialize strategy component.

        Parameters
        ----------
        config : MLSignalActorConfig
            Actor configuration containing strategy settings
        actor_id : str
            Actor identifier for logging
        log : Logger
            Logger instance

        """
        self._config = config
        self._actor_id = actor_id
        self._log = log
        self._swapper = SignalPolicySwapper()

    def create_strategy(
        self,
        config: MLSignalActorConfig,
        metadata: dict[str, Any] | None = None,
    ) -> SignalGenerationStrategy:
        """
        Create strategy from config/metadata with 3-level priority.

        Priority (first match wins):
        1. config.custom_strategy (used as-is)
        2. metadata.decision_policy (load adapter)
        3. config.signal_strategy (built-in factory)

        Parameters
        ----------
        config : MLSignalActorConfig
            Actor configuration containing strategy settings
        metadata : dict[str, Any] | None
            Model metadata that may contain decision_policy

        Returns
        -------
        SignalGenerationStrategy
            Constructed strategy instance

        Raises
        ------
        ValueError
            If strategy type unknown or creation fails

        Notes
        -----
        - Priority 1 (custom) always wins
        - Priority 2 (policy) falls back to Priority 3 on error
        - Priority 3 (built-in) always works (fallback to threshold)

        """
        # Priority 1: Custom strategy (use as-is)
        if config.custom_strategy is not None:
            return cast(SignalGenerationStrategy, config.custom_strategy)

        # Priority 2: Model-driven decision policy (adapter)
        try:
            if metadata:
                policy = metadata.get("decision_policy")
                if policy:
                    from ml.actors.adapters import build_strategy_from_policy

                    cfg = metadata.get("decision_config", {}) if isinstance(metadata, dict) else {}
                    # Cast result to our protocol (adapter returns signal.py protocol)
                    strategy = build_strategy_from_policy(
                        policy_path=str(policy),
                        actor=None,  # type: ignore[arg-type]  # Actor not needed for policy loading
                        config=cfg,
                    )
                    return strategy
        except Exception as exc:
            # Silent fallback to built-ins; keep hot path clean — telemetry debug
            log_best_effort(
                self._log,
                "debug",
                f"ml_actor.decision_policy_load_failed error={exc!r}",
                exc_info=True,
            )

        # Priority 3: Built-in strategy mapping (backwards compatibility)
        return self._create_builtin_strategy(config)

    def _create_builtin_strategy(
        self,
        config: MLSignalActorConfig,
    ) -> SignalGenerationStrategy:
        """
        Create built-in strategy from config.

        Uses config.signal_strategy enum to select strategy type.
        Falls back to ThresholdSignalStrategy if unknown.

        Parameters
        ----------
        config : MLSignalActorConfig
            Actor configuration

        Returns
        -------
        SignalGenerationStrategy
            Built-in strategy instance

        """
        strategy_key = str(config.signal_strategy).lower()
        threshold = config.prediction_threshold

        # Get strategy-specific config (with defaults)
        strat_config = (
            config.strategy_config
            if hasattr(config, "strategy_config") and config.strategy_config
            else None
        )

        def _mk_threshold() -> SignalGenerationStrategy:
            return ThresholdSignalStrategy(threshold)

        def _mk_extremes() -> SignalGenerationStrategy:
            top_pct = strat_config.extremes_top_pct if strat_config else 0.1
            window = config.adaptive_window if hasattr(config, "adaptive_window") else 50
            return ExtremesStrategy(top_pct, threshold, window)

        def _mk_momentum() -> SignalGenerationStrategy:
            lookback = strat_config.momentum_lookback if strat_config else 10
            return MomentumStrategy(lookback, threshold, 0.01)

        def _mk_ensemble() -> SignalGenerationStrategy:
            strategies = {
                "threshold": _mk_threshold(),
                "extremes": _mk_extremes(),
                "momentum": _mk_momentum(),
            }
            weights = (
                strat_config.ensemble_weights
                if strat_config and strat_config.ensemble_weights
                else {
                    "threshold": 0.4,
                    "extremes": 0.3,
                    "momentum": 0.3,
                }
            )
            return EnsembleStrategy(strategies, weights, threshold)

        def _mk_adaptive() -> SignalGenerationStrategy:
            base = threshold
            vol_factor = strat_config.adaptive_volatility_factor if strat_config else 2.0
            min_thresh = strat_config.min_threshold if strat_config else 0.1
            max_thresh = strat_config.max_threshold if strat_config else 0.95
            return AdaptiveStrategy(base, vol_factor, min_thresh, max_thresh)

        factory = {
            "threshold": _mk_threshold,
            SignalStrategy.THRESHOLD.value: _mk_threshold,
            "extremes": _mk_extremes,
            SignalStrategy.EXTREMES.value: _mk_extremes,
            "momentum": _mk_momentum,
            SignalStrategy.MOMENTUM.value: _mk_momentum,
            "ensemble": _mk_ensemble,
            SignalStrategy.ENSEMBLE.value: _mk_ensemble,
            "adaptive": _mk_adaptive,
            SignalStrategy.ADAPTIVE.value: _mk_adaptive,
        }

        maker = factory.get(strategy_key)
        if maker is None:
            self._log.warning(f"Unknown strategy {strategy_key}, using threshold")
            return _mk_threshold()
        return maker()

    def prepare_strategy_swap(
        self,
        strategy: SignalGenerationStrategy,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Prepare atomic strategy swap (off hot path).

        Parameters
        ----------
        strategy : SignalGenerationStrategy
            Strategy to swap to
        metadata : dict[str, Any] | None
            Optional metadata to associate with strategy

        """
        self._swapper.prepare_swap(strategy, metadata)

    def apply_pending_swap(self) -> bool:
        """
        Apply pending swap if ready (on hot path, O(1)).

        Returns
        -------
        bool
            True if swap was applied, False otherwise

        """
        return self._swapper.execute_swap()

    def get_current_strategy(self) -> SignalGenerationStrategy:
        """
        Get current strategy instance.

        Returns
        -------
        SignalGenerationStrategy
            Current strategy instance

        Raises
        ------
        RuntimeError
            If no strategy set

        """
        strategy = self._swapper.current_strategy
        if strategy is None:
            raise RuntimeError("No strategy set")
        return strategy

    @property
    def current_strategy(self) -> SignalGenerationStrategy:
        """
        Property accessor for current strategy.

        Returns
        -------
        SignalGenerationStrategy
            Current strategy instance

        Raises
        ------
        RuntimeError
            If no strategy set

        """
        return self.get_current_strategy()
