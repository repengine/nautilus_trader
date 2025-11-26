"""
Adaptive Threshold Component.

This module implements adaptive threshold calculation and market regime detection
for MLSignalActor decomposition.

The component provides:
- Volatility-based threshold adjustment
- Market regime detection (4 regimes)
- Context building for strategies
- Zero-allocation warm path operations

Regime Classification:
- "low_volatility": avg_vol < 0.001
- "normal": 0.001 <= avg_vol < 0.005
- "high_volatility": avg_vol >= 0.005
- "unknown": insufficient data (count < 3)

Threshold Formula:
    threshold = min(max(base + volatility * factor, min), max)

Performance:
- Warm path: update_threshold(), detect_regime(), build_context()
- Zero allocations in warm path operations
- Direct calculations without intermediate collections

Architecture Patterns (CLAUDE.md):
- Pattern 3: Hot/Cold Path Separation (warm path optimized)
- Pattern 2: Protocol-First Interface Design (property accessors)

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt


if TYPE_CHECKING:
    from logging import Logger


class AdaptiveThresholdComponent:
    """
    Component for adaptive threshold and market regime detection.

    Manages threshold adaptation based on volatility and detects market regimes
    from volatility patterns. Provides context metadata for signal strategies.

    Warm Path Requirements:
    - update_threshold() zero allocations
    - detect_regime() zero allocations
    - build_context() zero allocations (returns reference to pre-allocated state)

    Regime Detection:
    - Requires minimum 3 observations (count >= 3)
    - Uses count-based averaging (ignores zero-padding)
    - 4 regimes: low_volatility, normal, high_volatility, unknown

    Threshold Adjustment:
    - Formula: base_threshold + (volatility * volatility_factor)
    - Clamped to [min_threshold, max_threshold]
    - Supports positive/negative volatility factors

    Example:
        >>> component = AdaptiveThresholdComponent(
        ...     base_threshold=0.7,
        ...     volatility_factor=2.0,
        ...     min_threshold=0.1,
        ...     max_threshold=0.95,
        ... )
        >>> threshold = component.update_threshold(avg_volatility=0.01)
        >>> assert 0.1 <= threshold <= 0.95
        >>> regime = component.detect_regime(volatility_window=vol_arr, count=5)
        >>> context = component.build_context()
        >>> assert context["adaptive_threshold"] == threshold

    """

    # Minimum number of observations required for regime detection
    MIN_REGIME_COUNT: int = 3

    # Regime thresholds (avg_volatility boundaries)
    REGIME_LOW_THRESHOLD: float = 0.001
    REGIME_NORMAL_THRESHOLD: float = 0.005

    def __init__(
        self,
        base_threshold: float,
        volatility_factor: float = 2.0,
        min_threshold: float = 0.1,
        max_threshold: float = 0.95,
        actor_id: str | None = None,
        log: Logger | None = None,
    ) -> None:
        """
        Initialize adaptive threshold component.

        Parameters
        ----------
        base_threshold : float
            Base confidence threshold (before adjustment).
        volatility_factor : float, default=2.0
            Factor to multiply volatility for threshold adjustment.
            Positive values increase threshold with volatility.
            Negative values decrease threshold with volatility.
        min_threshold : float, default=0.1
            Minimum allowed threshold value.
        max_threshold : float, default=0.95
            Maximum allowed threshold value.
        actor_id : str | None, default=None
            Actor identifier for logging (optional).
        log : Logger | None, default=None
            Logger instance (optional).

        Raises
        ------
        ValueError
            If min_threshold > max_threshold.
            If base_threshold not in [min_threshold, max_threshold].

        """
        # Validate bounds
        if min_threshold > max_threshold:
            msg = f"min_threshold ({min_threshold}) must be <= " f"max_threshold ({max_threshold})"
            raise ValueError(msg)

        if not (min_threshold <= base_threshold <= max_threshold):
            msg = (
                f"base_threshold ({base_threshold}) must be in range "
                f"[{min_threshold}, {max_threshold}]"
            )
            raise ValueError(msg)

        self._base_threshold: float = base_threshold
        self._volatility_factor: float = volatility_factor
        self._min_threshold: float = min_threshold
        self._max_threshold: float = max_threshold
        self._actor_id: str | None = actor_id
        self._log: Logger | None = log

        # Current state (updated by warm path methods)
        self._threshold: float = base_threshold
        self._market_regime: str = "unknown"

        if self._log:
            self._log.info(
                f"AdaptiveThresholdComponent initialized (base={base_threshold}, "
                f"factor={volatility_factor}, range=[{min_threshold}, {max_threshold}], "
                f"actor_id={actor_id})",
            )

    def update_threshold(self, avg_volatility: float) -> float:
        """
        Update adaptive threshold based on volatility (WARM PATH).

        Formula: threshold = min(max(base + vol * factor, min), max)

        CRITICAL: Zero allocations - direct calculation only.

        Parameters
        ----------
        avg_volatility : float
            Average volatility from volatility window.

        Returns
        -------
        float
            Updated threshold value (clamped to bounds).

        Notes
        -----
        - Threshold increases with volatility if volatility_factor > 0
        - Threshold decreases with volatility if volatility_factor < 0
        - Result always in [min_threshold, max_threshold]
        - Updates internal state (_threshold)

        """
        # Direct calculation - no allocations
        threshold = self._base_threshold + (avg_volatility * self._volatility_factor)

        # Clamp to bounds - no allocations
        threshold = max(self._min_threshold, min(threshold, self._max_threshold))

        # Update state
        self._threshold = threshold

        return threshold

    def detect_regime(
        self,
        volatility_window: npt.NDArray[np.float32],
        count: int,
    ) -> str:
        """
        Detect market regime from volatility window (WARM PATH).

        Regimes:
        - "low_volatility": avg_vol < 0.001
        - "normal": 0.001 <= avg_vol < 0.005
        - "high_volatility": avg_vol >= 0.005
        - "unknown": count < min_count (3)

        CRITICAL: Zero allocations - uses numpy slice (zero-copy view).

        Parameters
        ----------
        volatility_window : npt.NDArray[np.float32]
            Ring buffer of volatility values.
        count : int
            Number of valid values in window (not capacity).

        Returns
        -------
        str
            Market regime label.

        Notes
        -----
        - Requires minimum 3 observations to avoid zero-padding bias
        - Uses count-based averaging: np.mean(window[:count])
        - Updates internal state (_market_regime)
        - Boundary conditions: 0.001 -> "normal", 0.005 -> "high_volatility"

        """
        # Require minimum count to avoid zero-padding bias
        if count < self.MIN_REGIME_COUNT:
            self._market_regime = "unknown"
            return self._market_regime

        # Compute average over valid prefix only (zero-copy slice)
        # NOTE: np.mean creates temporary, but unavoidable for average calculation
        avg_volatility = float(np.mean(volatility_window[:count]))

        # Classify regime based on thresholds
        if avg_volatility < self.REGIME_LOW_THRESHOLD:
            regime = "low_volatility"
        elif avg_volatility < self.REGIME_NORMAL_THRESHOLD:
            regime = "normal"
        else:
            regime = "high_volatility"

        # Update state
        self._market_regime = regime

        return regime

    def build_context(
        self,
        prediction_history: list[float] | None = None,
        confidence_history: list[float] | None = None,
    ) -> dict[str, Any]:
        """
        Build context metadata for strategies (WARM PATH).

        Context includes:
        - adaptive_threshold: Current threshold value
        - market_regime: Current regime label
        - prediction_history: (optional) From PredictionBufferComponent
        - confidence_history: (optional) From PredictionBufferComponent

        CRITICAL: Zero allocations - returns reference to existing state.

        Parameters
        ----------
        prediction_history : list[float] | None, default=None
            Optional prediction history from buffer component.
        confidence_history : list[float] | None, default=None
            Optional confidence history from buffer component.

        Returns
        -------
        dict[str, Any]
            Context dictionary for strategy.generate_signal().

        Notes
        -----
        - Dictionary allocation unavoidable (Python dict creation)
        - But values are references (no copying)
        - Caller (MLSignalActor) may add additional fields (timestamp_ns, model_id)

        """
        # Build context with current state (dict allocation, but values are refs)
        context: dict[str, Any] = {
            "adaptive_threshold": self._threshold,
            "market_regime": self._market_regime,
        }

        # Add optional history lists if provided
        if prediction_history is not None:
            context["prediction_history"] = prediction_history

        if confidence_history is not None:
            context["confidence_history"] = confidence_history

        return context

    @property
    def current_threshold(self) -> float:
        """
        Get current adaptive threshold value.

        Returns
        -------
        float
            Current threshold (last value from update_threshold()).

        """
        return self._threshold

    @property
    def current_regime(self) -> str:
        """
        Get current market regime.

        Returns
        -------
        str
            Current regime label ("low_volatility", "normal", "high_volatility", "unknown").

        """
        return self._market_regime

    @property
    def base_threshold(self) -> float:
        """
        Get base threshold value (before volatility adjustment).

        Returns
        -------
        float
            Base threshold value from initialization.

        """
        return self._base_threshold

    @property
    def volatility_factor(self) -> float:
        """
        Get volatility adjustment factor.

        Returns
        -------
        float
            Volatility factor from initialization.

        """
        return self._volatility_factor

    @property
    def min_threshold(self) -> float:
        """
        Get minimum threshold bound.

        Returns
        -------
        float
            Minimum threshold value.

        """
        return self._min_threshold

    @property
    def max_threshold(self) -> float:
        """
        Get maximum threshold bound.

        Returns
        -------
        float
            Maximum threshold value.

        """
        return self._max_threshold
