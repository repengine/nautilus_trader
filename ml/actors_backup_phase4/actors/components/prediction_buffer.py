"""
Prediction Buffer Component.

This module implements prediction history and ring buffer management for MLSignalActor
decomposition.

The component provides:
- Zero-allocation ring buffer management (hot path)
- Optional history list management (cold path)
- Ring metadata provider for strategy context
- Complete state reset functionality

Ring Buffers (Hot Path):
- Pre-allocated numpy arrays (fixed size, circular indexing)
- Zero allocations during update()
- Used by strategies for momentum/extremes calculations
- P99 <50μs per update()

History Lists (Cold Path):
- Python lists (append operations)
- Optional in optimized mode (enable_history=False)
- Used for diagnostics and cold path analysis

Performance:
- Hot path: <50μs per update()
- Memory: O(capacity) fixed, no growth
- Thread-safe: No (designed for single-threaded actor)

Architecture Patterns (CLAUDE.md):
- Pattern 3: Hot/Cold Path Separation (zero allocations in update())
- Pattern 2: Protocol-First Interface Design (property accessors)

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt


if TYPE_CHECKING:
    from logging import Logger


class PredictionBufferComponent:
    """
    Component for prediction history and ring buffer management.

    Manages prediction, confidence, and volatility buffers with zero-allocation
    hot path guarantees. Ring buffers use circular indexing with fixed capacity.
    History lists are optional (cold path only).

    Hot Path Requirements:
    - update() MUST use zero allocations
    - get_ring_metadata() MUST return zero-copy references
    - P99 <50μs per update()

    Cold Path:
    - get_history() allocations allowed (list slicing)
    - reset() allocations allowed (clear + fill)

    Example:
        >>> buffer = PredictionBufferComponent(capacity=100, enable_history=True)
        >>> buffer.update(prediction=0.75, confidence=0.9, volatility=0.01)
        >>> metadata = buffer.get_ring_metadata()
        >>> assert metadata["_prediction_ring_count"] == 1
        >>> history_pred, history_conf = buffer.get_history(lookback=10)

    """

    def __init__(
        self,
        capacity: int,
        enable_history: bool = True,
        actor_id: str | None = None,
        log: Logger | None = None,
    ) -> None:
        """
        Initialize prediction buffer component.

        Parameters
        ----------
        capacity : int
            The fixed capacity for ring buffers. Must be > 0.
        enable_history : bool, default=True
            Whether to maintain history lists (cold path).
            Set False in optimized mode to avoid allocations.
        actor_id : str | None, default=None
            Actor identifier for logging (optional).
        log : Logger | None, default=None
            Logger instance (optional).

        Raises
        ------
        ValueError
            If capacity <= 0.

        """
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")

        self._capacity: int = capacity
        self._enable_history: bool = enable_history
        self._actor_id: str | None = actor_id
        self._log: Logger | None = log

        # Ring buffers (hot path, pre-allocated, zero-copy)
        self._prediction_window: npt.NDArray[np.float32] = np.zeros(
            capacity,
            dtype=np.float32,
        )
        self._confidence_window: npt.NDArray[np.float32] = np.zeros(
            capacity,
            dtype=np.float32,
        )
        self._volatility_window: npt.NDArray[np.float32] = np.zeros(
            capacity,
            dtype=np.float32,
        )

        # Window indexing (hot path)
        self._window_index: int = 0
        self._window_count: int = 0

        # History lists (cold path, optional)
        self._prediction_history: list[float] = []
        self._confidence_history: list[float] = []

        if self._log:
            self._log.info(
                f"PredictionBufferComponent initialized (capacity={capacity}, "
                f"enable_history={enable_history}, actor_id={actor_id})",
            )

    def update(
        self,
        prediction: float,
        confidence: float,
        volatility: float = 0.0,
    ) -> None:
        """
        Update ring buffers and history with new prediction/confidence (HOT PATH).

        CRITICAL: Zero allocations - only writes to pre-allocated arrays.
        Performance target: <50μs P99.

        Parameters
        ----------
        prediction : float
            The model prediction value.
        confidence : float
            The confidence score (0.0 to 1.0).
        volatility : float, default=0.0
            The volatility value (e.g., price change).

        Notes
        -----
        - Ring buffer updates use circular indexing (modulo capacity)
        - Count saturates at capacity
        - History lists updated only if enable_history=True (cold path)
        - Zero allocations in ring buffer updates

        """
        # Write to current index (no allocation)
        idx = self._window_index
        self._prediction_window[idx] = np.float32(prediction)
        self._confidence_window[idx] = np.float32(confidence)
        self._volatility_window[idx] = np.float32(volatility)

        # Update index (circular, no allocation)
        self._window_index = (idx + 1) % self._capacity

        # Update count (saturate at capacity, no allocation)
        if self._window_count < self._capacity:
            self._window_count += 1

        # Optional: Update history lists (cold path, allocations OK)
        if self._enable_history:
            self._prediction_history.append(prediction)
            self._confidence_history.append(confidence)

    def get_ring_metadata(self) -> dict[str, Any]:
        """
        Get ring buffer metadata for strategy context (HOT PATH).

        Returns dictionary with zero-copy references to ring buffers.
        Strategies use this to access prediction history without allocations.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - _prediction_ring: Zero-copy reference to prediction window
            - _prediction_ring_index: Current write index
            - _prediction_ring_count: Number of valid samples
            - _confidence_ring: Zero-copy reference to confidence window
            - _volatility_ring: Zero-copy reference to volatility window

        Notes
        -----
        - Returns references, not copies (zero-copy)
        - Strategies should not modify returned arrays
        - Ring index points to next write position
        - Ring count is number of valid samples (min(updates, capacity))

        """
        return {
            "_prediction_ring": self._prediction_window,  # Zero-copy reference
            "_prediction_ring_index": self._window_index,
            "_prediction_ring_count": self._window_count,
            "_confidence_ring": self._confidence_window,  # Zero-copy reference
            "_volatility_ring": self._volatility_window,  # Zero-copy reference
        }

    def get_history(
        self,
        lookback: int | None = None,
    ) -> tuple[list[float], list[float]]:
        """
        Get prediction and confidence history (COLD PATH).

        Returns most recent N items if lookback specified, else all history.
        Allocations allowed (cold path).

        Parameters
        ----------
        lookback : int | None, default=None
            Number of recent items to return.
            If None, returns all history.
            If > len(history), returns all available.

        Returns
        -------
        tuple[list[float], list[float]]
            (prediction_history, confidence_history)
            Empty lists if enable_history=False.

        Notes
        -----
        - Returns copies, not references (safe for mutation)
        - Cold path operation (allocations allowed)
        - Returns empty lists if history disabled

        """
        if not self._enable_history:
            return [], []

        if lookback is None:
            return (
                self._prediction_history.copy(),
                self._confidence_history.copy(),
            )

        return (
            self._prediction_history[-lookback:],
            self._confidence_history[-lookback:],
        )

    def reset(self) -> None:
        """
        Reset all buffers and history (COLD PATH).

        Clears history lists, zeros ring buffers, resets counters.
        Called during actor reset/restart.

        Notes
        -----
        - Cold path operation (allocations allowed)
        - History lists cleared (if enabled)
        - Ring buffers zeroed (numpy operation)
        - Counters reset to 0

        """
        # Clear history (cold path, allocations OK)
        if self._enable_history:
            self._prediction_history.clear()
            self._confidence_history.clear()

        # Zero ring buffers (numpy operation, no per-element allocation)
        self._prediction_window.fill(0.0)
        self._confidence_window.fill(0.0)
        self._volatility_window.fill(0.0)

        # Reset counters
        self._window_index = 0
        self._window_count = 0

        if self._log:
            self._log.info(
                f"PredictionBufferComponent reset (capacity={self._capacity}, "
                f"actor_id={self._actor_id})",
            )

    @property
    def prediction_window(self) -> npt.NDArray[np.float32]:
        """
        Direct access to prediction ring buffer (zero-copy).

        Returns
        -------
        npt.NDArray[np.float32]
            Prediction ring buffer reference.

        Notes
        -----
        - Returns reference, not copy (zero-copy)
        - Caller should not modify array
        - Use get_ring_metadata() for strategy context

        """
        return self._prediction_window

    @property
    def confidence_window(self) -> npt.NDArray[np.float32]:
        """
        Direct access to confidence ring buffer (zero-copy).

        Returns
        -------
        npt.NDArray[np.float32]
            Confidence ring buffer reference.

        Notes
        -----
        - Returns reference, not copy (zero-copy)
        - Caller should not modify array

        """
        return self._confidence_window

    @property
    def volatility_window(self) -> npt.NDArray[np.float32]:
        """
        Direct access to volatility ring buffer (zero-copy).

        Returns
        -------
        npt.NDArray[np.float32]
            Volatility ring buffer reference.

        Notes
        -----
        - Returns reference, not copy (zero-copy)
        - Caller should not modify array

        """
        return self._volatility_window

    @property
    def window_index(self) -> int:
        """
        Current write index in ring buffer.

        Returns
        -------
        int
            Index where next update() will write (range: [0, capacity)).

        """
        return self._window_index

    @property
    def window_count(self) -> int:
        """
        Number of valid samples in ring buffer.

        Returns
        -------
        int
            Number of samples written (range: [0, capacity]).

        Notes
        -----
        - Count saturates at capacity
        - Count < capacity during warm-up phase
        - Count == capacity after capacity updates

        """
        return self._window_count

    @property
    def capacity(self) -> int:
        """
        Fixed capacity of ring buffers.

        Returns
        -------
        int
            Ring buffer capacity.

        """
        return self._capacity

    @property
    def enable_history(self) -> bool:
        """
        Whether history lists are enabled.

        Returns
        -------
        bool
            True if history lists maintained, False otherwise.

        """
        return self._enable_history
