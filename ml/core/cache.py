"""
Optimized feature caching and ring buffer implementations for hot path performance.

This module provides lock-free ring buffers and reservoir sampling for maintaining
prediction history and computing percentiles with zero allocations in the hot path.

"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = [
    "RingBufferProtocol",
    "FeatureCacheProtocol",
    "SamplerProtocol",
    "LockFreeRingBuffer",
    "ReservoirSampler",
    "PreAllocatedFeatureCache",
]


@runtime_checkable
class RingBufferProtocol(Protocol):
    @property
    def count(self) -> int: ...

    def append(self, value: float) -> None: ...

    def get_last(self, n: int = 1) -> npt.NDArray[np.float64]: ...

    def get_window(self, start: int, length: int) -> npt.NDArray[np.float64]: ...

    def reset(self) -> None: ...


@runtime_checkable
class FeatureCacheProtocol(Protocol):
    @property
    def n_features(self) -> int: ...

    def get_current_buffer(self) -> npt.NDArray[np.float64]: ...

    def store_current_features(self, features: npt.NDArray[np.float64]) -> None: ...

    def prepare_onnx_input(self, use_normalized: bool = True) -> npt.NDArray[np.float64]: ...

    def reset(self) -> None: ...


@runtime_checkable
class SamplerProtocol(Protocol):
    @property
    def count(self) -> int: ...

    def add_sample(self, value: float) -> None: ...

    def get_percentile(self, q: float) -> float: ...

    def reset(self) -> None: ...


class LockFreeRingBuffer(RingBufferProtocol):
    """
    Lock-free ring buffer for high-performance history tracking.

    This implementation provides O(1) append operations and efficient windowed
    access patterns without any memory allocations in the hot path.

    Parameters
    ----------
    size : int
        Maximum number of elements in the buffer.
    dtype : np.dtype, default np.float32
        NumPy data type for buffer elements.

    """

    def __init__(self, size: int, dtype: type[np.floating[Any]] = np.float32) -> None:
        if size <= 0:
            msg = f"Buffer size must be positive, got {size}"
            raise ValueError(msg)

        self._buffer = np.empty(size, dtype=dtype)
        self._size = size
        self._index = 0
        self._count = 0
        self._dtype = dtype
        self._random = random.Random()

    @property
    def size(self) -> int:
        """
        Return maximum buffer size.
        """
        return self._size

    @property
    def count(self) -> int:
        """
        Return current number of elements.
        """
        return self._count

    @property
    def is_full(self) -> bool:
        """
        Return True if buffer is at capacity.
        """
        return self._count == self._size

    def append(self, value: float) -> None:
        """
        Add value to ring buffer (overwrites oldest if full).

        This is a zero-allocation O(1) operation optimized for hot path usage.

        Parameters
        ----------
        value : float
            Value to append to the buffer.

        """
        self._buffer[self._index] = value
        self._index = (self._index + 1) % self._size
        self._count = min(self._count + 1, self._size)

    def append_array(self, values: npt.NDArray[np.float64]) -> None:
        """
        Append multiple values efficiently.

        Parameters
        ----------
        values : npt.NDArray[np.float64]
            Array of values to append.

        """
        for value in values:
            self.append(float(value))

    def get_last(self, n: int = 1) -> npt.NDArray[np.float64]:
        """
        Get last n values as NumPy array view (zero-copy when possible).

        Parameters
        ----------
        n : int, default 1
            Number of recent values to retrieve.

        Returns
        -------
        npt.NDArray[np.float64]
            Array view containing the last n values (no copy when contiguous).

        """
        if n <= 0:
            return np.array([], dtype=self._dtype)

        if n > self._count:
            n = self._count

        if n == 0:
            return np.array([], dtype=self._dtype)

        if self._count < self._size:
            # Buffer not yet full, return view
            start = max(0, self._count - n)
            return self._buffer[start : self._count]  # Return view, not copy

        # Buffer is full, handle wrap-around
        start_idx = (self._index - n) % self._size
        if start_idx + n <= self._size:
            # No wrap-around, return view
            return self._buffer[start_idx : start_idx + n]  # Return view, not copy

        # Handle wrap-around - must concatenate (allocation unavoidable here)
        # This only happens when crossing buffer boundary
        first_part = self._buffer[start_idx:]
        second_part = self._buffer[: (start_idx + n) % self._size]
        return np.concatenate([first_part, second_part])

    def get_window(self, start: int, length: int) -> npt.NDArray[np.float64]:
        """
        Get a window of values starting from relative position.

        Parameters
        ----------
        start : int
            Starting position (0 = oldest, negative = from end).
        length : int
            Number of values to retrieve.

        Returns
        -------
        npt.NDArray[np.float64]
            Array view containing the requested window (no copy when contiguous).

        """
        if self._count == 0 or length <= 0:
            return np.array([], dtype=self._dtype)

        # Convert negative indices
        if start < 0:
            start = max(0, self._count + start)

        if start >= self._count:
            return np.array([], dtype=self._dtype)

        # Clamp length to available data
        length = min(length, self._count - start)

        if self._count < self._size:
            # Buffer not yet full, return view
            return self._buffer[start : start + length]  # Return view, not copy

        # Buffer is full, calculate actual indices
        actual_start = (self._index - self._count + start) % self._size
        if actual_start + length <= self._size:
            return self._buffer[actual_start : actual_start + length]  # Return view, not copy

        # Handle wrap-around - must concatenate (allocation unavoidable here)
        # This only happens when crossing buffer boundary
        first_part = self._buffer[actual_start:]
        remaining = length - (self._size - actual_start)
        second_part = self._buffer[:remaining]
        return np.concatenate([first_part, second_part])

    def get_all(self) -> npt.NDArray[np.float64]:
        """
        Get all values in chronological order.

        Returns
        -------
        npt.NDArray[np.float64]
            Array containing all values from oldest to newest.

        """
        return self.get_last(self._count)

    def reset(self) -> None:
        """
        Reset buffer to empty state.
        """
        self._index = 0
        self._count = 0
        # Don't reallocate buffer, just reset pointers

    def mean(self) -> float:
        """
        Calculate mean of current values.
        """
        if self._count == 0:
            return 0.0
        return float(np.mean(self.get_all()))

    def std(self) -> float:
        """
        Calculate standard deviation of current values.
        """
        if self._count <= 1:
            return 0.0
        return float(np.std(self.get_all()))

    def percentile(self, q: float) -> float:
        """
        Calculate percentile of current values.

        Parameters
        ----------
        q : float
            Percentile to compute (0-100).

        Returns
        -------
        float
            The q-th percentile value.

        """
        if self._count == 0:
            return 0.0
        return float(np.percentile(self.get_all(), q))


class ReservoirSampler(SamplerProtocol):
    """
    Reservoir sampling for maintaining representative sample for percentile calculation.

    This implementation uses Reservoir Sampling Algorithm R to maintain a uniform
    random sample from a stream of values, enabling efficient percentile calculation
    without storing all historical data.

    Parameters
    ----------
    reservoir_size : int
        Size of the reservoir (sample size).
    dtype : np.dtype, default np.float32
        NumPy data type for stored values.

    """

    def __init__(self, reservoir_size: int, dtype: type[np.floating[Any]] = np.float32) -> None:
        if reservoir_size <= 0:
            msg = f"Reservoir size must be positive, got {reservoir_size}"
            raise ValueError(msg)

        self._reservoir = np.empty(reservoir_size, dtype=dtype)
        self._reservoir_size = reservoir_size
        self._count = 0
        self._total_seen = 0
        self._dtype = dtype

    @property
    def reservoir_size(self) -> int:
        """
        Return reservoir size.
        """
        return self._reservoir_size

    @property
    def count(self) -> int:
        """
        Return current number of samples in reservoir.
        """
        return self._count

    @property
    def total_seen(self) -> int:
        """
        Return total number of values processed.
        """
        return self._total_seen

    def add_sample(self, value: float) -> None:
        """
        Add a new sample using reservoir sampling algorithm.

        Parameters
        ----------
        value : float
            Value to potentially add to reservoir.

        """
        self._total_seen += 1

        if self._count < self._reservoir_size:
            # Fill reservoir
            self._reservoir[self._count] = value
            self._count += 1
        else:
            # Reservoir full, randomly replace
            j = random.randint(0, self._total_seen - 1)
            if j < self._reservoir_size:
                self._reservoir[j] = value

    def add_samples(self, values: Sequence[float]) -> None:
        """
        Add multiple samples efficiently.

        Parameters
        ----------
        values : Sequence[float]
            Sequence of values to add.

        """
        for value in values:
            self.add_sample(value)

    def get_percentile(self, q: float) -> float:
        """
        Calculate percentile from current reservoir sample.

        Parameters
        ----------
        q : float
            Percentile to compute (0-100).

        Returns
        -------
        float
            The q-th percentile value.

        """
        if self._count == 0:
            return 0.0

        sample = self._reservoir[: self._count]
        return float(np.percentile(sample, q))

    def get_percentiles(self, percentiles: Sequence[float]) -> dict[float, float]:
        """
        Calculate multiple percentiles efficiently.

        Parameters
        ----------
        percentiles : Sequence[float]
            List of percentiles to compute (0-100).

        Returns
        -------
        dict[float, float]
            Dictionary mapping percentile to value.

        """
        if self._count == 0:
            return dict.fromkeys(percentiles, 0.0)

        sample = self._reservoir[: self._count]
        return {p: float(np.percentile(sample, p)) for p in percentiles}

    def get_sample(self) -> npt.NDArray[np.float32]:
        """
        Get current reservoir sample.

        Returns
        -------
        npt.NDArray[np.float64]
            Array view containing current reservoir sample.

        """
        return self._reservoir[: self._count]  # Return view, not copy

    def reset(self) -> None:
        """
        Reset reservoir to empty state.
        """
        self._count = 0
        self._total_seen = 0


class PreAllocatedFeatureCache(FeatureCacheProtocol):
    """
    Pre-allocated cache for feature vectors with zero-allocation hot path operations.

    This cache maintains pre-allocated buffers for feature computation and provides
    memoryview access for zero-copy operations where possible.

    Parameters
    ----------
    n_features : int
        Number of features per vector.
    history_size : int, default 1000
        Size of history buffer for feature vectors.
    dtype : np.dtype, default np.float32
        NumPy data type for feature data.

    """

    def __init__(
        self,
        n_features: int,
        history_size: int = 1000,
        dtype: type[np.floating[Any]] = np.float32,
    ) -> None:
        if n_features <= 0:
            msg = f"Number of features must be positive, got {n_features}"
            raise ValueError(msg)
        if history_size <= 0:
            msg = f"History size must be positive, got {history_size}"
            raise ValueError(msg)

        self._n_features = n_features
        self._dtype = dtype

        # Pre-allocate all buffers
        self._current_features = np.zeros(n_features, dtype=dtype)
        self._normalized_features = np.zeros(n_features, dtype=dtype)
        self._feature_history = np.zeros((history_size, n_features), dtype=dtype)

        # ONNX input buffer (batch size 1)
        self._onnx_input_buffer = np.zeros((1, n_features), dtype=dtype)

        # Ring buffer for managing history
        self._history_index = 0
        self._history_count = 0
        self._history_size = history_size

        # Memory views for zero-copy access
        self._current_features_view = memoryview(self._current_features.data)
        self._normalized_features_view = memoryview(self._normalized_features.data)
        self._onnx_input_view = memoryview(self._onnx_input_buffer.data)

    @property
    def n_features(self) -> int:
        """
        Return number of features.
        """
        return self._n_features

    @property
    def history_size(self) -> int:
        """
        Return maximum history size.
        """
        return self._history_size

    @property
    def history_count(self) -> int:
        """
        Return current number of stored feature vectors.
        """
        return self._history_count

    def get_current_buffer(self) -> npt.NDArray[np.float32]:
        """
        Get the current feature buffer for in-place computation.

        Returns
        -------
        npt.NDArray[np.float64]
            Pre-allocated buffer for current features.

        """
        return self._current_features

    def get_current_view(self) -> memoryview:
        """
        Get memoryview of current feature buffer for zero-copy access.

        Returns
        -------
        memoryview
            Memory view of current feature buffer.

        """
        return self._current_features_view

    def get_normalized_buffer(self) -> npt.NDArray[np.float32]:
        """
        Get the normalized feature buffer for in-place computation.

        Returns
        -------
        npt.NDArray[np.float64]
            Pre-allocated buffer for normalized features.

        """
        return self._normalized_features

    def get_normalized_view(self) -> memoryview:
        """
        Get memoryview of normalized feature buffer for zero-copy access.

        Returns
        -------
        memoryview
            Memory view of normalized feature buffer.

        """
        return self._normalized_features_view

    def get_onnx_input_buffer(self) -> npt.NDArray[np.float32]:
        """
        Get ONNX input buffer (shape: [1, n_features]) for inference.

        Returns
        -------
        npt.NDArray[np.float64]
            Pre-allocated ONNX input buffer.

        """
        return self._onnx_input_buffer

    def prepare_onnx_input(self, use_normalized: bool = True) -> npt.NDArray[np.float32]:
        """
        Prepare ONNX input buffer with current features.

        Parameters
        ----------
        use_normalized : bool, default True
            Whether to use normalized features.

        Returns
        -------
        npt.NDArray[np.float64]
            ONNX input buffer ready for inference.

        """
        source = self._normalized_features if use_normalized else self._current_features
        self._onnx_input_buffer[0] = source
        return self._onnx_input_buffer

    def store_current_features(self) -> None:
        """
        Store current features in history using ring buffer.

        This operation is zero-allocation and operates in-place on pre-allocated
        buffers.

        """
        # Copy current features to history
        self._feature_history[self._history_index] = self._current_features

        # Update ring buffer pointers
        self._history_index = (self._history_index + 1) % self._history_size
        self._history_count = min(self._history_count + 1, self._history_size)

    def get_feature_history(self, n_latest: int | None = None) -> npt.NDArray[np.float32]:
        """
        Get feature history in chronological order.

        Parameters
        ----------
        n_latest : int, optional
            Number of latest feature vectors to return. If None, returns all.

        Returns
        -------
        npt.NDArray[np.float64]
            Feature history array view with shape [n_vectors, n_features].

        """
        if self._history_count == 0:
            return np.array([], dtype=self._dtype).reshape(0, self._n_features)

        if n_latest is None:
            n_latest = self._history_count
        else:
            n_latest = min(n_latest, self._history_count)

        if n_latest <= 0:
            return np.array([], dtype=self._dtype).reshape(0, self._n_features)

        # Get indices in chronological order
        if self._history_count < self._history_size:
            # History buffer not yet full, return view
            start_idx = max(0, self._history_count - n_latest)
            return self._feature_history[start_idx : self._history_count]  # Return view

        # History buffer is full, handle wrap-around
        start_idx = (self._history_index - n_latest) % self._history_size
        if start_idx + n_latest <= self._history_size:
            # No wrap-around, return view
            return self._feature_history[start_idx : start_idx + n_latest]  # Return view

        # Handle wrap-around - must concatenate (allocation unavoidable here)
        # This only happens when crossing buffer boundary
        first_part = self._feature_history[start_idx:]
        second_part = self._feature_history[: (start_idx + n_latest) % self._history_size]
        return np.vstack([first_part, second_part])

    def reset(self) -> None:
        """
        Reset all buffers and history.
        """
        self._current_features.fill(0.0)
        self._normalized_features.fill(0.0)
        self._onnx_input_buffer.fill(0.0)
        self._feature_history.fill(0.0)
        self._history_index = 0
        self._history_count = 0


class MultiChannelRingBuffer:
    """
    Lock-free multi-channel ring buffer for high-frequency metrics.

    Stores multiple channels in a fixed-capacity circular buffer with O(1) append and
    no allocations on the hot path. Cold-path methods can materialize chronological
    arrays for statistics.

    Parameters
    ----------
    size : int
        Number of rows (ring capacity).
    channels : int
        Number of parallel channels (columns) to store.
    dtype : np.dtype, default np.float32
        NumPy dtype for stored values.

    """

    def __init__(
        self,
        size: int,
        channels: int,
        dtype: type[np.floating[Any]] = np.float32,
    ) -> None:
        if size <= 0:
            raise ValueError(f"Buffer size must be positive, got {size}")
        if channels <= 0:
            raise ValueError(f"Channels must be positive, got {channels}")

        self._cap = int(size)
        self._channels = int(channels)
        self._dtype = dtype
        self._buf = np.zeros((self._cap, self._channels), dtype=dtype)
        self._idx = 0
        self._count = 0

    @property
    def capacity(self) -> int:
        """
        Return ring capacity (rows).
        """
        return self._cap

    @property
    def channels(self) -> int:
        """
        Return number of channels.
        """
        return self._channels

    @property
    def index(self) -> int:
        """
        Return the next write index (0..capacity-1).
        """
        return self._idx

    @property
    def count(self) -> int:
        """
        Return number of valid rows (<= capacity).
        """
        return self._count

    def append(self, values: Sequence[float]) -> None:
        """
        Append one row of channel values in-place (no allocations).
        """
        if len(values) != self._channels:
            raise ValueError(f"expected {self._channels} values, got {len(values)}")

        i = self._idx
        self._buf[i, :] = values
        i += 1
        if i >= self._cap:
            i = 0
        self._idx = i
        if self._count < self._cap:
            self._count += 1

    def get_last_row(self) -> npt.NDArray[np.float32]:
        """
        Return a view of the most recently written row, or empty array if none.
        """
        if self._count == 0:
            return np.array([], dtype=self._dtype)
        last = (self._idx - 1) % self._cap
        return self._buf[last, :]

    def get_channel_view(self, channel: int) -> npt.NDArray[np.float32]:
        """
        Return direct view of a channel in ring order (interpret with index/count).
        """
        if channel < 0 or channel >= self._channels:
            raise IndexError(f"channel out of range: {channel}")
        return self._buf[:, channel]

    def get_channel_chronological(self, channel: int) -> npt.NDArray[np.float32]:
        """
        Return channel data in chronological order (allocates on wrap-around).
        """
        if self._count == 0:
            return np.array([], dtype=self._dtype)
        if channel < 0 or channel >= self._channels:
            raise IndexError(f"channel out of range: {channel}")

        n = self._count
        if n < self._cap:
            return self._buf[:n, channel]

        start = self._idx
        if start == 0:
            return self._buf[:, channel]

        first_part = self._buf[start:, channel]
        second_part = self._buf[:start, channel]
        return np.concatenate([first_part, second_part])

    def reset(self) -> None:
        """
        Reset ring to empty state.
        """
        self._idx = 0
        self._count = 0
