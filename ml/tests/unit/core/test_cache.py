"""
Unit tests for ML core cache functionality.

Tests focus on observable cache behavior rather than implementation details, following
the testing protocol principles.

"""

import numpy as np
import pytest

from ml.core.cache import LockFreeRingBuffer
from ml.core.cache import PreAllocatedFeatureCache
from ml.core.cache import ReservoirSampler


@pytest.mark.parallel_safe
@pytest.mark.unit
class TestLockFreeRingBuffer:
    """
    Test suite for LockFreeRingBuffer behavior.
    """

    def test_init_with_valid_size(self) -> None:
        """
        Test buffer initialization with valid size.
        """
        buffer = LockFreeRingBuffer(size=10)
        assert buffer.size == 10
        assert buffer.count == 0
        assert not buffer.is_full

    def test_init_with_invalid_size_raises(self) -> None:
        """
        Test buffer initialization with invalid size raises error.
        """
        with pytest.raises(ValueError, match="Buffer size must be positive"):
            LockFreeRingBuffer(size=0)

        with pytest.raises(ValueError, match="Buffer size must be positive"):
            LockFreeRingBuffer(size=-1)

    def test_append_single_value(self) -> None:
        """
        Test appending single values to buffer.
        """
        buffer = LockFreeRingBuffer(size=5)

        buffer.append(1.0)
        assert buffer.count == 1
        assert not buffer.is_full

        buffer.append(2.0)
        assert buffer.count == 2

    def test_append_fills_buffer(self) -> None:
        """
        Test buffer correctly reports when full.
        """
        buffer = LockFreeRingBuffer(size=3)

        buffer.append(1.0)
        buffer.append(2.0)
        assert not buffer.is_full

        buffer.append(3.0)
        assert buffer.is_full
        assert buffer.count == 3

    def test_append_overwrites_oldest(self) -> None:
        """
        Test ring buffer overwrites oldest values when full.
        """
        buffer = LockFreeRingBuffer(size=3)

        # Fill buffer
        for i in range(3):
            buffer.append(float(i))

        # Overwrite oldest
        buffer.append(3.0)
        assert buffer.count == 3  # Count stays at max

        # Check that oldest value was overwritten
        last_values = buffer.get_last(3)
        np.testing.assert_array_equal(last_values, [1.0, 2.0, 3.0])

    def test_append_array(self) -> None:
        """
        Test appending multiple values at once.
        """
        buffer = LockFreeRingBuffer(size=5)
        values = np.array([1.0, 2.0, 3.0])

        buffer.append_array(values)
        assert buffer.count == 3

        result = buffer.get_last(3)
        np.testing.assert_array_equal(result, values)

    def test_get_last_with_various_sizes(self) -> None:
        """
        Test retrieving last n values with different scenarios.
        """
        buffer = LockFreeRingBuffer(size=5)

        # Empty buffer
        result = buffer.get_last(3)
        assert len(result) == 0

        # Partially filled buffer
        buffer.append(1.0)
        buffer.append(2.0)
        result = buffer.get_last(3)
        np.testing.assert_array_equal(result, [1.0, 2.0])

        # Request more than available
        result = buffer.get_last(10)
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_get_last_with_wraparound(self) -> None:
        """
        Test retrieving values when buffer has wrapped around.
        """
        buffer = LockFreeRingBuffer(size=3)

        # Fill and wrap buffer
        for i in range(5):
            buffer.append(float(i))

        # Should get [2, 3, 4] (oldest values overwritten)
        result = buffer.get_last(3)
        np.testing.assert_array_equal(result, [2.0, 3.0, 4.0])

    def test_get_window(self) -> None:
        """
        Test retrieving windowed data from buffer.
        """
        buffer = LockFreeRingBuffer(size=10)

        # Add values
        for i in range(7):
            buffer.append(float(i))

        # Get window from start
        result = buffer.get_window(0, 3)
        np.testing.assert_array_equal(result, [0.0, 1.0, 2.0])

        # Get window from middle
        result = buffer.get_window(2, 3)
        np.testing.assert_array_equal(result, [2.0, 3.0, 4.0])

        # Get window with negative start (from end)
        result = buffer.get_window(-3, 2)
        np.testing.assert_array_equal(result, [4.0, 5.0])

    def test_get_window_edge_cases(self) -> None:
        """
        Test window retrieval edge cases.
        """
        buffer = LockFreeRingBuffer(size=5)

        # Empty buffer
        result = buffer.get_window(0, 3)
        assert len(result) == 0

        # Add some values
        for i in range(3):
            buffer.append(float(i))

        # Start beyond count
        result = buffer.get_window(10, 2)
        assert len(result) == 0

        # Zero length
        result = buffer.get_window(0, 0)
        assert len(result) == 0

        # Negative length (should return empty)
        result = buffer.get_window(0, -1)
        assert len(result) == 0

    def test_get_all_and_reset(self) -> None:
        """
        Test getting all values and resetting buffer.
        """
        buffer = LockFreeRingBuffer(size=5)

        # Add values
        for i in range(3):
            buffer.append(float(i))

        # Get all values
        array = buffer.get_all()
        np.testing.assert_array_equal(array, [0.0, 1.0, 2.0])

        # Reset buffer
        buffer.reset()
        assert buffer.count == 0
        assert not buffer.is_full

    def test_percentile_calculation(self) -> None:
        """
        Test percentile calculations on buffer data.
        """
        buffer = LockFreeRingBuffer(size=100)

        # Add values
        np.random.seed(42)  # For reproducibility
        values = np.random.randn(50)
        for v in values:
            buffer.append(v)

        # Get percentiles
        p50 = buffer.percentile(50)
        p95 = buffer.percentile(95)

        # Verify against numpy
        expected_p50 = np.percentile(values, 50)
        expected_p95 = np.percentile(values, 95)

        np.testing.assert_allclose(p50, expected_p50, rtol=1e-7)
        np.testing.assert_allclose(p95, expected_p95, rtol=1e-7)

    def test_mean_and_std(self) -> None:
        """
        Test mean and standard deviation calculations.
        """
        buffer = LockFreeRingBuffer(size=10)

        # Add known values
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            buffer.append(v)

        # Check mean
        assert buffer.mean() == 3.0

        # Check std
        expected_std = np.std(values)
        np.testing.assert_allclose(buffer.std(), expected_std, rtol=1e-7)


class TestReservoirSampler:
    """
    Test suite for ReservoirSampler behavior.
    """

    def test_init_with_valid_size(self) -> None:
        """
        Test sampler initialization.
        """
        sampler = ReservoirSampler(reservoir_size=10)
        assert sampler.reservoir_size == 10
        assert sampler.count == 0

    def test_init_with_invalid_size_raises(self) -> None:
        """
        Test sampler initialization with invalid size.
        """
        with pytest.raises(ValueError, match="Reservoir size must be positive"):
            ReservoirSampler(reservoir_size=0)

    def test_add_samples_up_to_size(self) -> None:
        """
        Test adding samples up to reservoir size.
        """
        sampler = ReservoirSampler(reservoir_size=5)

        for i in range(5):
            sampler.add_sample(float(i))

        assert sampler.count == 5

        # Get sample array
        samples = sampler.get_sample()
        assert len(samples) == 5
        # All values should be present when not exceeding size
        assert set(samples) == {0.0, 1.0, 2.0, 3.0, 4.0}

    def test_reservoir_sampling_maintains_size(self) -> None:
        """
        Test reservoir maintains fixed size after filling.
        """
        np.random.seed(42)  # Set seed for reproducibility
        sampler = ReservoirSampler(reservoir_size=5)

        # Add more values than reservoir size
        for i in range(20):
            sampler.add_sample(float(i))

        assert sampler.total_seen == 20  # Total count tracked

        # Get samples
        samples = sampler.get_sample()
        assert len(samples) == 5  # But only keeps reservoir size

    def test_add_multiple_samples(self) -> None:
        """
        Test adding multiple samples at once.
        """
        np.random.seed(42)
        sampler = ReservoirSampler(reservoir_size=10)

        samples = np.arange(5, dtype=np.float64)
        sampler.add_samples(samples.tolist())

        assert sampler.count == 5
        assert sampler.total_seen == 5

    def test_get_percentile(self) -> None:
        """
        Test percentile calculation from reservoir.
        """
        np.random.seed(42)
        sampler = ReservoirSampler(reservoir_size=100)

        # Add values from known distribution
        values = np.arange(100, dtype=np.float64)
        sampler.add_samples(values.tolist())

        # Since reservoir size equals data size, should be exact
        p50 = sampler.get_percentile(50)
        p90 = sampler.get_percentile(90)

        np.testing.assert_allclose(p50, 49.5, rtol=0.1)
        np.testing.assert_allclose(p90, 89.1, rtol=0.1)

    def test_get_multiple_percentiles(self) -> None:
        """
        Test getting multiple percentiles at once.
        """
        np.random.seed(42)
        sampler = ReservoirSampler(reservoir_size=100)

        # Add values
        values = np.arange(100, dtype=np.float64)
        sampler.add_samples(values.tolist())

        # Get multiple percentiles
        percentiles = sampler.get_percentiles([25, 50, 75])

        assert len(percentiles) == 3
        assert percentiles[25] < percentiles[50] < percentiles[75]

    def test_reset_sampler(self) -> None:
        """
        Test resetting the reservoir.
        """
        sampler = ReservoirSampler(reservoir_size=5)

        # Add values
        for i in range(10):
            sampler.add_sample(float(i))

        assert sampler.count > 0

        # Reset
        sampler.reset()
        assert sampler.count == 0
        assert sampler.total_seen == 0

    def test_edge_cases(self) -> None:
        """
        Test edge cases for reservoir sampler.
        """
        sampler = ReservoirSampler(reservoir_size=5)

        # Get sample from empty sampler
        sample = sampler.get_sample()
        assert len(sample) == 0

        # Get percentile from empty sampler
        result = sampler.get_percentile(50)
        assert result == 0.0  # Returns 0 for empty sampler


class TestPreAllocatedFeatureCache:
    """
    Test suite for PreAllocatedFeatureCache behavior.
    """

    def test_init_cache(self) -> None:
        """
        Test cache initialization.
        """
        cache = PreAllocatedFeatureCache(n_features=10, history_size=100)
        assert cache.n_features == 10
        assert cache.history_size == 100

    def test_init_with_invalid_params_raises(self) -> None:
        """
        Test cache initialization with invalid parameters.
        """
        with pytest.raises(ValueError, match="features must be positive"):
            PreAllocatedFeatureCache(n_features=0, history_size=100)

        with pytest.raises(ValueError, match="History size must be positive"):
            PreAllocatedFeatureCache(n_features=10, history_size=0)

    def test_store_and_retrieve_features(self) -> None:
        """
        Test storing and retrieving features.
        """
        cache = PreAllocatedFeatureCache(n_features=5, history_size=10)

        # Create feature array
        features = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

        # Write to current buffer then store
        buffer = cache.get_current_buffer()
        buffer[:] = features
        cache.store_current_features()

        # Get current features
        current = cache.get_current_buffer()
        np.testing.assert_array_equal(current, features)

    def test_get_feature_history(self) -> None:
        """
        Test retrieving historical features.
        """
        cache = PreAllocatedFeatureCache(n_features=3, history_size=5)

        # Add multiple feature sets
        for i in range(4):
            buffer = cache.get_current_buffer()
            buffer[:] = float(i)
            cache.store_current_features()

        # Get feature history
        history = cache.get_feature_history(n_latest=3)

        # Should get last 3 feature sets
        assert history.shape == (3, 3)
        np.testing.assert_array_equal(history[-1], [3.0, 3.0, 3.0])

    def test_buffer_wraparound(self) -> None:
        """
        Test cache handles buffer wraparound correctly.
        """
        cache = PreAllocatedFeatureCache(n_features=2, history_size=3)

        # Fill beyond buffer size
        for i in range(5):
            buffer = cache.get_current_buffer()
            buffer[0] = float(i)
            buffer[1] = float(i + 1)
            cache.store_current_features()

        # Check history count
        assert cache.history_count == 3  # Max is history_size

    def test_get_normalized_view(self) -> None:
        """
        Test getting normalized feature view.
        """
        cache = PreAllocatedFeatureCache(n_features=3, history_size=10)

        # Store features
        buffer = cache.get_current_buffer()
        buffer[:] = [10.0, 20.0, 30.0]
        cache.store_current_features()

        # Get normalized view
        normalized = cache.get_normalized_buffer()

        # Check shape
        assert normalized.shape == (3,)

        # Values should be normalized (implementation dependent)
        assert normalized is not None

    def test_prepare_onnx_input(self) -> None:
        """
        Test preparing features for ONNX inference.
        """
        cache = PreAllocatedFeatureCache(n_features=5, history_size=10)

        # Store features
        np.random.seed(42)
        buffer = cache.get_current_buffer()
        buffer[:] = np.random.randn(5).astype(np.float32)
        cache.store_current_features()

        # Prepare ONNX input
        onnx_ready = cache.prepare_onnx_input()

        # Should be properly shaped for ONNX
        assert onnx_ready.shape == (1, 5)  # Batch dimension added
        assert onnx_ready.dtype == np.float32

    def test_get_onnx_input_buffer(self) -> None:
        """
        Test getting pre-allocated ONNX input buffer.
        """
        cache = PreAllocatedFeatureCache(n_features=4, history_size=10)

        # Get ONNX buffer
        buffer = cache.get_onnx_input_buffer()

        # Should be pre-allocated with correct shape
        assert buffer.shape == (1, 4)
        assert buffer.dtype == np.float32

    def test_cache_reset(self) -> None:
        """
        Test resetting cache to initial state.
        """
        cache = PreAllocatedFeatureCache(n_features=3, history_size=5)

        # Add some features
        for i in range(3):
            buffer = cache.get_current_buffer()
            buffer[:] = i
            cache.store_current_features()

        assert cache.history_count > 0

        # Reset cache
        cache.reset()

        # Should be empty
        assert cache.history_count == 0


class TestCacheIntegration:
    """
    Integration tests for cache components working together.
    """

    def test_ring_buffer_and_percentile_tracking(self) -> None:
        """
        Test using ring buffer for percentile tracking.
        """
        # Simulate tracking prediction confidence
        confidence_buffer = LockFreeRingBuffer(size=100)

        # Add confidence scores
        np.random.seed(42)
        for i in range(150):
            confidence = np.random.beta(5, 2)  # Skewed distribution
            confidence_buffer.append(confidence)

        # Buffer should only keep last 100
        assert confidence_buffer.count == 100
        assert confidence_buffer.is_full

        # Check percentiles
        p50 = confidence_buffer.percentile(50)
        p95 = confidence_buffer.percentile(95)

        assert 0 <= p50 <= 1
        assert p50 < p95 <= 1

    def test_feature_cache_with_ring_buffer_history(self) -> None:
        """
        Test combining feature cache with ring buffer for metrics.
        """
        feature_cache = PreAllocatedFeatureCache(n_features=5, history_size=50)
        metric_buffer = LockFreeRingBuffer(size=50)

        # Simulate feature updates with metric tracking
        np.random.seed(42)
        for i in range(30):
            buffer = feature_cache.get_current_buffer()
            buffer[:] = np.random.randn(5).astype(np.float32)
            feature_cache.store_current_features()

            # Compute and track a metric
            feature_norm = np.linalg.norm(buffer)
            metric_buffer.append(float(feature_norm))

        # Both should have same count
        assert feature_cache.history_count == 30
        assert metric_buffer.count == 30

        # Metrics should correspond to features
        recent_features = feature_cache.get_current_buffer()
        last_metric = metric_buffer.get_last(1)[0]
        expected_metric = np.linalg.norm(recent_features)
        np.testing.assert_allclose(last_metric, expected_metric, rtol=1e-5)

    def test_reservoir_sampler_for_outlier_detection(self) -> None:
        """
        Test using reservoir sampler for outlier detection.
        """
        np.random.seed(42)
        sampler = ReservoirSampler(reservoir_size=100)
        outlier_buffer = LockFreeRingBuffer(size=10)

        # Stream values with occasional outliers
        np.random.seed(42)
        for i in range(500):
            if i % 50 == 0:
                value = np.random.randn() * 10  # Outlier
            else:
                value = np.random.randn()  # Normal

            sampler.add_sample(value)

            # Check if outlier based on reservoir percentiles
            if sampler.count > 10:
                p95 = sampler.get_percentile(95)
                p5 = sampler.get_percentile(5)

                if value > p95 or value < p5:
                    outlier_buffer.append(value)

        # Should have detected some outliers
        assert outlier_buffer.count > 0

        # Outliers should be extreme values
        outliers = outlier_buffer.get_all()
        assert np.max(np.abs(outliers)) > 2.0  # Should be beyond 2 std
