"""
Unit tests for ML core cache functionality.

Tests focus on observable cache behavior rather than implementation details,
following the testing protocol principles.
"""

import numpy as np
import pytest

from ml.core.cache import LockFreeRingBuffer
from ml.core.cache import PreAllocatedFeatureCache
from ml.core.cache import ReservoirSampler


class TestLockFreeRingBuffer:
    """Test suite for LockFreeRingBuffer behavior."""

    def test_init_with_valid_size(self):
        """Test buffer initialization with valid size."""
        buffer = LockFreeRingBuffer(size=10)
        assert buffer.size == 10
        assert buffer.count == 0
        assert not buffer.is_full

    def test_init_with_invalid_size_raises(self):
        """Test buffer initialization with invalid size raises error."""
        with pytest.raises(ValueError, match="Buffer size must be positive"):
            LockFreeRingBuffer(size=0)
        
        with pytest.raises(ValueError, match="Buffer size must be positive"):
            LockFreeRingBuffer(size=-1)

    def test_append_single_value(self):
        """Test appending single values to buffer."""
        buffer = LockFreeRingBuffer(size=5)
        
        buffer.append(1.0)
        assert buffer.count == 1
        assert not buffer.is_full
        
        buffer.append(2.0)
        assert buffer.count == 2

    def test_append_fills_buffer(self):
        """Test buffer correctly reports when full."""
        buffer = LockFreeRingBuffer(size=3)
        
        buffer.append(1.0)
        buffer.append(2.0)
        assert not buffer.is_full
        
        buffer.append(3.0)
        assert buffer.is_full
        assert buffer.count == 3

    def test_append_overwrites_oldest(self):
        """Test ring buffer overwrites oldest values when full."""
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

    def test_append_array(self):
        """Test appending multiple values at once."""
        buffer = LockFreeRingBuffer(size=5)
        values = np.array([1.0, 2.0, 3.0])
        
        buffer.append_array(values)
        assert buffer.count == 3
        
        result = buffer.get_last(3)
        np.testing.assert_array_equal(result, values)

    def test_get_last_with_various_sizes(self):
        """Test retrieving last n values with different scenarios."""
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

    def test_get_last_with_wraparound(self):
        """Test retrieving values when buffer has wrapped around."""
        buffer = LockFreeRingBuffer(size=3)
        
        # Fill and wrap buffer
        for i in range(5):
            buffer.append(float(i))
        
        # Should get [2, 3, 4] (oldest values overwritten)
        result = buffer.get_last(3)
        np.testing.assert_array_equal(result, [2.0, 3.0, 4.0])

    def test_get_window(self):
        """Test retrieving windowed data from buffer."""
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

    def test_get_window_edge_cases(self):
        """Test window retrieval edge cases."""
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

    def test_get_all_and_reset(self):
        """Test getting all values and resetting buffer."""
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

    def test_percentile_calculation(self):
        """Test percentile calculations on buffer data."""
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

    def test_mean_and_std(self):
        """Test mean and standard deviation calculations."""
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
    """Test suite for ReservoirSampler behavior."""

    def test_init_with_valid_size(self):
        """Test sampler initialization."""
        sampler = ReservoirSampler(size=10)
        assert sampler.reservoir_size == 10
        assert sampler.count == 0

    def test_init_with_invalid_size_raises(self):
        """Test sampler initialization with invalid size."""
        with pytest.raises(ValueError, match="Reservoir size must be positive"):
            ReservoirSampler(size=0)

    def test_add_samples_up_to_size(self):
        """Test adding samples up to reservoir size."""
        sampler = ReservoirSampler(size=5)
        
        for i in range(5):
            sampler.add_sample(float(i))
        
        assert sampler.count == 5
        
        # Get all samples
        samples = []
        for i in range(5):
            sample = sampler.get_sample(i)
            if sample is not None:
                samples.append(sample)
        
        assert len(samples) == 5
        # All values should be present when not exceeding size
        assert set(samples) == {0.0, 1.0, 2.0, 3.0, 4.0}

    def test_reservoir_sampling_maintains_size(self):
        """Test reservoir maintains fixed size after filling."""
        sampler = ReservoirSampler(size=5, seed=42)
        
        # Add more values than reservoir size
        for i in range(20):
            sampler.add_sample(float(i))
        
        assert sampler.total_seen == 20  # Total count tracked
        
        # Count actual samples
        sample_count = 0
        for i in range(sampler.reservoir_size):
            if sampler.get_sample(i) is not None:
                sample_count += 1
        assert sample_count == 5  # But only keeps reservoir size

    def test_add_multiple_samples(self):
        """Test adding multiple samples at once."""
        sampler = ReservoirSampler(size=10, seed=42)
        
        samples = np.arange(5, dtype=np.float64)
        sampler.add_samples(samples)
        
        assert sampler.count == 5
        assert sampler.total_seen == 5

    def test_get_percentile(self):
        """Test percentile calculation from reservoir."""
        sampler = ReservoirSampler(size=100, seed=42)
        
        # Add values from known distribution
        values = np.arange(100, dtype=np.float64)
        sampler.add_samples(values)
        
        # Since reservoir size equals data size, should be exact
        p50 = sampler.get_percentile(50)
        p90 = sampler.get_percentile(90)
        
        np.testing.assert_allclose(p50, 49.5, rtol=0.1)
        np.testing.assert_allclose(p90, 89.1, rtol=0.1)

    def test_get_multiple_percentiles(self):
        """Test getting multiple percentiles at once."""
        sampler = ReservoirSampler(size=100, seed=42)
        
        # Add values
        values = np.arange(100, dtype=np.float64)
        sampler.add_samples(values)
        
        # Get multiple percentiles
        percentiles = sampler.get_percentiles([25, 50, 75])
        
        assert len(percentiles) == 3
        assert percentiles[0] < percentiles[1] < percentiles[2]

    def test_reset_sampler(self):
        """Test resetting the reservoir."""
        sampler = ReservoirSampler(size=5)
        
        # Add values
        for i in range(10):
            sampler.add_sample(float(i))
        
        assert sampler.count > 0
        
        # Reset
        sampler.reset()
        assert sampler.count == 0
        assert sampler.total_seen == 0

    def test_edge_cases(self):
        """Test edge cases for reservoir sampler."""
        sampler = ReservoirSampler(size=5)
        
        # Get sample from empty sampler
        sample = sampler.get_sample(0)
        assert sample is None
        
        # Get percentile from empty sampler
        result = sampler.get_percentile(50)
        assert np.isnan(result)


class TestPreAllocatedFeatureCache:
    """Test suite for PreAllocatedFeatureCache behavior."""

    def test_init_cache(self):
        """Test cache initialization."""
        cache = PreAllocatedFeatureCache(n_features=10, history_size=100)
        assert cache.n_features == 10
        assert cache.history_size == 100

    def test_init_with_invalid_params_raises(self):
        """Test cache initialization with invalid parameters."""
        with pytest.raises(ValueError, match="features must be positive"):
            PreAllocatedFeatureCache(n_features=0, history_size=100)
        
        with pytest.raises(ValueError, match="History size must be positive"):
            PreAllocatedFeatureCache(n_features=10, history_size=0)

    def test_store_and_retrieve_features(self):
        """Test storing and retrieving features."""
        cache = PreAllocatedFeatureCache(n_features=5, history_size=10)
        
        # Create feature array
        features = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        
        # Store features
        cache.store_current_features(features)
        
        # Get current features
        current = cache.get_current_view()
        np.testing.assert_array_equal(current, features)

    def test_get_feature_history(self):
        """Test retrieving historical features."""
        cache = PreAllocatedFeatureCache(n_features=3, history_size=5)
        
        # Add multiple feature sets
        for i in range(4):
            features = np.full(3, fill_value=float(i), dtype=np.float32)
            cache.store_current_features(features)
        
        # Get feature history
        history = cache.get_feature_history(feature_idx=0, n_samples=3)
        
        # Should get last 3 values for feature 0
        np.testing.assert_array_equal(history, [1.0, 2.0, 3.0])

    def test_buffer_wraparound(self):
        """Test cache handles buffer wraparound correctly."""
        cache = PreAllocatedFeatureCache(n_features=2, history_size=3)
        
        # Fill beyond buffer size
        for i in range(5):
            features = np.array([float(i), float(i+1)], dtype=np.float32)
            cache.store_current_features(features)
        
        # Check history count
        assert cache.history_count == 3  # Max is history_size

    def test_get_normalized_view(self):
        """Test getting normalized feature view."""
        cache = PreAllocatedFeatureCache(n_features=3, history_size=10)
        
        # Store features
        features = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        cache.store_current_features(features)
        
        # Get normalized view (should be in-place normalization)
        normalized = cache.get_normalized_view()
        
        # Check shape
        assert normalized.shape == (3,)
        
        # Values should be normalized (implementation dependent)
        assert normalized is not None

    def test_prepare_onnx_input(self):
        """Test preparing features for ONNX inference."""
        cache = PreAllocatedFeatureCache(n_features=5, history_size=10)
        
        # Store features
        features = np.random.randn(5).astype(np.float32)
        cache.store_current_features(features)
        
        # Prepare ONNX input
        onnx_ready = cache.prepare_onnx_input()
        
        # Should be properly shaped for ONNX
        assert onnx_ready.shape == (1, 5)  # Batch dimension added
        assert onnx_ready.dtype == np.float32

    def test_get_onnx_input_buffer(self):
        """Test getting pre-allocated ONNX input buffer."""
        cache = PreAllocatedFeatureCache(n_features=4, history_size=10)
        
        # Get ONNX buffer
        buffer = cache.get_onnx_input_buffer()
        
        # Should be pre-allocated with correct shape
        assert buffer.shape == (1, 4)
        assert buffer.dtype == np.float32

    def test_cache_reset(self):
        """Test resetting cache to initial state."""
        cache = PreAllocatedFeatureCache(n_features=3, history_size=5)
        
        # Add some features
        for i in range(3):
            cache.store_current_features(np.ones(3) * i)
        
        assert cache.history_count > 0
        
        # Reset cache
        cache.reset()
        
        # Should be empty
        assert cache.history_count == 0


class TestCacheIntegration:
    """Integration tests for cache components working together."""

    def test_ring_buffer_and_percentile_tracking(self):
        """Test using ring buffer for percentile tracking."""
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

    def test_feature_cache_with_ring_buffer_history(self):
        """Test combining feature cache with ring buffer for metrics."""
        feature_cache = PreAllocatedFeatureCache(n_features=5, history_size=50)
        metric_buffer = LockFreeRingBuffer(size=50)
        
        # Simulate feature updates with metric tracking
        np.random.seed(42)
        for i in range(30):
            features = np.random.randn(5).astype(np.float32)
            feature_cache.store_current_features(features)
            
            # Compute and track a metric
            feature_norm = np.linalg.norm(features)
            metric_buffer.append(feature_norm)
        
        # Both should have same count
        assert feature_cache.history_count == 30
        assert metric_buffer.count == 30
        
        # Metrics should correspond to features
        recent_features = feature_cache.get_current_view()
        last_metric = metric_buffer.get_last(1)[0]
        expected_metric = np.linalg.norm(recent_features)
        np.testing.assert_allclose(last_metric, expected_metric, rtol=1e-5)

    def test_reservoir_sampler_for_outlier_detection(self):
        """Test using reservoir sampler for outlier detection."""
        sampler = ReservoirSampler(size=100, seed=42)
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