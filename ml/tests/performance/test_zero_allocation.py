"""
Tests for verifying zero-allocation hot path in ML infrastructure.

This module tests that the ML signal generation hot path truly has zero memory
allocations by tracking memory usage and array sharing.

"""

import gc
import logging
import tracemalloc
from typing import Any

import numpy as np
import pytest

from ml.core.cache import LockFreeRingBuffer
from ml.core.cache import PreAllocatedFeatureCache
from ml.core.cache import ReservoirSampler
from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.features.engineering import IndicatorManager
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity


# Configure module logger
logger = logging.getLogger(__name__)


class TestZeroAllocationHotPath:
    """
    Test suite for verifying zero-allocation hot path.
    """

    @pytest.fixture
    def setup_bar_data(self) -> tuple[BarType, list[Bar]]:
        """
        Create test bar data.
        """
        instrument_id = InstrumentId.from_str("TEST.ZERO")
        bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
        bar_type = BarType(instrument_id, bar_spec, AggressorSide.BUYER)

        bars = []
        for i in range(100):
            bar = Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{100.0 + i * 0.1:.2f}"),
                high=Price.from_str(f"{100.5 + i * 0.1:.2f}"),
                low=Price.from_str(f"{99.5 + i * 0.1:.2f}"),
                close=Price.from_str(f"{100.0 + i * 0.1:.2f}"),
                volume=Quantity.from_str(f"{1000 + i * 10}"),
                ts_event=i * 1000000000,
                ts_init=i * 1000000000,
            )
            bars.append(bar)

        return bar_type, bars

    def test_ring_buffer_get_last_returns_view(self) -> None:
        """
        Test that get_last returns a view when possible.
        """
        buffer = LockFreeRingBuffer(100)

        # Add some data
        for i in range(50):
            buffer.append(float(i))

        # Get last values - should be a view
        result = buffer.get_last(10)

        # Check that it's a view of the internal buffer
        assert np.shares_memory(result, buffer._buffer), "get_last should return a view"

        # Verify data is correct
        expected = np.array([40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0])
        np.testing.assert_array_equal(result, expected)

    def test_ring_buffer_get_window_returns_view(self) -> None:
        """
        Test that get_window returns a view when possible.
        """
        buffer = LockFreeRingBuffer(100)

        # Add some data
        for i in range(50):
            buffer.append(float(i))

        # Get window - should be a view when contiguous
        result = buffer.get_window(10, 20)

        # Check that it's a view of the internal buffer
        assert np.shares_memory(result, buffer._buffer), "get_window should return a view"

        # Verify data is correct
        expected = np.arange(10.0, 30.0)
        np.testing.assert_array_equal(result, expected)

    def test_ring_buffer_wraparound_requires_allocation(self) -> None:
        """
        Test that wraparound cases require allocation (unavoidable).
        """
        buffer = LockFreeRingBuffer(10)

        # Fill buffer beyond capacity to cause wraparound
        for i in range(15):
            buffer.append(float(i))

        # This should require concatenation due to wraparound
        result = buffer.get_last(10)

        # Result should still be correct
        expected = np.arange(5.0, 15.0)
        np.testing.assert_array_equal(result, expected)

        # Note: We can't avoid allocation here due to wraparound

    def test_reservoir_sampler_get_sample_returns_view(self) -> None:
        """
        Test that get_sample returns a view.
        """
        sampler = ReservoirSampler(100)

        # Add samples
        for i in range(50):
            sampler.add_sample(float(i))

        # Get sample - should be a view
        result = sampler.get_sample()

        # Check that it's a view of the internal reservoir
        assert np.shares_memory(result, sampler._reservoir), "get_sample should return a view"
        assert len(result) == 50

    def test_feature_cache_returns_views(self) -> None:
        """
        Test that feature cache returns views of pre-allocated buffers.
        """
        cache = PreAllocatedFeatureCache(n_features=10, history_size=100)

        # Get buffers - should be views of pre-allocated arrays
        current_buffer = cache.get_current_buffer()
        normalized_buffer = cache.get_normalized_buffer()
        onnx_buffer = cache.get_onnx_input_buffer()

        # Verify they're the actual pre-allocated buffers
        assert current_buffer is cache._current_features
        assert normalized_buffer is cache._normalized_features
        assert onnx_buffer is cache._onnx_input_buffer

        # Modify buffer and verify it affects the cache
        current_buffer[0] = 42.0
        assert cache._current_features[0] == 42.0

    def test_feature_cache_history_returns_view_when_contiguous(self) -> None:
        """
        Test that get_feature_history returns view when data is contiguous.
        """
        cache = PreAllocatedFeatureCache(n_features=5, history_size=100)

        # Store some features
        for i in range(10):
            cache._current_features[:] = float(i)
            cache.store_current_features()

        # Get history - should be a view when contiguous
        history = cache.get_feature_history(5)

        # Check if it shares memory with the history buffer
        assert np.shares_memory(
            history,
            cache._feature_history,
        ), "get_feature_history should return a view when contiguous"

    def test_feature_engineer_returns_buffer_view(self) -> None:
        """
        Test that feature engineer returns a view of its buffer in hot path.
        """
        config = FeatureConfig()
        engineer = FeatureEngineer(config)

        # Setup indicator manager
        indicator_mgr = IndicatorManager(config)

        # Create bar data
        bar_data = {
            "close": 100.0,
            "volume": 1000.0,
            "high": 101.0,
            "low": 99.0,
        }

        # Warm up indicators with some bars
        from nautilus_trader.model.data import Bar
        from nautilus_trader.model.data import BarSpecification
        from nautilus_trader.model.data import BarType
        from nautilus_trader.model.enums import AggressorSide
        from nautilus_trader.model.enums import BarAggregation
        from nautilus_trader.model.enums import PriceType
        from nautilus_trader.model.identifiers import InstrumentId
        from nautilus_trader.model.objects import Price
        from nautilus_trader.model.objects import Quantity

        instrument_id = InstrumentId.from_str("TEST.ZERO")
        bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
        bar_type = BarType(instrument_id, bar_spec, AggressorSide.BUYER)

        for i in range(30):  # Warm up indicators
            bar = Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{100.0 + i * 0.1:.2f}"),
                high=Price.from_str(f"{100.5 + i * 0.1:.2f}"),
                low=Price.from_str(f"{99.5 + i * 0.1:.2f}"),
                close=Price.from_str(f"{100.0 + i * 0.1:.2f}"),
                volume=Quantity.from_str(f"{1000 + i * 10}"),
                ts_event=i * 1000000000,
                ts_init=i * 1000000000,
            )
            indicator_mgr.update_from_bar(bar)

        # Calculate features online (hot path)
        features = engineer.calculate_features_online(
            current_bar=bar_data,
            indicator_manager=indicator_mgr,
            scaler=None,
        )

        # Verify it's a view of the feature buffer
        assert np.shares_memory(
            features,
            engineer.feature_buffer,
        ), "calculate_features_online should return a view of feature_buffer"

    def test_hot_path_memory_stability(self, setup_bar_data: tuple[BarType, list[Bar]]) -> None:
        """
        Test that hot path maintains stable memory over many iterations.
        """
        bar_type, bars = setup_bar_data

        # Setup feature engineering
        config = FeatureConfig()
        engineer = FeatureEngineer(config)
        indicator_mgr = IndicatorManager(config)

        # Warm up with initial bars
        for bar in bars[:30]:
            indicator_mgr.update_from_bar(bar)

        # Force garbage collection to get clean baseline
        gc.collect()

        # Track memory allocations
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        # Process many bars in hot path
        bar_data = {
            "close": 100.0,
            "volume": 1000.0,
            "high": 101.0,
            "low": 99.0,
        }

        for _ in range(1000):  # Process many iterations
            features = engineer.calculate_features_online(
                current_bar=bar_data,
                indicator_manager=indicator_mgr,
                scaler=None,
            )
            # Simulate using features (without allocation)
            _ = features[0]

        # Take second snapshot
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Compare snapshots
        top_stats = snapshot2.compare_to(snapshot1, "lineno")

        # Filter to only show allocations from our module
        ml_allocations = [
            stat for stat in top_stats if stat.size_diff > 0 and "ml/" in str(stat.traceback)
        ]

        # There should be minimal or no allocations in the hot path
        total_allocated = sum(stat.size_diff for stat in ml_allocations)

        # Allow small allocation for Python overhead, but should be minimal
        assert (
            total_allocated < 10000
        ), f"Hot path allocated {total_allocated} bytes, expected near zero. Allocations: {ml_allocations[:5]}"

    def test_feature_parity_with_views(self, setup_bar_data: tuple[BarType, list[Bar]]) -> None:
        """
        Test that using views maintains feature parity.
        """
        bar_type, bars = setup_bar_data

        # Setup feature engineering
        config = FeatureConfig()
        engineer = FeatureEngineer(config)
        indicator_mgr = IndicatorManager(config)

        # Warm up indicators
        for bar in bars[:30]:
            indicator_mgr.update_from_bar(bar)

        # Calculate features for the same bar multiple times
        bar_data = {
            "close": float(bars[30].close),
            "volume": float(bars[30].volume),
            "high": float(bars[30].high),
            "low": float(bars[30].low),
        }

        results = []
        for _ in range(10):
            features = engineer.calculate_features_online(
                current_bar=bar_data,
                indicator_manager=indicator_mgr,
                scaler=None,
            )
            # Store a copy for comparison (only for test, not in hot path)
            results.append(features.copy())

        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_allclose(
                results[0],
                results[i],
                rtol=1e-10,
                err_msg=f"Feature parity violation at iteration {i}",
            )

    def test_ring_buffer_memory_views_are_safe(self) -> None:
        """
        Test that ring buffer views remain valid after updates.
        """
        buffer = LockFreeRingBuffer(10)

        # Fill buffer
        for i in range(10):
            buffer.append(float(i))

        # Get a view
        view1 = buffer.get_last(5)
        original_values = view1.copy()  # Save for comparison

        # Add more data (will overwrite oldest)
        for i in range(10, 15):
            buffer.append(float(i))

        # The view's memory is still valid but contains new data
        # This is expected behavior for views
        view2 = buffer.get_last(5)

        # New view should have different values
        assert not np.array_equal(view1, original_values) or not np.array_equal(
            view2,
            original_values,
        )

        # But new view should be correct
        expected = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        np.testing.assert_array_equal(view2, expected)


def create_test_bar_data() -> tuple[Any, list[Any]]:
    """
    Create test bar data for standalone run.
    """
    from nautilus_trader.model.data import Bar
    from nautilus_trader.model.data import BarSpecification
    from nautilus_trader.model.data import BarType
    from nautilus_trader.model.enums import AggressorSide
    from nautilus_trader.model.enums import BarAggregation
    from nautilus_trader.model.enums import PriceType
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.objects import Price
    from nautilus_trader.model.objects import Quantity

    instrument_id = InstrumentId.from_str("TEST.ZERO")
    bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
    bar_type = BarType(instrument_id, bar_spec, AggressorSide.BUYER)

    bars = []
    for i in range(100):
        bar = Bar(
            bar_type=bar_type,
            open=Price.from_str(f"{100.0 + i * 0.1:.2f}"),
            high=Price.from_str(f"{100.5 + i * 0.1:.2f}"),
            low=Price.from_str(f"{99.5 + i * 0.1:.2f}"),
            close=Price.from_str(f"{100.0 + i * 0.1:.2f}"),
            volume=Quantity.from_str(f"{1000 + i * 10}"),
            ts_event=i * 1000000000,
            ts_init=i * 1000000000,
        )
        bars.append(bar)

    return bar_type, bars


if __name__ == "__main__":
    # Run tests
    test = TestZeroAllocationHotPath()

    # Create test data
    bar_data = create_test_bar_data()

    # Run key tests
    logger.info("Testing ring buffer views...")
    test.test_ring_buffer_get_last_returns_view()
    test.test_ring_buffer_get_window_returns_view()
    logger.info(" Ring buffer returns views")

    logger.info("\nTesting feature cache views...")
    test.test_feature_cache_returns_views()
    test.test_feature_cache_history_returns_view_when_contiguous()
    logger.info(" Feature cache returns views")

    logger.info("\nTesting feature engineer views...")
    test.test_feature_engineer_returns_buffer_view()
    logger.info(" Feature engineer returns buffer views")

    logger.info("\nTesting memory stability...")
    test.test_hot_path_memory_stability(bar_data)
    logger.info(" Hot path has stable memory")

    logger.info("\nTesting feature parity...")
    test.test_feature_parity_with_views(bar_data)
    logger.info(" Feature parity maintained with views")

    logger.info("\n All zero-allocation hot path tests passed!")
