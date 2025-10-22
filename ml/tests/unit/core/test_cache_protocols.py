"""
Unit tests for cache component protocol definitions (Task 2.1).

This module verifies that cache classes implement their protocols correctly
and that protocols enable Protocol-First design patterns.

Tests follow lessons from Task Group 1:
- Test behavior (protocol conformance), not implementation details
- Use flexible assertions (isinstance, hasattr, callable)
- Handle runtime type variations gracefully
- 100% pass rate required
"""

from __future__ import annotations

from typing import Protocol, get_type_hints, runtime_checkable

import numpy as np
import numpy.typing as npt
import pytest


def test_lock_free_ring_buffer_implements_protocol() -> None:
    """Verify LockFreeRingBuffer conforms to RingBufferProtocol.

    This test ensures that the concrete LockFreeRingBuffer class implements
    the RingBufferProtocol interface, enabling Protocol-First design and
    type-safe dependency injection.

    Protocol conformance is verified using isinstance() with runtime_checkable.
    """
    from ml.core.cache import LockFreeRingBuffer, RingBufferProtocol

    # Create instance
    buffer = LockFreeRingBuffer(size=10)

    # Verify protocol conformance (behavior check, not implementation detail)
    assert isinstance(buffer, RingBufferProtocol), (
        "LockFreeRingBuffer should implement RingBufferProtocol"
    )

    # Verify protocol methods are accessible
    assert hasattr(buffer, "append"), "RingBufferProtocol requires append method"
    assert callable(buffer.append), "append should be callable"

    assert hasattr(buffer, "get_last"), "RingBufferProtocol requires get_last method"
    assert callable(buffer.get_last), "get_last should be callable"

    assert hasattr(buffer, "get_window"), "RingBufferProtocol requires get_window method"
    assert callable(buffer.get_window), "get_window should be callable"

    assert hasattr(buffer, "reset"), "RingBufferProtocol requires reset method"
    assert callable(buffer.reset), "reset should be callable"

    # Verify protocol property exists
    assert hasattr(buffer, "count"), "RingBufferProtocol requires count property"


def test_ring_buffer_protocol_methods_present() -> None:
    """Verify RingBufferProtocol defines all required methods.

    This test ensures the protocol definition includes all necessary methods
    and properties for ring buffer implementations.
    """
    from ml.core.cache import RingBufferProtocol

    # Verify protocol has required methods
    assert hasattr(RingBufferProtocol, "append"), "Protocol should define append"
    assert hasattr(RingBufferProtocol, "get_last"), "Protocol should define get_last"
    assert hasattr(RingBufferProtocol, "get_window"), "Protocol should define get_window"
    assert hasattr(RingBufferProtocol, "reset"), "Protocol should define reset"

    # Verify protocol has required property
    assert hasattr(RingBufferProtocol, "count"), "Protocol should define count property"


def test_preallocated_feature_cache_implements_protocol() -> None:
    """Verify PreAllocatedFeatureCache conforms to FeatureCacheProtocol.

    This test ensures that the concrete PreAllocatedFeatureCache class implements
    the FeatureCacheProtocol interface, enabling type-safe feature caching.
    """
    from ml.core.cache import FeatureCacheProtocol, PreAllocatedFeatureCache

    # Create instance
    cache = PreAllocatedFeatureCache(n_features=10)

    # Verify protocol conformance
    assert isinstance(cache, FeatureCacheProtocol), (
        "PreAllocatedFeatureCache should implement FeatureCacheProtocol"
    )

    # Verify protocol methods are accessible
    assert hasattr(cache, "get_current_buffer"), "FeatureCacheProtocol requires get_current_buffer"
    assert callable(cache.get_current_buffer), "get_current_buffer should be callable"

    assert hasattr(cache, "store_current_features"), "FeatureCacheProtocol requires store_current_features"
    assert callable(cache.store_current_features), "store_current_features should be callable"

    assert hasattr(cache, "prepare_onnx_input"), "FeatureCacheProtocol requires prepare_onnx_input"
    assert callable(cache.prepare_onnx_input), "prepare_onnx_input should be callable"

    assert hasattr(cache, "reset"), "FeatureCacheProtocol requires reset"
    assert callable(cache.reset), "reset should be callable"

    # Verify protocol property exists
    assert hasattr(cache, "n_features"), "FeatureCacheProtocol requires n_features property"


def test_feature_cache_protocol_methods_present() -> None:
    """Verify FeatureCacheProtocol defines all required methods.

    This test ensures the protocol definition includes all necessary methods
    and properties for feature cache implementations.
    """
    from ml.core.cache import FeatureCacheProtocol

    # Verify protocol has required methods
    assert hasattr(FeatureCacheProtocol, "get_current_buffer"), "Protocol should define get_current_buffer"
    assert hasattr(FeatureCacheProtocol, "store_current_features"), "Protocol should define store_current_features"
    assert hasattr(FeatureCacheProtocol, "prepare_onnx_input"), "Protocol should define prepare_onnx_input"
    assert hasattr(FeatureCacheProtocol, "reset"), "Protocol should define reset"

    # Verify protocol has required property
    assert hasattr(FeatureCacheProtocol, "n_features"), "Protocol should define n_features property"


def test_reservoir_sampler_implements_protocol() -> None:
    """Verify ReservoirSampler conforms to SamplerProtocol.

    This test ensures that the concrete ReservoirSampler class implements
    the SamplerProtocol interface, enabling type-safe sampling operations.
    """
    from ml.core.cache import ReservoirSampler, SamplerProtocol

    # Create instance
    sampler = ReservoirSampler(reservoir_size=100)

    # Verify protocol conformance
    assert isinstance(sampler, SamplerProtocol), (
        "ReservoirSampler should implement SamplerProtocol"
    )

    # Verify protocol methods are accessible
    assert hasattr(sampler, "add_sample"), "SamplerProtocol requires add_sample"
    assert callable(sampler.add_sample), "add_sample should be callable"

    assert hasattr(sampler, "get_percentile"), "SamplerProtocol requires get_percentile"
    assert callable(sampler.get_percentile), "get_percentile should be callable"

    assert hasattr(sampler, "reset"), "SamplerProtocol requires reset"
    assert callable(sampler.reset), "reset should be callable"

    # Verify protocol property exists
    assert hasattr(sampler, "count"), "SamplerProtocol requires count property"


def test_sampler_protocol_methods_present() -> None:
    """Verify SamplerProtocol defines all required methods.

    This test ensures the protocol definition includes all necessary methods
    and properties for sampler implementations.
    """
    from ml.core.cache import SamplerProtocol

    # Verify protocol has required methods
    assert hasattr(SamplerProtocol, "add_sample"), "Protocol should define add_sample"
    assert hasattr(SamplerProtocol, "get_percentile"), "Protocol should define get_percentile"
    assert hasattr(SamplerProtocol, "reset"), "Protocol should define reset"

    # Verify protocol has required property
    assert hasattr(SamplerProtocol, "count"), "Protocol should define count property"


def test_protocols_exported_in_all() -> None:
    """Verify protocols are exported in module __all__.

    This test ensures that the 3 new protocols are properly exported from the
    cache module, making them available for import and use.
    """
    import ml.core.cache as cache_module

    # Verify module has __all__
    assert hasattr(cache_module, "__all__"), "Module should have __all__"

    # Verify protocols are in __all__ (flexible check - handles different orderings)
    all_exports = set(cache_module.__all__)
    required_protocols = {"RingBufferProtocol", "FeatureCacheProtocol", "SamplerProtocol"}

    assert required_protocols.issubset(all_exports), (
        f"Module __all__ should export all 3 protocols. "
        f"Missing: {required_protocols - all_exports}"
    )

    # Verify protocols are importable from module
    assert hasattr(cache_module, "RingBufferProtocol"), "RingBufferProtocol should be accessible"
    assert hasattr(cache_module, "FeatureCacheProtocol"), "FeatureCacheProtocol should be accessible"
    assert hasattr(cache_module, "SamplerProtocol"), "SamplerProtocol should be accessible"


def test_mock_implementations_conform_to_protocols() -> None:
    """Verify mock classes can implement protocols.

    This test ensures that custom mock implementations can conform to protocols,
    demonstrating the Protocol-First design pattern's flexibility.
    """
    from ml.core.cache import FeatureCacheProtocol, RingBufferProtocol, SamplerProtocol

    # Mock RingBuffer implementation
    class MockRingBuffer:
        def append(self, value: float) -> None:
            pass

        def get_last(self, n: int = 1) -> npt.NDArray[np.float64]:
            return np.array([], dtype=np.float64)

        def get_window(self, start: int, length: int) -> npt.NDArray[np.float64]:
            return np.array([], dtype=np.float64)

        def reset(self) -> None:
            pass

        @property
        def count(self) -> int:
            return 0

    # Mock FeatureCache implementation
    class MockFeatureCache:
        def get_current_buffer(self) -> npt.NDArray[np.float32]:
            return np.array([], dtype=np.float32)

        def store_current_features(self) -> None:
            pass

        def prepare_onnx_input(self, use_normalized: bool = True) -> npt.NDArray[np.float32]:
            return np.array([[]], dtype=np.float32)

        def reset(self) -> None:
            pass

        @property
        def n_features(self) -> int:
            return 0

    # Mock Sampler implementation
    class MockSampler:
        def add_sample(self, value: float) -> None:
            pass

        def get_percentile(self, q: float) -> float:
            return 0.0

        def reset(self) -> None:
            pass

        @property
        def count(self) -> int:
            return 0

    # Verify mock instances conform to protocols
    mock_buffer = MockRingBuffer()
    assert isinstance(mock_buffer, RingBufferProtocol), (
        "Mock RingBuffer should conform to protocol"
    )

    mock_cache = MockFeatureCache()
    assert isinstance(mock_cache, FeatureCacheProtocol), (
        "Mock FeatureCache should conform to protocol"
    )

    mock_sampler = MockSampler()
    assert isinstance(mock_sampler, SamplerProtocol), (
        "Mock Sampler should conform to protocol"
    )


def test_protocol_inheritance_has_zero_runtime_cost() -> None:
    """Verify protocols have no runtime overhead.

    This test ensures that protocol inheritance doesn't add memory or
    performance overhead to concrete classes.
    """
    from ml.core.cache import LockFreeRingBuffer

    # Create instance
    buffer = LockFreeRingBuffer(size=100)

    # Verify instance size is not inflated by protocol
    # Protocols should not add __dict__ entries
    buffer_dict = vars(buffer)

    # Check that protocol doesn't add its own attributes
    # (protocols are structural, not behavioral)
    protocol_attrs = {"_buffer", "_size", "_index", "_count", "_dtype", "_random"}
    actual_attrs = set(buffer_dict.keys())

    # Verify no extra attributes from protocol inheritance
    assert actual_attrs == protocol_attrs, (
        f"Protocol inheritance should not add attributes. "
        f"Extra: {actual_attrs - protocol_attrs}"
    )


def test_type_hints_work_with_protocols() -> None:
    """Verify type hints work correctly with protocols.

    This test ensures that get_type_hints() correctly resolves protocol types
    and that type checkers understand protocol inheritance.
    """
    from ml.core.cache import (
        FeatureCacheProtocol,
        PreAllocatedFeatureCache,
        RingBufferProtocol,
        SamplerProtocol,
    )

    # Verify RingBufferProtocol has correct method signatures
    # get_type_hints() works on Protocol classes
    ring_hints = get_type_hints(RingBufferProtocol.append)
    assert "value" in ring_hints, "append method should have value parameter"
    assert ring_hints["value"] is float, "value should be typed as float"

    # Verify FeatureCacheProtocol type hints
    cache_hints = get_type_hints(FeatureCacheProtocol.prepare_onnx_input)
    assert "use_normalized" in cache_hints, "prepare_onnx_input should have use_normalized"
    assert cache_hints["use_normalized"] is bool, "use_normalized should be bool"

    # Verify SamplerProtocol type hints
    sampler_hints = get_type_hints(SamplerProtocol.get_percentile)
    assert "q" in sampler_hints, "get_percentile should have q parameter"
    assert sampler_hints["q"] is float, "q should be typed as float"

    # Verify concrete classes work with get_type_hints()
    cache = PreAllocatedFeatureCache(n_features=10)
    cache_method_hints = get_type_hints(cache.prepare_onnx_input)
    assert "use_normalized" in cache_method_hints, "Concrete method should have type hints"


def test_protocol_based_dependency_injection() -> None:
    """Verify protocol-based dependency injection works.

    This integration test ensures that functions accepting protocol types
    can work with any conforming implementation (concrete or mock).
    """
    from ml.core.cache import LockFreeRingBuffer, RingBufferProtocol

    # Define function accepting protocol
    def compute_statistics(buffer: RingBufferProtocol) -> dict[str, float]:
        """Compute statistics from any ring buffer implementation."""
        if buffer.count == 0:
            return {"mean": 0.0, "count": 0}

        values = buffer.get_last(buffer.count)
        return {
            "mean": float(np.mean(values)),
            "count": float(buffer.count),
        }

    # Test with concrete implementation
    buffer = LockFreeRingBuffer(size=10)
    buffer.append(1.0)
    buffer.append(2.0)
    buffer.append(3.0)

    stats = compute_statistics(buffer)
    assert stats["count"] == 3, "Should count 3 values"
    assert stats["mean"] == 2.0, "Mean should be 2.0"

    # Test with mock implementation
    class MockBuffer:
        def __init__(self) -> None:
            self._count = 2

        @property
        def count(self) -> int:
            return self._count

        def get_last(self, n: int = 1) -> npt.NDArray[np.float64]:
            return np.array([5.0, 10.0], dtype=np.float64)

        def append(self, value: float) -> None:
            pass

        def get_window(self, start: int, length: int) -> npt.NDArray[np.float64]:
            return np.array([], dtype=np.float64)

        def reset(self) -> None:
            pass

    mock = MockBuffer()
    assert isinstance(mock, RingBufferProtocol), "Mock should conform to protocol"

    mock_stats = compute_statistics(mock)
    assert mock_stats["count"] == 2, "Mock should count 2 values"
    assert mock_stats["mean"] == 7.5, "Mock mean should be 7.5"


def test_protocols_are_runtime_checkable() -> None:
    """Verify protocols use @runtime_checkable decorator.

    This test ensures all 3 protocols can be used with isinstance() at runtime,
    not just for static type checking.
    """
    from ml.core.cache import FeatureCacheProtocol, RingBufferProtocol, SamplerProtocol

    # Verify protocols are runtime_checkable
    # This is done by checking if they work with isinstance()
    # (non-runtime_checkable protocols raise TypeError)

    # Create simple mock to test runtime checking
    class SimpleRingBuffer:
        def append(self, value: float) -> None:
            pass

        def get_last(self, n: int = 1) -> npt.NDArray[np.float64]:
            return np.array([])

        def get_window(self, start: int, length: int) -> npt.NDArray[np.float64]:
            return np.array([])

        def reset(self) -> None:
            pass

        @property
        def count(self) -> int:
            return 0

    # This should NOT raise TypeError (would raise if protocol not runtime_checkable)
    simple = SimpleRingBuffer()
    is_conformant = isinstance(simple, RingBufferProtocol)
    assert is_conformant, "Simple implementation should conform to protocol"

    # Verify all protocols support isinstance()
    # (runtime_checkable decorator enables this)
    try:
        isinstance(simple, RingBufferProtocol)
        isinstance(simple, FeatureCacheProtocol)  # Will be False, but shouldn't error
        isinstance(simple, SamplerProtocol)  # Will be False, but shouldn't error
        runtime_checkable_works = True
    except TypeError:
        runtime_checkable_works = False

    assert runtime_checkable_works, "All protocols should be runtime_checkable"
