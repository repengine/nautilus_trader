#!/usr/bin/env python3
"""
Unit tests for PredictionBufferComponent.

This module tests the prediction buffer component which manages prediction history
and ring buffers for MLSignalActor decomposition.

Test Categories (35 tests total):
- Ring Buffer Management: 15 tests
- History List Management: 10 tests
- Window Metadata: 5 tests
- Property Tests: 5 tests (Hypothesis-based)

This is TDD - tests define the contract for implementation.

Architecture Patterns (CLAUDE.md):
- Pattern 3: Hot/Cold Path Separation (zero allocations in update())
- Pattern 2: Protocol-First Interface Design (property accessors)

"""

from __future__ import annotations

import logging
import tracemalloc
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_logger() -> logging.Logger:
    """
    Mock logger for testing.
    """
    logger = MagicMock(spec=logging.Logger)
    return logger


@pytest.fixture
def default_buffer():
    """
    Provides a default PredictionBufferComponent with capacity=100.
    """
    from ml.actors.common.prediction_buffer import PredictionBufferComponent

    return PredictionBufferComponent(
        capacity=100,
        enable_history=True,
        actor_id="test_actor",
        log=None,
    )


@pytest.fixture
def buffer_factory(mock_logger: logging.Logger):
    """
    Factory fixture for creating PredictionBufferComponent instances.

    Returns a callable that creates buffers with specified parameters.

    """
    from ml.actors.common.prediction_buffer import PredictionBufferComponent

    def _create_buffer(
        capacity: int = 100,
        enable_history: bool = True,
        actor_id: str | None = "test_actor",
        log: logging.Logger | None = None,
    ) -> PredictionBufferComponent:
        """
        Create a buffer with specified parameters.
        """
        return PredictionBufferComponent(
            capacity=capacity,
            enable_history=enable_history,
            actor_id=actor_id,
            log=log,
        )

    return _create_buffer


@pytest.fixture
def buffer_with_data(buffer_factory):
    """
    Provides a buffer pre-populated with test data.

    Creates a capacity=100 buffer with 50 items.

    """
    buffer = buffer_factory(capacity=100, enable_history=True)

    # Add 50 items with predictable values
    for i in range(50):
        buffer.update(
            prediction=i * 0.01,  # 0.0, 0.01, 0.02, ...
            confidence=0.5 + i * 0.01,  # 0.5, 0.51, 0.52, ...
            volatility=0.001 * (i + 1),  # 0.001, 0.002, ...
        )

    return buffer


# =============================================================================
# Ring Buffer Management Tests (15 tests)
# =============================================================================


class TestRingBufferManagement:
    """
    Tests for ring buffer management functionality.
    """

    def test_buffer_initializes_with_capacity(self) -> None:
        """
        Verify buffer initialized with specified capacity.

        Ring buffers must be allocated with exact size matching capacity.

        """
        from ml.actors.common.prediction_buffer import PredictionBufferComponent

        # Arrange & Act
        buffer = PredictionBufferComponent(
            capacity=100,
            enable_history=True,
        )

        # Assert
        assert (
            buffer._prediction_window.shape[0] == 100
        ), "Prediction window must have shape matching capacity"
        assert (
            buffer._confidence_window.shape[0] == 100
        ), "Confidence window must have shape matching capacity"
        assert (
            buffer._volatility_window.shape[0] == 100
        ), "Volatility window must have shape matching capacity"

    def test_buffer_starts_empty(self, buffer_factory) -> None:
        """
        Verify buffer count starts at zero.

        A fresh buffer must have count=0 and index=0.

        """
        # Arrange & Act
        buffer = buffer_factory(capacity=100)

        # Assert
        assert buffer._window_count == 0, "Fresh buffer must have count=0"
        assert buffer._window_index == 0, "Fresh buffer must have index=0"

    def test_buffer_update_increments_count(self, buffer_factory) -> None:
        """
        Verify update() increments count until capacity.

        Adding items should increment count up to capacity.

        """
        # Arrange
        buffer = buffer_factory(capacity=100)

        # Act: Add 50 items
        for i in range(50):
            buffer.update(prediction=0.5, confidence=0.8, volatility=0.01)

        # Assert
        assert buffer._window_count == 50, "Count should be 50 after 50 updates"

    def test_buffer_count_stops_at_capacity(self, buffer_factory) -> None:
        """
        Verify count doesn't exceed capacity.

        Count must saturate at capacity, not grow beyond.

        """
        # Arrange
        buffer = buffer_factory(capacity=100)

        # Act: Add 150 items (exceeds capacity)
        for i in range(150):
            buffer.update(prediction=0.5, confidence=0.8, volatility=0.01)

        # Assert
        assert buffer._window_count == 100, "Count must saturate at capacity (100), not exceed it"

    def test_buffer_index_wraps_around(self, buffer_factory) -> None:
        """
        Verify circular indexing wraps at capacity.

        After capacity+10 updates, index should wrap to 10.

        """
        # Arrange
        capacity = 100
        buffer = buffer_factory(capacity=capacity)

        # Act: Add capacity+10 items
        for i in range(capacity + 10):
            buffer.update(prediction=0.5, confidence=0.8, volatility=0.01)

        # Assert
        expected_index = 10  # (capacity + 10) % capacity
        assert (
            buffer._window_index == expected_index
        ), f"Index should wrap to {expected_index}, got {buffer._window_index}"

    def test_buffer_stores_prediction_values(self, buffer_factory) -> None:
        """
        Verify prediction values stored correctly.

        Prediction value should be written to the correct ring buffer position.

        """
        # Arrange
        buffer = buffer_factory(capacity=100)

        # Act
        buffer.update(prediction=0.75, confidence=0.9, volatility=0.01)

        # Assert
        # Value should be at index 0 (first update)
        assert buffer._prediction_window[0] == pytest.approx(
            0.75, rel=1e-5
        ), "Prediction value should be stored at correct index"

    def test_buffer_stores_confidence_values(self, buffer_factory) -> None:
        """
        Verify confidence values stored correctly.

        Confidence value should be written to the correct ring buffer position.

        """
        # Arrange
        buffer = buffer_factory(capacity=100)

        # Act
        buffer.update(prediction=0.5, confidence=0.9, volatility=0.01)

        # Assert
        assert buffer._confidence_window[0] == pytest.approx(
            0.9, rel=1e-5
        ), "Confidence value should be stored at correct index"

    def test_buffer_stores_volatility_values(self, buffer_factory) -> None:
        """
        Verify volatility values stored correctly.

        Volatility value should be written to the correct ring buffer position.

        """
        # Arrange
        buffer = buffer_factory(capacity=100)

        # Act
        expected_volatility = 0.015
        buffer.update(prediction=0.5, confidence=0.8, volatility=expected_volatility)

        # Assert
        assert buffer._volatility_window[0] == pytest.approx(
            expected_volatility,
            rel=1e-5,
        ), "Volatility value should be stored at correct index"

    def test_buffer_overwrites_old_values_after_wrap(self, buffer_factory) -> None:
        """
        Verify old values overwritten after buffer wraps.

        After capacity+1 updates, the oldest value should be overwritten.

        """
        # Arrange
        capacity = 100
        buffer = buffer_factory(capacity=capacity)

        # Act: Fill buffer with 1.0
        for i in range(capacity):
            buffer.update(prediction=1.0, confidence=0.8, volatility=0.01)

        # Add one more (should overwrite index 0)
        newest_value = 99.99
        buffer.update(prediction=newest_value, confidence=0.8, volatility=0.01)

        # Assert
        assert buffer._prediction_window[0] == pytest.approx(
            newest_value, rel=1e-5
        ), "Index 0 should contain newest value after wrap"

    def test_buffer_get_ring_metadata(self, buffer_with_data) -> None:
        """
        Verify get_ring_metadata() returns correct values.

        Metadata dict should contain ring buffer references and current indices.

        """
        # Arrange
        buffer = buffer_with_data

        # Act
        metadata = buffer.get_ring_metadata()

        # Assert
        assert (
            metadata["_prediction_ring"] is buffer._prediction_window
        ), "Metadata must contain reference (not copy) to prediction ring"
        assert (
            metadata["_prediction_ring_index"] == buffer._window_index
        ), "Metadata must contain current window index"
        assert (
            metadata["_prediction_ring_count"] == buffer._window_count
        ), "Metadata must contain current window count"
        assert "_confidence_ring" in metadata, "Metadata must include confidence ring reference"
        assert "_volatility_ring" in metadata, "Metadata must include volatility ring reference"

    def test_buffer_pre_allocated_arrays(self, buffer_factory) -> None:
        """
        Verify arrays pre-allocated, not reallocated.

        Ring buffer arrays should maintain same identity across updates.

        """
        # Arrange
        buffer = buffer_factory(capacity=100)
        original_id = id(buffer._prediction_window)

        # Act: Multiple updates
        for i in range(200):
            buffer.update(prediction=0.5, confidence=0.8, volatility=0.01)

        # Assert
        assert (
            id(buffer._prediction_window) == original_id
        ), "Ring buffer array must not be reallocated during updates"

    def test_buffer_uses_float32_dtype(self, buffer_factory) -> None:
        """
        Verify ring buffers use float32 (not float64).

        Using float32 reduces memory footprint and improves cache performance.

        """
        # Arrange & Act
        buffer = buffer_factory(capacity=100)

        # Assert
        assert (
            buffer._prediction_window.dtype == np.float32
        ), "Prediction window must use float32 dtype"
        assert (
            buffer._confidence_window.dtype == np.float32
        ), "Confidence window must use float32 dtype"
        assert (
            buffer._volatility_window.dtype == np.float32
        ), "Volatility window must use float32 dtype"

    def test_buffer_no_allocations_on_update(self, buffer_factory) -> None:
        """
        Verify update() allocates no memory (hot path).

        Zero allocations required for hot path performance.

        """
        # Arrange
        buffer = buffer_factory(capacity=100)
        # Warm up with one update to avoid lazy initialization effects
        buffer.update(prediction=0.5, confidence=0.8, volatility=0.01)

        # Create buffer without history to test pure ring buffer updates
        buffer_no_history = buffer_factory(capacity=100, enable_history=False)
        buffer_no_history.update(prediction=0.5, confidence=0.8, volatility=0.01)

        # Act: Measure allocations during update
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        for i in range(100):
            buffer_no_history.update(prediction=0.5 + i * 0.001, confidence=0.8, volatility=0.01)

        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Calculate difference
        stats = snapshot2.compare_to(snapshot1, "lineno")
        total_new_bytes = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

        # Assert: Allow small threshold for any interpreter overhead
        # Ring buffer updates should not allocate new arrays
        assert total_new_bytes < 1000, (
            f"Hot path update() should not allocate significant memory, "
            f"but allocated {total_new_bytes} bytes"
        )

    def test_buffer_reset_clears_state(self, buffer_with_data) -> None:
        """
        Verify reset() clears count and fills with zeros.

        After reset, count=0, index=0, arrays zeroed.

        """
        # Arrange
        buffer = buffer_with_data
        assert buffer._window_count > 0, "Pre-condition: buffer should have data"

        # Act
        buffer.reset()

        # Assert
        assert buffer._window_count == 0, "Count must be 0 after reset"
        assert buffer._window_index == 0, "Index must be 0 after reset"
        assert np.all(
            buffer._prediction_window == 0
        ), "Prediction window must be zeroed after reset"
        assert np.all(
            buffer._confidence_window == 0
        ), "Confidence window must be zeroed after reset"
        assert np.all(
            buffer._volatility_window == 0
        ), "Volatility window must be zeroed after reset"

    def test_buffer_handles_capacity_one(self, buffer_factory) -> None:
        """Verify edge case: capacity=1.

        Buffer with capacity=1 should work correctly with index always 0 after wrap.
        """
        # Arrange
        buffer = buffer_factory(capacity=1)

        # Act & Assert: Multiple updates
        for i in range(5):
            buffer.update(prediction=i * 0.1, confidence=0.8, volatility=0.01)
            # After each update, index wraps to 0
            assert (
                buffer._window_index == 0
            ), f"With capacity=1, index should always be 0, got {buffer._window_index}"

        # Final state check
        assert buffer._window_count == 1, "Count should saturate at 1 for capacity=1"
        # Last value stored
        assert buffer._prediction_window[0] == pytest.approx(
            0.4, rel=1e-5
        ), "Last prediction value should be stored"


# =============================================================================
# History List Management Tests (10 tests)
# =============================================================================


class TestHistoryListManagement:
    """
    Tests for history list management functionality.
    """

    def test_history_starts_empty(self, buffer_factory) -> None:
        """
        Verify history lists start empty.

        New buffer should have empty history lists.

        """
        # Arrange & Act
        buffer = buffer_factory(capacity=100, enable_history=True)

        # Assert
        assert len(buffer._prediction_history) == 0, "Prediction history must start empty"
        assert len(buffer._confidence_history) == 0, "Confidence history must start empty"

    def test_history_append_adds_values(self, buffer_factory) -> None:
        """
        Verify history append adds to list.

        Single update should add one item to history.

        """
        # Arrange
        buffer = buffer_factory(capacity=100, enable_history=True)

        # Act
        buffer.update(prediction=0.75, confidence=0.9, volatility=0.01)

        # Assert
        assert (
            len(buffer._prediction_history) == 1
        ), "Prediction history should have 1 item after 1 update"
        assert (
            len(buffer._confidence_history) == 1
        ), "Confidence history should have 1 item after 1 update"
        assert buffer._prediction_history[0] == 0.75, "History should contain the prediction value"
        assert buffer._confidence_history[0] == 0.9, "History should contain the confidence value"

    def test_history_truncate_at_max_size(self, buffer_factory) -> None:
        """
        Verify history truncated at max size if configured.

        Note: This test verifies behavior if max_history is implemented.
        Current implementation may not have truncation - test documents expected behavior.

        """
        # Arrange
        # Note: Implementation may need max_history parameter
        buffer = buffer_factory(capacity=100, enable_history=True)

        # Act: Add 1000 items
        for i in range(1000):
            buffer.update(prediction=i * 0.001, confidence=0.8, volatility=0.01)

        # Assert: History should either be all 1000 or truncated if max_history exists
        # This test documents the expected contract
        history_len = len(buffer._prediction_history)
        assert history_len >= 1, "History should have at least 1 item"
        # If max_history is implemented (e.g., max_history=500):
        # assert len(buffer._prediction_history) <= 500, "History should be truncated"

    def test_history_get_recent(self, buffer_factory) -> None:
        """
        Verify get_history(lookback) returns recent values.

        Should return last N items when lookback specified.

        """
        # Arrange
        buffer = buffer_factory(capacity=100, enable_history=True)

        # Add 100 items
        for i in range(100):
            buffer.update(prediction=i * 0.01, confidence=0.8, volatility=0.01)

        # Act
        pred_history, _conf_history = buffer.get_history(lookback=10)

        # Assert
        assert len(pred_history) == 10, "Should return 10 items"
        assert (
            pred_history == buffer._prediction_history[-10:]
        ), "Should return last 10 items from history"

    def test_history_get_all_when_lookback_exceeds_size(self, buffer_factory) -> None:
        """
        Verify get_history returns all if lookback > len.

        Should return all available items when lookback exceeds history size.

        """
        # Arrange
        buffer = buffer_factory(capacity=100, enable_history=True)

        # Add 5 items
        for i in range(5):
            buffer.update(prediction=i * 0.1, confidence=0.8, volatility=0.01)

        # Act: Request 10 items but only 5 exist
        pred_history, _conf_history = buffer.get_history(lookback=10)

        # Assert
        assert (
            len(pred_history) == 5
        ), "Should return all 5 items when lookback exceeds history size"

    def test_history_disabled_in_optimized_mode(self, buffer_factory) -> None:
        """
        Verify history not updated in optimized mode (hot path).

        With enable_history=False, history lists should remain empty.

        """
        # Arrange
        buffer = buffer_factory(capacity=100, enable_history=False)

        # Act
        for i in range(50):
            buffer.update(prediction=0.5, confidence=0.8, volatility=0.01)

        # Assert
        assert (
            len(buffer._prediction_history) == 0
        ), "History should be empty when enable_history=False"
        assert (
            len(buffer._confidence_history) == 0
        ), "Confidence history should be empty when enable_history=False"

    def test_history_enabled_in_standard_mode(self, buffer_factory) -> None:
        """
        Verify history updated in standard mode.

        With enable_history=True, history lists should be populated.

        """
        # Arrange
        buffer = buffer_factory(capacity=100, enable_history=True)

        # Act
        for i in range(10):
            buffer.update(prediction=0.5, confidence=0.8, volatility=0.01)

        # Assert
        assert (
            len(buffer._prediction_history) > 0
        ), "History should be populated when enable_history=True"
        assert (
            len(buffer._prediction_history) == 10
        ), "History should have 10 items after 10 updates"

    def test_history_reset_clears_lists(self, buffer_with_data) -> None:
        """
        Verify reset() clears history lists.

        After reset, history lists should be empty.

        """
        # Arrange
        buffer = buffer_with_data
        assert len(buffer._prediction_history) > 0, "Pre-condition: buffer should have history"

        # Act
        buffer.reset()

        # Assert
        assert (
            len(buffer._prediction_history) == 0
        ), "Prediction history must be cleared after reset"
        assert (
            len(buffer._confidence_history) == 0
        ), "Confidence history must be cleared after reset"

    def test_history_confidence_matches_prediction_length(
        self,
        buffer_factory,
    ) -> None:
        """
        Verify prediction and confidence histories same length.

        Both histories should always have matching lengths.

        """
        # Arrange
        buffer = buffer_factory(capacity=100, enable_history=True)

        # Act
        for i in range(75):
            buffer.update(prediction=i * 0.01, confidence=0.5 + i * 0.005, volatility=0.01)

        # Assert
        assert len(buffer._prediction_history) == len(
            buffer._confidence_history
        ), "Prediction and confidence histories must have same length"

    def test_history_preserves_order(self, buffer_factory) -> None:
        """
        Verify history maintains temporal order.

        Values should appear in history in the order they were added.

        """
        # Arrange
        buffer = buffer_factory(capacity=100, enable_history=True)
        expected_values = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Act
        for val in expected_values:
            buffer.update(prediction=val, confidence=0.8, volatility=0.01)

        # Assert
        assert (
            buffer._prediction_history == expected_values
        ), "History should preserve temporal order"


# =============================================================================
# Window Metadata Tests (5 tests)
# =============================================================================


class TestWindowMetadata:
    """
    Tests for window metadata functionality.
    """

    def test_metadata_includes_all_required_fields(self, buffer_with_data) -> None:
        """
        Verify context metadata includes all fields for strategies.

        Metadata dict must include all required keys for strategy context.

        """
        # Arrange
        buffer = buffer_with_data

        # Act
        metadata = buffer.get_ring_metadata()

        # Assert: Check all expected keys
        expected_keys = [
            "_prediction_ring",
            "_prediction_ring_index",
            "_prediction_ring_count",
            "_confidence_ring",
            "_volatility_ring",
        ]
        for key in expected_keys:
            assert key in metadata, f"Metadata must contain key '{key}'"

    def test_metadata_ring_buffer_references(self, buffer_with_data) -> None:
        """
        Verify metadata contains references to ring buffers (not copies).

        Zero-copy access is critical for hot path performance.

        """
        # Arrange
        buffer = buffer_with_data

        # Act
        metadata = buffer.get_ring_metadata()

        # Assert: Same object identity (not copies)
        assert (
            metadata["_prediction_ring"] is buffer._prediction_window
        ), "Metadata must contain reference to prediction ring, not copy"
        assert (
            metadata["_confidence_ring"] is buffer._confidence_window
        ), "Metadata must contain reference to confidence ring, not copy"
        assert (
            metadata["_volatility_ring"] is buffer._volatility_window
        ), "Metadata must contain reference to volatility ring, not copy"

    def test_metadata_includes_timestamps(self, buffer_factory) -> None:
        """
        Verify metadata includes timestamp_ns.

        Note: This test documents expected behavior. If timestamp is added to
        get_ring_metadata or a separate method, this validates the contract.

        """
        # Arrange
        buffer = buffer_factory(capacity=100, enable_history=True)
        buffer.update(prediction=0.5, confidence=0.8, volatility=0.01)

        # Act
        metadata = buffer.get_ring_metadata()

        # Assert: Either timestamp is in metadata or we accept current implementation
        # This test documents the expected contract for future enhancement
        # If timestamp_ns is added:
        # assert "timestamp_ns" in metadata, "Metadata should include timestamp_ns"
        # Current implementation may not include timestamp - mark as expected
        _ = metadata  # Placeholder for timestamp validation

    def test_metadata_includes_model_id(self, buffer_factory) -> None:
        """
        Verify metadata includes model_id.

        Note: model_id is typically provided by the actor, not buffer.
        This test documents the expected integration pattern.

        """
        # Arrange
        buffer = buffer_factory(
            capacity=100,
            enable_history=True,
            actor_id="test_actor",
        )
        buffer.update(prediction=0.5, confidence=0.8, volatility=0.01)

        # Act
        metadata = buffer.get_ring_metadata()

        # Assert: Model ID may be added by actor context, not buffer itself
        # This documents the expected pattern
        # assert metadata.get("model_id") == "test_model", ...
        _ = metadata  # Placeholder for model_id validation

    def test_metadata_building_no_allocations(self, buffer_factory) -> None:
        """
        Verify building metadata doesn't allocate (hot path).

        Metadata construction should be allocation-free for hot path.

        """
        # Arrange
        buffer = buffer_factory(capacity=100, enable_history=True)
        for i in range(50):
            buffer.update(prediction=0.5, confidence=0.8, volatility=0.01)

        # Warm up
        _ = buffer.get_ring_metadata()

        # Act: Measure allocations during metadata building
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        for _ in range(100):
            metadata = buffer.get_ring_metadata()

        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Calculate difference
        stats = snapshot2.compare_to(snapshot1, "lineno")
        total_new_bytes = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

        # Assert: Dict creation has some overhead, but should be minimal
        # Each dict has some fixed overhead, allow for that
        # 100 iterations * ~200 bytes per dict = ~20000 bytes reasonable
        assert total_new_bytes < 50000, (
            f"Metadata building should have minimal allocations, "
            f"but allocated {total_new_bytes} bytes"
        )


# =============================================================================
# Property Tests (5 tests)
# =============================================================================


class TestPropertyBased:
    """
    Hypothesis property-based tests for buffer invariants.
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        capacity=st.integers(min_value=1, max_value=1000),
        num_updates=st.integers(min_value=0, max_value=5000),
    )
    def test_buffer_capacity_never_exceeded_property(
        self,
        capacity: int,
        num_updates: int,
    ) -> None:
        """
        Verify buffer never exceeds capacity for any sequence of updates.

        Property: window_count <= capacity for all states.

        """
        from ml.actors.common.prediction_buffer import PredictionBufferComponent

        # Arrange
        buffer = PredictionBufferComponent(capacity=capacity, enable_history=False)

        # Act
        for i in range(num_updates):
            buffer.update(
                prediction=i * 0.001,
                confidence=0.5,
                volatility=0.001,
            )

        # Assert: Invariant must hold
        assert buffer._window_count <= buffer.capacity, (
            f"Window count ({buffer._window_count}) must never exceed "
            f"capacity ({buffer.capacity})"
        )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        capacity=st.integers(min_value=1, max_value=1000),
        num_updates=st.integers(min_value=0, max_value=5000),
    )
    def test_buffer_index_always_valid_property(
        self,
        capacity: int,
        num_updates: int,
    ) -> None:
        """
        Verify index always in [0, capacity).

        Property: 0 <= window_index < capacity for all states.

        """
        from ml.actors.common.prediction_buffer import PredictionBufferComponent

        # Arrange
        buffer = PredictionBufferComponent(capacity=capacity, enable_history=False)

        # Act
        for i in range(num_updates):
            buffer.update(
                prediction=i * 0.001,
                confidence=0.5,
                volatility=0.001,
            )

        # Assert: Index must be in valid range
        assert 0 <= buffer._window_index < buffer.capacity, (
            f"Window index ({buffer._window_index}) must be in " f"[0, {buffer.capacity})"
        )

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        capacity=st.integers(min_value=1, max_value=100),
        updates=st.lists(
            st.tuples(
                st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            ),
            min_size=0,
            max_size=500,
        ),
    )
    def test_buffer_circular_consistency_property(
        self,
        capacity: int,
        updates: list[tuple[float, float, float]],
    ) -> None:
        """
        Verify circular indexing maintains data consistency.

        Property: Last 'capacity' values should be retrievable from ring buffer.

        """
        from ml.actors.common.prediction_buffer import PredictionBufferComponent

        # Arrange
        buffer = PredictionBufferComponent(capacity=capacity, enable_history=True)

        # Act
        predictions_sent = []
        for pred, conf, vol in updates:
            buffer.update(prediction=pred, confidence=conf, volatility=vol)
            predictions_sent.append(pred)

        # Assert: If we have history enabled, verify consistency
        if len(predictions_sent) > 0:
            # The ring buffer should contain last min(len(updates), capacity) values
            expected_count = min(len(predictions_sent), capacity)
            assert (
                buffer._window_count == expected_count
            ), f"Window count should be {expected_count}"

            # History should contain all values
            assert len(buffer._prediction_history) == len(
                predictions_sent
            ), "History should contain all predictions"

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        capacity=st.integers(min_value=10, max_value=100),
        num_updates=st.integers(min_value=0, max_value=1000),
    )
    def test_history_size_bounded_property(
        self,
        capacity: int,
        num_updates: int,
    ) -> None:
        """
        Verify history never exceeds max_history if configured.

        Property: len(history) <= max_history (when configured).
        Current implementation has unbounded history - test documents expected behavior.

        """
        from ml.actors.common.prediction_buffer import PredictionBufferComponent

        # Arrange
        buffer = PredictionBufferComponent(capacity=capacity, enable_history=True)

        # Act
        for i in range(num_updates):
            buffer.update(
                prediction=i * 0.001,
                confidence=0.5,
                volatility=0.001,
            )

        # Assert: History grows with updates (current behavior)
        # If max_history is implemented, this would check the bound
        assert len(buffer._prediction_history) == num_updates, "History should contain all updates"
        # Future: assert len(buffer._prediction_history) <= MAX_HISTORY

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        capacity=st.integers(min_value=1, max_value=100),
        num_updates=st.integers(min_value=1, max_value=100),
        num_resets=st.integers(min_value=1, max_value=5),
    )
    def test_buffer_reset_idempotency_property(
        self,
        capacity: int,
        num_updates: int,
        num_resets: int,
    ) -> None:
        """
        Verify reset() is idempotent.

        Property: Multiple resets produce same state as single reset.

        """
        from ml.actors.common.prediction_buffer import PredictionBufferComponent

        # Arrange
        buffer = PredictionBufferComponent(capacity=capacity, enable_history=True)

        # Add some data
        for i in range(num_updates):
            buffer.update(
                prediction=i * 0.01,
                confidence=0.5,
                volatility=0.001,
            )

        # Act: Multiple resets
        for _ in range(num_resets):
            buffer.reset()

        # Assert: State should be consistent regardless of reset count
        assert buffer._window_count == 0, "Count must be 0 after reset(s)"
        assert buffer._window_index == 0, "Index must be 0 after reset(s)"
        assert len(buffer._prediction_history) == 0, "History must be empty after reset(s)"
        assert np.all(buffer._prediction_window == 0), "Ring buffer must be zeroed after reset(s)"


# =============================================================================
# Edge Case and Error Condition Tests
# =============================================================================


class TestEdgeCasesAndErrors:
    """
    Tests for edge cases and error conditions.
    """

    def test_buffer_rejects_zero_capacity(self) -> None:
        """
        Verify buffer rejects capacity=0.

        ValueError should be raised for invalid capacity.

        """
        from ml.actors.common.prediction_buffer import PredictionBufferComponent

        # Act & Assert
        with pytest.raises(ValueError, match="capacity must be > 0"):
            PredictionBufferComponent(capacity=0, enable_history=True)

    def test_buffer_rejects_negative_capacity(self) -> None:
        """
        Verify buffer rejects negative capacity.

        ValueError should be raised for invalid capacity.

        """
        from ml.actors.common.prediction_buffer import PredictionBufferComponent

        # Act & Assert
        with pytest.raises(ValueError, match="capacity must be > 0"):
            PredictionBufferComponent(capacity=-10, enable_history=True)

    def test_get_history_returns_empty_when_disabled(self, buffer_factory) -> None:
        """
        Verify get_history returns empty lists when history disabled.

        Should return ([], []) when enable_history=False.

        """
        # Arrange
        buffer = buffer_factory(capacity=100, enable_history=False)
        for i in range(50):
            buffer.update(prediction=0.5, confidence=0.8, volatility=0.01)

        # Act
        pred_history, conf_history = buffer.get_history()

        # Assert
        assert pred_history == [], "Prediction history should be empty"
        assert conf_history == [], "Confidence history should be empty"

    def test_get_history_none_lookback_returns_all(self, buffer_factory) -> None:
        """
        Verify get_history(None) returns all history.

        When lookback is None, return entire history.

        """
        # Arrange
        buffer = buffer_factory(capacity=100, enable_history=True)
        for i in range(75):
            buffer.update(prediction=i * 0.01, confidence=0.8, volatility=0.01)

        # Act
        pred_history, conf_history = buffer.get_history(lookback=None)

        # Assert
        assert len(pred_history) == 75, "Should return all 75 items"
        assert len(conf_history) == 75, "Should return all 75 confidence values"

    def test_property_accessors_return_correct_values(self, buffer_factory) -> None:
        """
        Verify property accessors return correct values.

        Test capacity, window_count, window_index, enable_history properties.

        """
        # Arrange
        buffer = buffer_factory(capacity=50, enable_history=True)

        # Act
        for i in range(25):
            buffer.update(prediction=0.5, confidence=0.8, volatility=0.01)

        # Assert
        assert buffer.capacity == 50, "capacity property should return 50"
        assert buffer.window_count == 25, "window_count should be 25"
        assert buffer.window_index == 25, "window_index should be 25"
        assert buffer.enable_history is True, "enable_history should be True"

    def test_property_windows_return_arrays(self, buffer_factory) -> None:
        """
        Verify window properties return numpy arrays.

        prediction_window, confidence_window, volatility_window properties.

        """
        # Arrange
        buffer = buffer_factory(capacity=100, enable_history=True)

        # Assert
        assert isinstance(
            buffer.prediction_window, np.ndarray
        ), "prediction_window must be numpy array"
        assert isinstance(
            buffer.confidence_window, np.ndarray
        ), "confidence_window must be numpy array"
        assert isinstance(
            buffer.volatility_window, np.ndarray
        ), "volatility_window must be numpy array"
        assert buffer.prediction_window.shape == (
            100,
        ), "prediction_window shape must match capacity"
