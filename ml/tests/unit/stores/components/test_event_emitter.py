#!/usr/bin/env python3

"""
Unit tests for EventEmitterComponent.

Tests event emission, message bus integration, and non-blocking behavior
for DataStore event tracking.

Phase 2.4.4 - EventEmitterComponent extraction and testing.

"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import call

import pytest

from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.stores.components.event_emitter import EventEmitterComponent


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def mock_registry() -> MagicMock:
    """
    Create mock data registry for event persistence.
    """
    registry = MagicMock()
    registry.emit_event = MagicMock()
    return registry


@pytest.fixture
def mock_publisher() -> MagicMock:
    """
    Create mock message bus publisher.
    """
    publisher = MagicMock()
    publisher.publish = MagicMock()
    return publisher


@pytest.fixture
def event_emitter(mock_registry: MagicMock, mock_publisher: MagicMock) -> EventEmitterComponent:
    """
    Create EventEmitterComponent with mock dependencies.
    """
    return EventEmitterComponent(
        registry=mock_registry,
        publisher=mock_publisher,
        enable_publishing=True,
        topic_scheme="hierarchical",
        topic_prefix="ml",
    )


@pytest.fixture
def event_emitter_no_publisher(mock_registry: MagicMock) -> EventEmitterComponent:
    """
    Create EventEmitterComponent without message bus publisher.
    """
    return EventEmitterComponent(
        registry=mock_registry,
        publisher=None,
        enable_publishing=False,
        topic_scheme="hierarchical",
        topic_prefix="ml",
    )


# =========================================================================
# Test: emit_event()
# =========================================================================


def test_emit_event_success(
    event_emitter: EventEmitterComponent,
    mock_registry: MagicMock,
    mock_publisher: MagicMock,
) -> None:
    """
    Test successful event emission to registry and message bus.

    Verifies:
    - Event persisted to registry with correct parameters
    - Event published to message bus with correct topic
    - Correlation ID generated and included in metadata
    """
    # Execute
    event_emitter.emit_event(
        dataset_id="bars_eurusd_1m",
        instrument_id="EURUSD.SIM",
        stage=Stage.DATA_INGESTED,
        source=Source.HISTORICAL,
        run_id="run_20240101_120000",
        ts_min=1699999900000000000,
        ts_max=1699999990000000000,
        count=100,
        status="success",
        metadata={"quality_score": 1.0},
    )

    # Verify registry.emit_event called with correct args
    assert mock_registry.emit_event.called
    call_kwargs = mock_registry.emit_event.call_args.kwargs
    assert call_kwargs["dataset_id"] == "bars_eurusd_1m"
    assert call_kwargs["instrument_id"] == "EURUSD.SIM"
    assert call_kwargs["stage"] == Stage.DATA_INGESTED
    assert call_kwargs["source"] == Source.HISTORICAL
    assert call_kwargs["run_id"] == "run_20240101_120000"
    assert call_kwargs["ts_min"] == 1699999900000000000
    assert call_kwargs["ts_max"] == 1699999990000000000
    assert call_kwargs["count"] == 100
    assert call_kwargs["status"] == EventStatus.SUCCESS
    assert call_kwargs["error"] is None
    assert "correlation_id" in call_kwargs["metadata"]
    assert call_kwargs["metadata"]["quality_score"] == 1.0

    # Verify message bus publish called
    assert mock_publisher.publish.called
    topic, payload = mock_publisher.publish.call_args[0]
    assert "eurusd" in topic.lower() or "EURUSD" in topic
    assert payload["dataset_id"] == "bars_eurusd_1m"
    assert payload["status"] == "success"


def test_emit_event_with_invalid_status(
    event_emitter: EventEmitterComponent,
    mock_registry: MagicMock,
) -> None:
    """
    Test event emission with string status (normalized to enum).

    Verifies:
    - String status values are converted to EventStatus enum
    - Event emission succeeds with normalized status
    """
    # Execute with string status
    event_emitter.emit_event(
        dataset_id="bars_eurusd_1m",
        instrument_id="EURUSD.SIM",
        stage="DATA_INGESTED",
        source="HISTORICAL",
        run_id="run_test",
        ts_min=1699999900000000000,
        ts_max=1699999990000000000,
        count=50,
        status="success",  # String, not enum
    )

    # Verify registry.emit_event called with EventStatus enum
    assert mock_registry.emit_event.called
    call_kwargs = mock_registry.emit_event.call_args.kwargs
    assert call_kwargs["status"] == EventStatus.SUCCESS


def test_emit_event_registry_failure_non_blocking(
    mock_registry: MagicMock,
    mock_publisher: MagicMock,
) -> None:
    """
    Test that registry failures don't crash event emission (non-blocking).

    Verifies:
    - Registry emit_event raises exception
    - Exception is caught and logged (not raised)
    - Message bus publish still attempted
    """
    # Setup registry to raise exception
    mock_registry.emit_event.side_effect = RuntimeError("Registry unavailable")

    emitter = EventEmitterComponent(
        registry=mock_registry,
        publisher=mock_publisher,
        enable_publishing=True,
    )

    # Execute - should NOT raise despite registry failure
    emitter.emit_event(
        dataset_id="bars_eurusd_1m",
        instrument_id="EURUSD.SIM",
        stage=Stage.DATA_INGESTED,
        source=Source.HISTORICAL,
        run_id="run_test",
        ts_min=1699999900000000000,
        ts_max=1699999990000000000,
        count=50,
        status="success",
    )

    # Verify registry.emit_event was called (and failed)
    assert mock_registry.emit_event.called

    # Verify message bus publish still attempted
    assert mock_publisher.publish.called


def test_emit_event_bus_failure_non_blocking(
    event_emitter: EventEmitterComponent,
    mock_registry: MagicMock,
    mock_publisher: MagicMock,
) -> None:
    """
    Test that message bus failures don't crash event emission (non-blocking).

    Verifies:
    - Message bus publish raises exception
    - Exception is caught and logged (not raised)
    - Registry event persisted successfully
    """
    # Setup publisher to raise exception
    mock_publisher.publish.side_effect = RuntimeError("Bus unavailable")

    # Execute - should NOT raise despite bus failure
    event_emitter.emit_event(
        dataset_id="bars_eurusd_1m",
        instrument_id="EURUSD.SIM",
        stage=Stage.DATA_INGESTED,
        source=Source.HISTORICAL,
        run_id="run_test",
        ts_min=1699999900000000000,
        ts_max=1699999990000000000,
        count=50,
        status="success",
    )

    # Verify registry.emit_event succeeded
    assert mock_registry.emit_event.called

    # Verify publisher.publish was called (and failed)
    assert mock_publisher.publish.called


# =========================================================================
# Test: emit_dataset_event()
# =========================================================================


def test_emit_dataset_event_success(
    event_emitter: EventEmitterComponent,
    mock_registry: MagicMock,
    mock_publisher: MagicMock,
) -> None:
    """
    Test dataset-specific event emission.

    Verifies:
    - Event emitted with simplified parameter set
    - Status normalized to EventStatus enum
    - Metadata passed through correctly
    """
    # Execute
    event_emitter.emit_dataset_event(
        dataset_id="bars_eurusd_1m",
        status=EventStatus.SUCCESS,
        metadata={"quality_score": 1.0},
    )

    # Verify registry.emit_event called
    assert mock_registry.emit_event.called
    call_kwargs = mock_registry.emit_event.call_args.kwargs
    assert call_kwargs["dataset_id"] == "bars_eurusd_1m"
    assert call_kwargs["status"] == EventStatus.SUCCESS
    assert "quality_score" in call_kwargs["metadata"]


def test_emit_dataset_event_with_missing_metadata(
    event_emitter: EventEmitterComponent,
    mock_registry: MagicMock,
) -> None:
    """
    Test dataset event emission without metadata (optional parameter).

    Verifies:
    - Event emission succeeds with None metadata
    - Default values used for missing parameters
    """
    # Execute without metadata
    event_emitter.emit_dataset_event(
        dataset_id="bars_eurusd_1m",
        status=EventStatus.SUCCESS,
    )

    # Verify registry.emit_event called
    assert mock_registry.emit_event.called
    call_kwargs = mock_registry.emit_event.call_args.kwargs
    assert call_kwargs["dataset_id"] == "bars_eurusd_1m"
    assert call_kwargs["status"] == EventStatus.SUCCESS
    # Metadata should have correlation_id at minimum
    assert "correlation_id" in call_kwargs["metadata"]


# =========================================================================
# Test: _emit_partial_event()
# =========================================================================


def test_emit_partial_event_success(
    event_emitter: EventEmitterComponent,
    mock_registry: MagicMock,
) -> None:
    """
    Test partial success event emission.

    Verifies:
    - Partial event emitted with correct status
    - Details included in metadata
    - Reason field extracted and included
    """
    # Execute
    event_emitter._emit_partial_event(
        operation="write_ingestion",
        details={
            "dataset_id": "bars_eurusd_1m",
            "instrument_id": "EURUSD.SIM",
            "run_id": "run_test",
            "ts_min": 1699999900000000000,
            "ts_max": 1699999990000000000,
            "records_written": 90,
            "records_failed": 10,
            "reason": "validation_failed",
        },
    )

    # Verify registry.emit_event called with PARTIAL status
    assert mock_registry.emit_event.called
    call_kwargs = mock_registry.emit_event.call_args.kwargs
    assert call_kwargs["dataset_id"] == "bars_eurusd_1m"
    assert call_kwargs["status"] == EventStatus.PARTIAL
    assert call_kwargs["metadata"]["operation"] == "write_ingestion"
    assert call_kwargs["metadata"]["reason"] == "validation_failed"
    assert call_kwargs["metadata"]["records_written"] == 90
    assert call_kwargs["metadata"]["records_failed"] == 10


def test_emit_partial_event_with_empty_details(
    event_emitter: EventEmitterComponent,
    mock_registry: MagicMock,
) -> None:
    """
    Test partial event emission with minimal details (defaults used).

    Verifies:
    - Event emission succeeds with empty details dict
    - Default values used for missing fields (UNKNOWN, current timestamp, etc.)
    """
    # Execute with minimal details
    event_emitter._emit_partial_event(
        operation="write_ingestion",
        details={},  # Empty details
    )

    # Verify registry.emit_event called with defaults
    assert mock_registry.emit_event.called
    call_kwargs = mock_registry.emit_event.call_args.kwargs
    assert call_kwargs["dataset_id"] == "UNKNOWN"
    assert call_kwargs["instrument_id"] == "UNKNOWN"
    assert call_kwargs["status"] == EventStatus.PARTIAL
    assert call_kwargs["metadata"]["operation"] == "write_ingestion"


def test_emit_partial_event_non_blocking(
    mock_registry: MagicMock,
    mock_publisher: MagicMock,
) -> None:
    """
    Test that partial event emission failures don't crash (non-blocking).

    Verifies:
    - Registry emit_event raises exception
    - Exception is caught and logged (not raised)
    """
    # Setup registry to raise exception
    mock_registry.emit_event.side_effect = RuntimeError("Registry unavailable")

    emitter = EventEmitterComponent(
        registry=mock_registry,
        publisher=mock_publisher,
    )

    # Execute - should NOT raise despite registry failure
    emitter._emit_partial_event(
        operation="write_ingestion",
        details={"dataset_id": "test"},
    )

    # Verify registry.emit_event was called (and failed)
    assert mock_registry.emit_event.called


# =========================================================================
# Test: _emit_failed_event()
# =========================================================================


def test_emit_failed_event_success(
    event_emitter: EventEmitterComponent,
    mock_registry: MagicMock,
) -> None:
    """
    Test failure event emission.

    Verifies:
    - Failed event emitted with FAILED status
    - Error message extracted from exception
    - Context details included in metadata
    """
    # Create exception
    error = ValueError("Validation failed: negative close price")

    # Execute
    event_emitter._emit_failed_event(
        operation="write_ingestion",
        error=error,
        context={
            "dataset_id": "bars_eurusd_1m",
            "instrument_id": "EURUSD.SIM",
            "run_id": "run_test",
            "ts_min": 1699999900000000000,
            "ts_max": 1699999990000000000,
            "count": 0,
        },
    )

    # Verify registry.emit_event called with FAILED status
    assert mock_registry.emit_event.called
    call_kwargs = mock_registry.emit_event.call_args.kwargs
    assert call_kwargs["dataset_id"] == "bars_eurusd_1m"
    assert call_kwargs["status"] == EventStatus.FAILED
    assert "Validation failed" in call_kwargs["error"]
    assert call_kwargs["metadata"]["operation"] == "write_ingestion"
    assert call_kwargs["metadata"]["error_type"] == "ValueError"


def test_emit_failed_event_with_nested_exception(
    event_emitter: EventEmitterComponent,
    mock_registry: MagicMock,
) -> None:
    """
    Test failure event with nested exception (chained __cause__).

    Verifies:
    - Nested exception message included in error string
    - Both outer and inner exception details captured
    """
    # Create nested exception
    inner_exc = RuntimeError("Database connection lost")
    outer_exc = ValueError("Write failed")
    outer_exc.__cause__ = inner_exc

    # Execute
    event_emitter._emit_failed_event(
        operation="write_ingestion",
        error=outer_exc,
        context={"dataset_id": "bars_eurusd_1m"},
    )

    # Verify registry.emit_event called with nested error
    assert mock_registry.emit_event.called
    call_kwargs = mock_registry.emit_event.call_args.kwargs
    assert call_kwargs["status"] == EventStatus.FAILED
    # Error message should include both outer and inner exceptions
    assert "Write failed" in call_kwargs["error"]
    assert "Database connection lost" in call_kwargs["error"]


def test_emit_failed_event_non_blocking(
    mock_registry: MagicMock,
    mock_publisher: MagicMock,
) -> None:
    """
    Test that failed event emission failures don't crash (non-blocking).

    Verifies:
    - Registry emit_event raises exception
    - Exception is caught and logged (not raised)
    """
    # Setup registry to raise exception
    mock_registry.emit_event.side_effect = RuntimeError("Registry unavailable")

    emitter = EventEmitterComponent(
        registry=mock_registry,
        publisher=mock_publisher,
    )

    error = ValueError("Test error")

    # Execute - should NOT raise despite registry failure
    emitter._emit_failed_event(
        operation="write_ingestion",
        error=error,
        context={"dataset_id": "test"},
    )

    # Verify registry.emit_event was called (and failed)
    assert mock_registry.emit_event.called


# =========================================================================
# Test: Message Bus Integration
# =========================================================================


def test_emit_event_without_publisher(
    event_emitter_no_publisher: EventEmitterComponent,
    mock_registry: MagicMock,
) -> None:
    """
    Test event emission when publisher is None (registry-only mode).

    Verifies:
    - Event persisted to registry
    - No message bus publish attempted
    - No exceptions raised
    """
    # Execute
    event_emitter_no_publisher.emit_event(
        dataset_id="bars_eurusd_1m",
        instrument_id="EURUSD.SIM",
        stage=Stage.DATA_INGESTED,
        source=Source.HISTORICAL,
        run_id="run_test",
        ts_min=1699999900000000000,
        ts_max=1699999990000000000,
        count=50,
        status="success",
    )

    # Verify registry.emit_event called
    assert mock_registry.emit_event.called

    # Verify no publisher attribute accessed (would raise AttributeError)
    # If code tried to call publisher.publish(), test would fail


def test_emit_event_topic_routing(
    event_emitter: EventEmitterComponent,
    mock_publisher: MagicMock,
) -> None:
    """
    Test message bus topic routing for different instruments.

    Verifies:
    - Topic includes instrument identifier
    - Topic follows hierarchical scheme
    - Topic includes prefix
    """
    # Execute with different instrument
    event_emitter.emit_event(
        dataset_id="bars_gbpusd_1m",
        instrument_id="GBPUSD.SIM",
        stage=Stage.DATA_INGESTED,
        source=Source.HISTORICAL,
        run_id="run_test",
        ts_min=1699999900000000000,
        ts_max=1699999990000000000,
        count=50,
        status="success",
    )

    # Verify topic routing
    assert mock_publisher.publish.called
    topic, _payload = mock_publisher.publish.call_args[0]
    # Topic should include instrument (case-insensitive)
    assert "gbpusd" in topic.lower() or "GBPUSD" in topic


# =========================================================================
# Test: Backward Compatibility
# =========================================================================


def test_emit_event_backward_compatible_registry(
    mock_publisher: MagicMock,
) -> None:
    """
    Test backward compatibility with registries that don't accept metadata.

    Verifies:
    - First emit_event call with metadata raises TypeError
    - Fallback to emit_event without metadata succeeds
    - No exceptions raised to caller
    """
    # Create mock registry that rejects metadata parameter
    mock_registry = MagicMock()
    mock_registry.emit_event.side_effect = [
        TypeError("unexpected keyword argument 'metadata'"),
        None,  # Second call succeeds
    ]

    emitter = EventEmitterComponent(
        registry=mock_registry,
        publisher=mock_publisher,
        enable_publishing=False,
    )

    # Execute - should fallback gracefully
    emitter.emit_event(
        dataset_id="bars_eurusd_1m",
        instrument_id="EURUSD.SIM",
        stage=Stage.DATA_INGESTED,
        source=Source.HISTORICAL,
        run_id="run_test",
        ts_min=1699999900000000000,
        ts_max=1699999990000000000,
        count=50,
        status="success",
        metadata={"quality_score": 1.0},
    )

    # Verify registry.emit_event called twice (first with metadata, second without)
    assert mock_registry.emit_event.call_count == 2
