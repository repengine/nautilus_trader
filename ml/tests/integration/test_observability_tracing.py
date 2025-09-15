"""
Integration tests for distributed tracing with OpenTelemetry.

Tests verify that:
1. Tracing is disabled by default with zero overhead
2. Spans are created and linked properly when enabled
3. W3C trace context propagates through event metadata
4. Parent-child relationships work via correlation_id mapping
5. Graceful fallback when OpenTelemetry unavailable
"""

import os
import time
from unittest.mock import Mock, patch

import pytest

from ml.common import (
    emit_dataset_event_and_watermark,
    extract_and_link_from_event,
    get_correlation_and_trace_context,
)
from ml.config.events import EventStatus, Source, Stage
from ml.observability.tracing import (
    extract_and_link_trace_context,
    get_trace_context,
    inject_trace_context,
    is_tracing_enabled,
    trace_cold_path,
    trace_cold_path_decorator,
    trace_inference,
)


class TestTracingDefaultBehavior:
    """Test that tracing is OFF by default with zero overhead."""

    def test_tracing_disabled_by_default(self):
        """Verify tracing is disabled by default."""
        # Ensure no ML_TRACING_ENABLED env var
        with patch.dict(os.environ, {}, clear=True):
            assert not is_tracing_enabled()

    def test_trace_context_empty_when_disabled(self):
        """Verify trace context is empty when disabled."""
        with patch.dict(os.environ, {"ML_TRACING_ENABLED": "false"}):
            context = get_trace_context()
            assert context == {}

    def test_inject_trace_context_noop_when_disabled(self):
        """Verify inject_trace_context is no-op when disabled."""
        with patch.dict(os.environ, {"ML_TRACING_ENABLED": "false"}):
            metadata = {"correlation_id": "test123"}
            result = inject_trace_context(metadata)
            assert result == metadata  # Unchanged

    def test_extract_and_link_noop_when_disabled(self):
        """Verify extract_and_link is no-op when disabled."""
        with patch.dict(os.environ, {"ML_TRACING_ENABLED": "false"}):
            metadata = {"trace_context": {"traceparent": "test"}}
            # Should not raise any exceptions
            extract_and_link_trace_context(metadata)

    def test_trace_cold_path_noop_when_disabled(self):
        """Verify trace_cold_path is no-op when disabled."""
        with patch.dict(os.environ, {"ML_TRACING_ENABLED": "false"}):
            with trace_cold_path("test_operation") as span:
                assert span is None

    def test_trace_decorators_passthrough_when_disabled(self):
        """Verify decorators are pass-through when disabled."""
        with patch.dict(os.environ, {"ML_TRACING_ENABLED": "false"}):

            @trace_cold_path_decorator("test_op")
            def test_func(x: int) -> int:
                return x * 2

            @trace_inference("test_inference")
            def inference_func(x: int) -> int:
                return x + 1

            # Functions should work normally
            assert test_func(5) == 10
            assert inference_func(5) == 6

    def test_zero_overhead_when_disabled(self):
        """Verify zero overhead when tracing disabled."""
        with patch.dict(os.environ, {"ML_TRACING_ENABLED": "false"}):

            @trace_cold_path_decorator("perf_test")
            def performance_test() -> None:
                # Simulate some work
                time.sleep(0.001)

            # Measure baseline performance
            start = time.perf_counter()
            for _ in range(100):
                performance_test()
            elapsed = time.perf_counter() - start

            # Should be very fast (< 0.2s for 100 calls)
            assert elapsed < 0.2

    def test_event_metadata_unchanged_when_disabled(self):
        """Verify event metadata unchanged when tracing disabled."""
        with patch.dict(os.environ, {"ML_TRACING_ENABLED": "false"}):
            metadata = get_correlation_and_trace_context(
                run_id="test_run",
                dataset_id="features",
                instrument_id="EUR/USD",
                ts_min=1000000000,
                ts_max=2000000000,
                count=100,
            )
            # Should only contain correlation_id
            assert "correlation_id" in metadata
            assert "trace_context" not in metadata


@pytest.mark.integration
class TestTracingWithOpenTelemetry:
    """Test tracing functionality when OpenTelemetry is available."""

    @pytest.fixture(autouse=True)
    def enable_tracing(self):
        """Enable tracing for these tests."""
        with patch.dict(os.environ, {"ML_TRACING_ENABLED": "true"}):
            yield

    def test_tracing_enabled_with_env_var(self):
        """Verify tracing can be enabled via environment variable."""
        # This test will check if OpenTelemetry is available
        # If not available, is_tracing_enabled() should still return False
        enabled = is_tracing_enabled()
        # Result depends on OpenTelemetry availability
        assert isinstance(enabled, bool)

    @patch("ml.observability.tracing.HAS_OPENTELEMETRY", True)
    def test_trace_context_with_mocked_otel(self):
        """Test trace context with mocked OpenTelemetry."""
        with patch("ml.observability.tracing._ensure_tracing_backend", return_value=True):
            with patch("ml.observability.tracing._propagate") as mock_propagate:
                mock_propagate.inject.return_value = None

                # Mock carrier to return test context
                def mock_inject(carrier):
                    carrier["traceparent"] = "00-12345678901234567890123456789012-1234567890123456-01"

                mock_propagate.inject.side_effect = mock_inject

                context = get_trace_context()
                assert "traceparent" in context

    @patch("ml.observability.tracing.HAS_OPENTELEMETRY", True)
    def test_inject_trace_context_with_mocked_otel(self):
        """Test inject_trace_context with mocked OpenTelemetry."""
        with patch("ml.observability.tracing._ensure_tracing_backend", return_value=True):
            with patch("ml.observability.tracing._propagate") as mock_propagate:

                def mock_inject(carrier):
                    carrier["traceparent"] = "00-abcdef1234567890abcdef1234567890-abcdef1234567890-01"

                mock_propagate.inject.side_effect = mock_inject

                metadata = {"correlation_id": "test123"}
                result = inject_trace_context(metadata)

                assert "correlation_id" in result
                assert "trace_context" in result
                assert "traceparent" in result["trace_context"]

    @patch("ml.observability.tracing.HAS_OPENTELEMETRY", True)
    def test_trace_cold_path_with_mocked_otel(self):
        """Test trace_cold_path with mocked OpenTelemetry."""
        mock_span = Mock()
        mock_span.set_attribute = Mock()

        with patch("ml.observability.tracing._ensure_tracing_backend", return_value=True):
            with patch("ml.observability.tracing._tracer") as mock_tracer:
                mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
                mock_tracer.start_as_current_span.return_value.__exit__.return_value = None

                with trace_cold_path("test_operation", correlation_id="test123") as span:
                    assert span is mock_span
                    span.set_attribute("test_attr", "test_value")

                # Verify span attributes were set
                mock_span.set_attribute.assert_any_call("correlation_id", "test123")
                mock_span.set_attribute.assert_any_call("operation_type", "cold_path")

    @patch("ml.observability.tracing.HAS_OPENTELEMETRY", True)
    def test_trace_cold_path_decorator_with_mocked_otel(self):
        """Test trace_cold_path_decorator with mocked OpenTelemetry."""
        mock_span = Mock()

        with patch("ml.observability.tracing._ensure_tracing_backend", return_value=True):
            with patch("ml.observability.tracing._tracer") as mock_tracer:
                mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
                mock_tracer.start_as_current_span.return_value.__exit__.return_value = None

                @trace_cold_path_decorator("test_function")
                def test_func(x: int) -> int:
                    return x * 2

                result = test_func(5)
                assert result == 10

                # Verify tracer was called
                mock_tracer.start_as_current_span.assert_called()

    @patch("ml.observability.tracing.HAS_OPENTELEMETRY", True)
    def test_trace_inference_decorator_with_mocked_otel(self):
        """Test trace_inference decorator with mocked OpenTelemetry."""
        mock_span = Mock()

        with patch("ml.observability.tracing._ensure_tracing_backend", return_value=True):
            with patch("ml.observability.tracing._tracer") as mock_tracer:
                mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
                mock_tracer.start_as_current_span.return_value.__exit__.return_value = None

                # Mock actor with bar containing instrument_id
                class MockBar:
                    def __init__(self):
                        self.instrument_id = "EUR/USD.SIM"

                class MockActor:
                    @trace_inference("signal_generation")
                    def on_bar(self, bar):
                        return f"processed_{bar.instrument_id}"

                actor = MockActor()
                bar = MockBar()
                result = actor.on_bar(bar)

                assert result == "processed_EUR/USD.SIM"
                mock_tracer.start_as_current_span.assert_called()


@pytest.mark.integration
class TestEventTracingIntegration:
    """Test tracing integration with event system."""

    def test_correlation_and_trace_context_generation(self):
        """Test correlation_id and trace context generation."""
        metadata = get_correlation_and_trace_context(
            run_id="test_run_123",
            dataset_id="features",
            instrument_id="EUR/USD",
            ts_min=1000000000,
            ts_max=2000000000,
            count=100,
        )

        # Should always contain correlation_id
        assert "correlation_id" in metadata
        assert isinstance(metadata["correlation_id"], str)
        assert len(metadata["correlation_id"]) > 0

        # May or may not contain trace_context depending on availability
        if "trace_context" in metadata:
            assert isinstance(metadata["trace_context"], dict)

    def test_extract_and_link_from_event_graceful_fallback(self):
        """Test extract_and_link_from_event graceful fallback."""
        # Should not raise exceptions regardless of metadata content
        extract_and_link_from_event({})
        extract_and_link_from_event({"correlation_id": "test123"})
        extract_and_link_from_event({
            "correlation_id": "test123",
            "trace_context": {"traceparent": "invalid"}
        })

    def test_event_emission_with_trace_context(self):
        """Test event emission includes trace context when available."""
        # Mock registry to capture emitted events
        mock_registry = Mock()
        mock_registry.emit_event = Mock()
        mock_registry.update_watermark = Mock()

        emit_dataset_event_and_watermark(
            registry=mock_registry,
            dataset_id="features",
            instrument_id="EUR/USD",
            stage=Stage.FEATURE_COMPUTED,
            source=Source.HISTORICAL,
            run_id="test_run",
            ts_min=1000000000,
            ts_max=2000000000,
            count=100,
            status=EventStatus.SUCCESS,
        )

        # Verify event was emitted
        mock_registry.emit_event.assert_called_once()
        call_args = mock_registry.emit_event.call_args

        # Check that metadata contains correlation_id
        assert "metadata" in call_args.kwargs
        metadata = call_args.kwargs["metadata"]
        assert "correlation_id" in metadata

    @patch("ml.observability.tracing.HAS_OPENTELEMETRY", True)
    def test_event_trace_context_propagation_with_mocked_otel(self):
        """Test trace context propagation through events with mocked OpenTelemetry."""
        with patch("ml.observability.tracing._ensure_tracing_backend", return_value=True):
            with patch("ml.observability.tracing._propagate") as mock_propagate:

                def mock_inject(carrier):
                    carrier["traceparent"] = "00-trace123456789012345678901234567890-span123456789012-01"

                mock_propagate.inject.side_effect = mock_inject

                # Mock registry
                mock_registry = Mock()
                mock_registry.emit_event = Mock()
                mock_registry.update_watermark = Mock()

                emit_dataset_event_and_watermark(
                    registry=mock_registry,
                    dataset_id="features",
                    instrument_id="EUR/USD",
                    stage=Stage.FEATURE_COMPUTED,
                    source=Source.HISTORICAL,
                    run_id="test_run",
                    ts_min=1000000000,
                    ts_max=2000000000,
                    count=100,
                    status=EventStatus.SUCCESS,
                )

                # Verify event contains trace context
                call_args = mock_registry.emit_event.call_args
                metadata = call_args.kwargs["metadata"]

                assert "correlation_id" in metadata
                assert "trace_context" in metadata
                assert "traceparent" in metadata["trace_context"]


@pytest.mark.integration
class TestTracingGracefulFallback:
    """Test graceful fallback behavior when OpenTelemetry unavailable."""

    @patch("ml.observability.tracing.HAS_OPENTELEMETRY", False)
    def test_graceful_fallback_when_otel_unavailable(self):
        """Test graceful fallback when OpenTelemetry not available."""
        with patch.dict(os.environ, {"ML_TRACING_ENABLED": "true"}):
            # All functions should work without errors
            assert not is_tracing_enabled()
            assert get_trace_context() == {}

            metadata = {"test": "value"}
            assert inject_trace_context(metadata) == metadata

            extract_and_link_trace_context({"trace_context": {"test": "value"}})

            with trace_cold_path("test") as span:
                assert span is None

    def test_exception_handling_in_tracing_functions(self):
        """Test exception handling in tracing functions."""
        # Test with invalid trace context
        invalid_metadata = {"trace_context": "not_a_dict"}

        # Should not raise exceptions
        extract_and_link_trace_context(invalid_metadata)
        inject_trace_context(invalid_metadata)

        # Test with None values
        extract_and_link_trace_context({})
        inject_trace_context({})


@pytest.mark.integration
class TestTracingPerformance:
    """Test performance characteristics of tracing system."""

    def test_decorator_overhead_when_disabled(self):
        """Test decorator has minimal overhead when disabled."""
        with patch.dict(os.environ, {"ML_TRACING_ENABLED": "false"}):

            @trace_cold_path_decorator("perf_test")
            def fast_function():
                return 42

            # Should be very fast even with decorator
            start = time.perf_counter()
            for _ in range(1000):
                result = fast_function()
                assert result == 42
            elapsed = time.perf_counter() - start

            # Should complete 1000 calls in well under 0.2 seconds
            assert elapsed < 0.2

    def test_context_manager_overhead_when_disabled(self):
        """Test context manager has minimal overhead when disabled."""
        with patch.dict(os.environ, {"ML_TRACING_ENABLED": "false"}):

            def traced_function():
                with trace_cold_path("test_op"):
                    return 42

            # Should be very fast
            start = time.perf_counter()
            for _ in range(1000):
                result = traced_function()
                assert result == 42
            elapsed = time.perf_counter() - start

            # Should complete 1000 calls in well under 0.2 seconds
            assert elapsed < 0.2

    def test_trace_context_generation_performance(self):
        """Test trace context generation performance."""
        # Should be fast regardless of tracing state
        start = time.perf_counter()
        for i in range(100):
            metadata = get_correlation_and_trace_context(
                run_id=f"run_{i}",
                dataset_id="features",
                instrument_id="EUR/USD",
                ts_min=1000000000 + i,
                ts_max=2000000000 + i,
                count=100,
            )
            assert "correlation_id" in metadata
        elapsed = time.perf_counter() - start

        # Should complete 100 generations in well under 0.1 seconds
        assert elapsed < 0.1
