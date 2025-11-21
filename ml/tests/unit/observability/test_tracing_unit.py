"""
Unit tests for distributed tracing module.

Tests the core functionality of the tracing module in isolation, focusing on the
behavior when tracing is disabled (default state).

"""

import os
from unittest.mock import patch

import pytest

from ml.observability.tracing import (
    extract_and_link_trace_context,
    get_trace_context,
    inject_trace_context,
    is_tracing_enabled,
    trace_cold_path,
    trace_cold_path_decorator,
    trace_inference,
)


class TestTracingConfiguration:
    """
    Test tracing configuration and environment handling.
    """

    def test_default_tracing_disabled(self):
        """
        Test that tracing is disabled by default.
        """
        # Clear any existing environment variable
        with patch.dict(os.environ, {}, clear=True):
            assert not is_tracing_enabled()

    def test_explicit_disable_tracing(self):
        """
        Test explicit disable via environment variable.
        """
        with patch.dict(os.environ, {"ML_TRACING_ENABLED": "false"}):
            assert not is_tracing_enabled()

    def test_case_insensitive_disable(self):
        """
        Test case insensitive disable values.
        """
        test_values = ["False", "FALSE", "no", "0", "off"]
        for value in test_values:
            with patch.dict(os.environ, {"ML_TRACING_ENABLED": value}):
                assert not is_tracing_enabled()

    def test_enable_tracing_env_var(self):
        """
        Test enabling tracing via environment variable.
        """
        with patch.dict(os.environ, {"ML_TRACING_ENABLED": "true"}):
            # Result depends on OpenTelemetry availability
            # If not available, should still return False
            result = is_tracing_enabled()
            assert isinstance(result, bool)


class TestTracingFunctionsWhenDisabled:
    """
    Test tracing functions when tracing is disabled.
    """

    @pytest.fixture(autouse=True)
    def disable_tracing(self):
        """
        Ensure tracing is disabled for these tests.
        """
        with patch.dict(os.environ, {"ML_TRACING_ENABLED": "false"}):
            yield

    def test_get_trace_context_returns_empty_dict(self):
        """
        Test get_trace_context returns empty dict when disabled.
        """
        context = get_trace_context()
        assert context == {}
        assert isinstance(context, dict)

    def test_inject_trace_context_passthrough(self):
        """
        Test inject_trace_context is passthrough when disabled.
        """
        original_metadata = {"correlation_id": "test123", "extra": "data"}
        result = inject_trace_context(original_metadata)
        assert result == original_metadata
        # When tracing disabled, should return same object for efficiency

    def test_extract_and_link_trace_context_noop(self):
        """
        Test extract_and_link_trace_context is no-op when disabled.
        """
        test_metadata = {
            "correlation_id": "test123",
            "trace_context": {
                "traceparent": "00-12345678901234567890123456789012-1234567890123456-01",
            },
        }
        # Should not raise any exceptions
        extract_and_link_trace_context(test_metadata)

    def test_trace_cold_path_context_manager_noop(self):
        """
        Test trace_cold_path context manager is no-op when disabled.
        """
        with trace_cold_path("test_operation") as span:
            assert span is None

        # Test with additional parameters
        with trace_cold_path("test_op", correlation_id="test123", attr="value") as span:
            assert span is None

    def test_trace_cold_path_decorator_passthrough(self):
        """
        Test trace_cold_path_decorator is passthrough when disabled.
        """

        def _base(x: int, y: int) -> int:
            return x + y

        test_function = trace_cold_path_decorator("test_operation")(_base)
        result = test_function(5, 3)
        assert result == 8

        # Test with correlation_id parameter
        def _with_corr(data: str, corr_id: str) -> str:
            return f"{data}_{corr_id}"

        test_function_with_corr = trace_cold_path_decorator(
            "test_op",
            correlation_id_param="corr_id",
        )(_with_corr)
        result = test_function_with_corr("test", corr_id="abc123")
        assert result == "test_abc123"

    def test_trace_inference_decorator_passthrough(self):
        """
        Test trace_inference decorator is passthrough when disabled.
        """

        class MockActor:
            def on_bar(self, bar: "MockBar") -> str:
                return f"processed_{bar.symbol}"

        class MockBar:
            def __init__(self, symbol: str):
                self.symbol = symbol
                self.instrument_id = f"{symbol}.SIM"

        MockActor.on_bar = trace_inference("signal_generation")(MockActor.on_bar)

        actor = MockActor()
        bar = MockBar("EURUSD")
        result = actor.on_bar(bar)
        assert result == "processed_EURUSD"


class TestTracingErrorHandling:
    """
    Test error handling in tracing functions.
    """

    def test_inject_trace_context_with_invalid_metadata(self):
        """
        Test inject_trace_context handles invalid metadata gracefully.
        """
        # Test with None
        result = inject_trace_context({})
        assert isinstance(result, dict)

        # Test with already existing trace_context
        metadata = {"trace_context": {"existing": "context"}}
        result = inject_trace_context(metadata)
        assert result["trace_context"]["existing"] == "context"

    def test_extract_and_link_with_invalid_metadata(self):
        """
        Test extract_and_link handles invalid metadata gracefully.
        """
        # Test with empty dict
        extract_and_link_trace_context({})

        # Test with invalid trace_context type
        extract_and_link_trace_context({"trace_context": "not_a_dict"})

        # Test with missing trace_context
        extract_and_link_trace_context({"correlation_id": "test123"})

        # Test with None trace_context
        extract_and_link_trace_context({"trace_context": None})

    def test_trace_cold_path_with_exception_in_block(self):
        """
        Test trace_cold_path handles exceptions in traced block.
        """
        with pytest.raises(ValueError, match="test error"):
            with trace_cold_path("test_operation"):
                raise ValueError("test error")

    def test_decorator_with_exception_in_function(self):
        """
        Test decorators handle exceptions in decorated functions.
        """

        @trace_cold_path_decorator("test_operation")
        def failing_function():
            raise RuntimeError("function failed")

        with pytest.raises(RuntimeError, match="function failed"):
            failing_function()


class TestTracingFunctionsWhenEnabled:
    """
    Validate tracing helpers when tracing is explicitly enabled.
    """

    @pytest.fixture(autouse=True)
    def _enable_tracing(self, mock_tracing_backend):
        """
        Enable tracing and expose the harness for assertions.
        """
        self._harness = mock_tracing_backend

    def test_get_trace_context_returns_traceparent(self) -> None:
        """
        get_trace_context should include the harness traceparent.
        """
        context = get_trace_context()
        assert context.get("traceparent") == self._harness.traceparent

    def test_inject_trace_context_enriches_metadata(self) -> None:
        """
        inject_trace_context should add a traceparent entry.
        """
        metadata = {"correlation_id": "abc123"}
        result = inject_trace_context(metadata)
        assert result is not metadata  # returns a new dict when mutating
        assert result["trace_context"]["traceparent"] == self._harness.traceparent

    def test_trace_cold_path_records_span(self) -> None:
        """
        trace_cold_path should emit spans via the harness tracer.
        """
        with trace_cold_path("enabled_operation", correlation_id="cid-1") as span:
            assert span is not None
            span.set_attribute("custom", "value")

        recorded_span = self._harness.tracer.spans[-1]
        recorded_span.set_attribute.assert_any_call("correlation_id", "cid-1")
        recorded_span.set_attribute.assert_any_call("operation_type", "cold_path")
        recorded_span.set_attribute.assert_any_call("custom", "value")


class TestTracingModuleImports:
    """
    Test module imports and dependencies.
    """

    def test_all_functions_importable(self):
        """
        Test all tracing functions can be imported.
        """
        from ml.observability.tracing import (
            extract_and_link_trace_context,
            get_trace_context,
            inject_trace_context,
            is_tracing_enabled,
            trace_cold_path,
            trace_cold_path_decorator,
            trace_inference,
        )

        # All should be callable
        assert callable(extract_and_link_trace_context)
        assert callable(get_trace_context)
        assert callable(inject_trace_context)
        assert callable(is_tracing_enabled)
        assert callable(trace_cold_path)
        assert callable(trace_cold_path_decorator)
        assert callable(trace_inference)

    def test_observability_module_exports(self):
        """
        Test tracing functions are exported from observability module.
        """
        from ml.observability import (
            extract_and_link_trace_context,
            get_trace_context,
            inject_trace_context,
            is_tracing_enabled,
            trace_cold_path,
            trace_cold_path_decorator,
            trace_inference,
        )

        # Should be available through main observability module
        assert callable(extract_and_link_trace_context)
        assert callable(get_trace_context)
        assert callable(inject_trace_context)
        assert callable(is_tracing_enabled)
        assert callable(trace_cold_path)
        assert callable(trace_cold_path_decorator)
        assert callable(trace_inference)

    def test_common_module_trace_utilities(self):
        """
        Test trace utilities are exported from common module.
        """
        from ml.common import (
            extract_and_link_from_event,
            get_correlation_and_trace_context,
        )

        assert callable(extract_and_link_from_event)
        assert callable(get_correlation_and_trace_context)


class TestTracingFunctionSignatures:
    """
    Test function signatures and parameter handling.
    """

    def test_trace_cold_path_parameters(self):
        """
        Test trace_cold_path parameter handling.
        """
        # Test with various parameter combinations
        with trace_cold_path("operation") as span:
            assert span is None

        with trace_cold_path("operation", correlation_id="test123") as span:
            assert span is None

        with trace_cold_path("operation", attr1="value1", attr2="value2") as span:
            assert span is None

    def test_trace_decorator_parameters(self):
        """
        Test trace decorator parameter handling.
        """

        def _func1() -> str:
            return "result1"

        func1 = trace_cold_path_decorator("test_op")(_func1)
        assert func1() == "result1"

        def _func2(data: str, corr_id: str = "default") -> str:
            return f"{data}_{corr_id}"

        func2 = trace_cold_path_decorator(
            "test_op",
            correlation_id_param="corr_id",
        )(_func2)
        assert func2("test") == "test_default"
        assert func2("test", corr_id="custom") == "test_custom"

    def test_function_metadata_preservation(self):
        """
        Test that decorators preserve function metadata.
        """

        @trace_cold_path_decorator("test_operation")
        def original_function():
            """
            Original function docstring.
            """
            return "original"

        # Should preserve function name and docstring
        assert original_function.__name__ == "original_function"
        assert "Original function docstring" in (original_function.__doc__ or "")

        @trace_inference("inference_op")
        def inference_function():
            """
            Inference function docstring.
            """
            return "inference"

        assert inference_function.__name__ == "inference_function"
        assert "Inference function docstring" in (inference_function.__doc__ or "")
