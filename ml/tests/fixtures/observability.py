#!/usr/bin/env python3
"""
Observability-focused fixtures for tracing and telemetry tests.

These helpers provide deterministic OpenTelemetry shims so the test-suite
can exercise tracing logic without requiring the optional OTEL dependency
or contacting a collector endpoint.
"""

from __future__ import annotations

import os
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest


@dataclass(frozen=True, slots=True)
class MockTracingComponents:
    """Container exposing patched tracing artefacts for assertions."""

    tracer: Any
    propagate: Any
    context: Any
    traceparent: str


class _MockTracer:
    """Lightweight tracer that yields MagicMock spans."""

    def __init__(self) -> None:
        self.spans: list[MagicMock] = []

    def start_as_current_span(self, operation_name: str):
        span = MagicMock(name=f"span[{operation_name}]")
        span.operation_name = operation_name
        self.spans.append(span)

        class _SpanContext:
            def __enter__(self_inner) -> MagicMock:
                return span

            def __exit__(self_inner, exc_type, exc, tb) -> bool:
                return False

        return _SpanContext()


class _MockPropagate:
    """Propagator that injects/extracts deterministic trace headers."""

    def __init__(self, traceparent: str) -> None:
        self._traceparent = traceparent
        self.inject_calls: list[dict[str, str]] = []
        self.extract_calls: list[dict[str, str]] = []

    def inject(self, carrier: dict[str, str]) -> None:
        carrier["traceparent"] = self._traceparent
        self.inject_calls.append(dict(carrier))

    def extract(self, carrier: dict[str, str]) -> dict[str, str]:
        self.extract_calls.append(dict(carrier))
        return {"traceparent": carrier.get("traceparent", self._traceparent)}


@contextmanager
def patch_tracing_backend(
    *,
    traceparent: str | None = None,
) -> Generator[MockTracingComponents, None, None]:
    """
    Patch the tracing backend with deterministic mocks.

    Parameters
    ----------
    traceparent : str | None
        Optional traceparent header to inject. When omitted a random but valid
        header is generated.
    """

    resolved_traceparent = traceparent or (
        f"00-{uuid.uuid4().hex:0>32}-{uuid.uuid4().hex[:16]:0>16}-01"
    )
    propagate = _MockPropagate(resolved_traceparent)
    tracer = _MockTracer()
    otel_context = MagicMock(name="otel_context")

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"ML_TRACING_ENABLED": "true"}, clear=False))
        stack.enter_context(patch("ml.observability.tracing.HAS_OPENTELEMETRY", True))
        stack.enter_context(patch("ml.observability.tracing._propagate", propagate))
        stack.enter_context(patch("ml.observability.tracing._context", otel_context))
        stack.enter_context(patch("ml.observability.tracing._tracer", tracer))
        stack.enter_context(patch("ml.observability.tracing._ensure_tracing_backend", lambda: True))
        yield MockTracingComponents(
            tracer=tracer,
            propagate=propagate,
            context=otel_context,
            traceparent=resolved_traceparent,
        )


@pytest.fixture
def mock_tracing_backend() -> Generator[MockTracingComponents, None, None]:
    """
    Provide deterministic OpenTelemetry shims for tracing tests.

    Returns
    -------
    MockTracingComponents
        Access to tracer/propagator/context mocks for assertions.
    """

    with patch_tracing_backend() as components:
        yield components


__all__ = [
    "MockTracingComponents",
    "mock_tracing_backend",
    "patch_tracing_backend",
]
