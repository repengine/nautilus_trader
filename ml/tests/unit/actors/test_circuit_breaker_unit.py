from __future__ import annotations

import types
from typing import Any

import pytest

from ml.actors.base import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState


class _StubTime:
    def __init__(self, start: float) -> None:
        self.now = float(start)

    def time(self) -> float:  # matches time.time()
        return float(self.now)

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


def test_circuit_breaker_transitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # noqa: ANN001 - pytest fixture
    # Configure small thresholds to keep test fast
    cfg = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=10, success_threshold=2)

    # Monkeypatch time in module to a controllable stub
    import ml.actors.base as base_mod

    stub_time = _StubTime(start=0.0)
    monkeypatch.setattr(base_mod, "time", stub_time)  # module attribute used in CircuitBreaker

    cb = CircuitBreaker(cfg, component_id="test_component")

    # Initially closed
    assert cb.state is CircuitBreakerState.CLOSED
    assert cb.can_execute() is True

    # Trigger failures up to threshold → OPEN
    cb.record_failure()
    cb.record_failure()
    assert cb.state is CircuitBreakerState.CLOSED  # not yet at threshold
    cb.record_failure()
    assert cb.state is CircuitBreakerState.OPEN
    assert cb.can_execute() is False  # within recovery window

    # Advance time to allow HALF_OPEN attempt
    stub_time.advance(cfg.recovery_timeout)
    assert cb.can_execute() is True
    assert cb.state is CircuitBreakerState.HALF_OPEN

    # One success not enough to close
    cb.record_success()
    assert cb.state is CircuitBreakerState.HALF_OPEN
    # Second success closes
    cb.record_success()
    assert cb.state is CircuitBreakerState.CLOSED


def test_circuit_breaker_half_open_failure_reopens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # noqa: ANN001
    cfg = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=5, success_threshold=2)
    import ml.actors.base as base_mod

    stub_time = _StubTime(start=100.0)
    monkeypatch.setattr(base_mod, "time", stub_time)

    cb = CircuitBreaker(cfg, component_id="test_component")

    # Open the circuit
    cb.record_failure()
    cb.record_failure()
    assert cb.state is CircuitBreakerState.OPEN

    # Move to half-open
    stub_time.advance(cfg.recovery_timeout)
    assert cb.can_execute() is True
    assert cb.state is CircuitBreakerState.HALF_OPEN

    # A failure in half-open reopens immediately
    cb.record_failure()
    assert cb.state is CircuitBreakerState.OPEN
