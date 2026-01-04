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

    stub_time = _StubTime(start=0.0)
    # Patch the globals used by CircuitBreaker methods directly so the test
    # remains stable even if other tests reload/import-scrub ml.actors.base.
    monkeypatch.setitem(CircuitBreaker.can_execute.__globals__, "time", stub_time)

    cb = CircuitBreaker(cfg, component_id="test_component")

    # Initially closed
    initial_state = cb.state
    assert initial_state.value == CircuitBreakerState.CLOSED.value
    assert cb.can_execute() is True

    # Trigger failures up to threshold → OPEN
    cb.record_failure()
    cb.record_failure()
    state_after_two_failures = cb.state
    assert (
        state_after_two_failures.value == CircuitBreakerState.CLOSED.value
    )  # not yet at threshold
    cb.record_failure()
    state_after_third_failure = cb.state
    assert state_after_third_failure.value == CircuitBreakerState.OPEN.value
    assert cb.can_execute() is False  # within recovery window

    # Advance time to allow HALF_OPEN attempt
    stub_time.advance(cfg.recovery_timeout)
    assert cb.can_execute() is True
    half_open_state = cb.state
    assert half_open_state.value == CircuitBreakerState.HALF_OPEN.value

    # One success not enough to close
    cb.record_success()
    after_first_success = cb.state
    assert after_first_success.value == CircuitBreakerState.HALF_OPEN.value
    # Second success closes
    cb.record_success()
    final_state = cb.state
    assert final_state.value == CircuitBreakerState.CLOSED.value


def test_circuit_breaker_half_open_failure_reopens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # noqa: ANN001
    cfg = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=5, success_threshold=2)
    stub_time = _StubTime(start=100.0)
    monkeypatch.setitem(CircuitBreaker.can_execute.__globals__, "time", stub_time)

    cb = CircuitBreaker(cfg, component_id="test_component")

    # Open the circuit
    cb.record_failure()
    cb.record_failure()
    state_after_open = cb.state
    assert state_after_open.value == CircuitBreakerState.OPEN.value

    # Move to half-open
    stub_time.advance(cfg.recovery_timeout)
    assert cb.can_execute() is True
    reopened_state = cb.state
    assert reopened_state.value == CircuitBreakerState.HALF_OPEN.value

    # A failure in half-open reopens immediately
    cb.record_failure()
    reopened_after_failure = cb.state
    assert reopened_after_failure.value == CircuitBreakerState.OPEN.value
