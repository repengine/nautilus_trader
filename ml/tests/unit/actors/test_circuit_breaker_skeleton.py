from __future__ import annotations

from ml.actors.base import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState


def test_circuit_breaker_transitions() -> None:
    cfg = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0, success_threshold=2)
    cb = CircuitBreaker(cfg, component_id="ml_actor")

    # Initially closed
    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.can_execute() is True

    # First failure does not open
    cb.record_failure()
    assert cb.state == CircuitBreakerState.CLOSED

    # Second failure opens
    cb.record_failure()
    assert cb.state == CircuitBreakerState.OPEN
    assert cb.can_execute() is True  # recovery_timeout=0 → HALF_OPEN immediately
    assert cb.state == CircuitBreakerState.HALF_OPEN

    # Need two consecutive successes to close
    cb.record_success()
    assert cb.state == CircuitBreakerState.HALF_OPEN
    cb.record_success()
    assert cb.state == CircuitBreakerState.CLOSED
