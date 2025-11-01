from __future__ import annotations

from ml.actors.base import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState


def test_circuit_breaker_transitions() -> None:
    cfg = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0, success_threshold=2)
    cb = CircuitBreaker(cfg, component_id="ml_actor")

    # Initially closed
    initial_state = cb.state
    assert initial_state.value == CircuitBreakerState.CLOSED.value
    assert cb.can_execute() is True

    # First failure does not open
    cb.record_failure()
    post_first_failure = cb.state
    assert post_first_failure.value == CircuitBreakerState.CLOSED.value

    # Second failure opens
    cb.record_failure()
    state_after_second_failure = cb.state
    assert state_after_second_failure.value == CircuitBreakerState.OPEN.value
    assert cb.can_execute() is True  # recovery_timeout=0 → HALF_OPEN immediately
    state_after_probe = cb.state
    assert state_after_probe.value == CircuitBreakerState.HALF_OPEN.value

    # Need two consecutive successes to close
    cb.record_success()
    after_first_success = cb.state
    assert after_first_success.value == CircuitBreakerState.HALF_OPEN.value
    cb.record_success()
    final_state = cb.state
    assert final_state.value == CircuitBreakerState.CLOSED.value
