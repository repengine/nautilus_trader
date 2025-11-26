"""
Circuit breaker pattern for resilience.

This module implements the circuit breaker pattern to prevent cascading failures
in distributed systems. The circuit breaker monitors operation failures and
temporarily blocks requests to failing services, allowing time for recovery.

Circuit Breaker States:
- CLOSED: Normal operation, requests pass through
- OPEN: Failure threshold exceeded, requests rejected immediately
- HALF_OPEN: Testing recovery, limited requests allowed

"""

from __future__ import annotations

import time
from dataclasses import dataclass
from dataclasses import field
from enum import Enum


class CircuitBreakerState(Enum):
    """
    Circuit breaker states.

    Attributes:
        CLOSED: Normal operation, all requests pass through
        OPEN: Failure threshold exceeded, requests rejected immediately
        HALF_OPEN: Testing recovery, limited requests allowed
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for preventing cascading failures.

    Implements the circuit breaker pattern with three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failure threshold exceeded, requests rejected immediately
    - HALF_OPEN: Testing recovery, limited requests allowed

    The circuit breaker tracks consecutive failures and opens the circuit when
    the failure threshold is reached. After a timeout period, it transitions to
    HALF_OPEN to test if the service has recovered. If test requests succeed,
    it closes the circuit. If they fail, it reopens.

    Attributes:
        failure_threshold: Number of consecutive failures before opening circuit
        timeout_seconds: Time to wait before attempting recovery (HALF_OPEN)
        half_open_max_requests: Max requests to allow in HALF_OPEN state

    Example:
        >>> breaker = CircuitBreaker(failure_threshold=5, timeout_seconds=60)
        >>>
        >>> # Normal operation (CLOSED)
        >>> if breaker.can_attempt():
        ...     try:
        ...         result = risky_operation()
        ...         breaker.record_success()
        ...     except Exception as exc:
        ...         logger.debug("risky_operation failed", exc_info=exc)
        ...         breaker.record_failure()
        ...
        >>> # After 5 failures, circuit opens
        >>> assert breaker.state == CircuitBreakerState.OPEN
        >>>
        >>> # After timeout, circuit transitions to HALF_OPEN
        >>> time.sleep(60)
        >>> assert breaker.state == CircuitBreakerState.HALF_OPEN
        >>>
        >>> # Successful test closes circuit
        >>> if breaker.can_attempt():
        ...     result = risky_operation()  # Success
        ...     breaker.record_success()
        >>> assert breaker.state == CircuitBreakerState.CLOSED
    """

    failure_threshold: int = 5
    timeout_seconds: float = 60.0
    half_open_max_requests: int = 1

    # Internal state (not part of public API)
    _state: CircuitBreakerState = field(default=CircuitBreakerState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float | None = field(default=None, init=False)
    _half_open_attempts: int = field(default=0, init=False)

    @property
    def state(self) -> CircuitBreakerState:
        """
        Get current circuit breaker state.

        Updates state based on timeout and current conditions before returning.

        Returns:
            Current circuit breaker state
        """
        self._update_state()
        return self._state

    def _update_state(self) -> None:
        """
        Update state based on timeout and current conditions.

        Handles OPEN → HALF_OPEN transition after timeout expires.
        """
        if self._state == CircuitBreakerState.OPEN:
            if self._last_failure_time is not None:
                time_since_failure = time.time() - self._last_failure_time
                if time_since_failure >= self.timeout_seconds:
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._half_open_attempts = 0

    def record_success(self) -> None:
        """
        Record a successful operation.

        Resets failure count and closes circuit if in HALF_OPEN state.
        """
        self._failure_count = 0
        if self._state == CircuitBreakerState.HALF_OPEN:
            self._state = CircuitBreakerState.CLOSED
            self._half_open_attempts = 0

    def record_failure(self) -> None:
        """
        Record a failed operation.

        Increments failure count and opens circuit if threshold exceeded.
        If in HALF_OPEN state, immediately reopens circuit.
        """
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitBreakerState.HALF_OPEN:
            # Failed during recovery attempt - reopen circuit
            self._state = CircuitBreakerState.OPEN
            self._half_open_attempts = 0
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitBreakerState.OPEN

    def can_attempt(self) -> bool:
        """
        Check if an operation attempt is allowed.

        Returns:
            True if attempt allowed, False if circuit is open
        """
        self._update_state()

        if self._state == CircuitBreakerState.CLOSED:
            return True
        elif self._state == CircuitBreakerState.HALF_OPEN:
            if self._half_open_attempts < self.half_open_max_requests:
                self._half_open_attempts += 1
                return True
            return False
        else:  # OPEN
            return False

    def reset(self) -> None:
        """
        Reset circuit breaker to initial state.

        Clears all internal state and returns to CLOSED.
        Useful for testing or manual recovery.
        """
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._half_open_attempts = 0


__all__ = ["CircuitBreaker", "CircuitBreakerState"]
