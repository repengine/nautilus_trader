"""
Shared causality guard utilities for monotonic ingress validation.

This module provides a single implementation for timestamp causality checks that
can be reused by actors, stores, and feature pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ml.config.policy import CausalityMonotonicEnforcement


class CausalityViolation(str, Enum):
    """
    Enumerates causality violation categories.
    """

    NON_MONOTONIC = "non_monotonic"
    FUTURE_TIMESTAMP = "future_timestamp"


class CausalityAction(str, Enum):
    """
    Enumerates guard actions emitted by causality enforcement.
    """

    ACCEPT = "accept"
    DROP = "drop"
    RESET = "reset"


@dataclass(frozen=True, slots=True)
class CausalityGuardResult:
    """
    Result payload produced by :class:`CausalityGuard`.

    Parameters
    ----------
    accepted : bool
        Whether the caller should accept and process the event.
    action : CausalityAction
        Action mapped from the configured enforcement policy.
    violation : CausalityViolation | None
        Violation kind, if any.
    message : str | None
        Human-readable violation description for logs/telemetry.
    timestamp_ns : int
        Candidate event timestamp (nanoseconds).
    previous_timestamp_ns : int | None
        Last accepted timestamp used for monotonic checks.
    reference_timestamp_ns : int | None
        Reference clock timestamp for future-time checks.
    """

    accepted: bool
    action: CausalityAction
    violation: CausalityViolation | None
    message: str | None
    timestamp_ns: int
    previous_timestamp_ns: int | None
    reference_timestamp_ns: int | None

    @property
    def has_violation(self) -> bool:
        """
        Return whether the evaluated timestamp violated causality rules.

        Returns
        -------
        bool
            ``True`` when a causality violation was detected.
        """
        return self.violation is not None

    @property
    def requires_reset(self) -> bool:
        """
        Return whether caller state should be reset.

        Returns
        -------
        bool
            ``True`` when action is :attr:`CausalityAction.RESET`.
        """
        return self.action == CausalityAction.RESET


class CausalityGuard:
    """
    Validate event timestamps against monotonic and no-future constraints.

    Parameters
    ----------
    enforcement : CausalityMonotonicEnforcement, default WARN_ONLY
        Policy mode used to map violations to actions.
    max_future_drift_ns : int, default 0
        Maximum allowed lead over ``reference_timestamp_ns`` when checking for
        future timestamps.

    Examples
    --------
    >>> guard = CausalityGuard(CausalityMonotonicEnforcement.DROP)
    >>> result = guard.validate_event_timestamp(timestamp_ns=90, previous_timestamp_ns=100)
    >>> result.accepted
    False
    """

    def __init__(
        self,
        enforcement: CausalityMonotonicEnforcement = CausalityMonotonicEnforcement.WARN_ONLY,
        *,
        max_future_drift_ns: int = 0,
    ) -> None:
        if max_future_drift_ns < 0:
            raise ValueError("max_future_drift_ns must be >= 0")
        self._enforcement = enforcement
        self._max_future_drift_ns = int(max_future_drift_ns)

    @property
    def enforcement(self) -> CausalityMonotonicEnforcement:
        """
        Return guard enforcement mode.

        Returns
        -------
        CausalityMonotonicEnforcement
            Configured enforcement policy.
        """
        return self._enforcement

    @property
    def max_future_drift_ns(self) -> int:
        """
        Return allowed future drift window in nanoseconds.

        Returns
        -------
        int
            Maximum lead over the reference timestamp.
        """
        return self._max_future_drift_ns

    def validate_event_timestamp(
        self,
        timestamp_ns: int,
        *,
        previous_timestamp_ns: int | None = None,
        reference_timestamp_ns: int | None = None,
    ) -> CausalityGuardResult:
        """
        Validate a timestamp against causality constraints.

        Parameters
        ----------
        timestamp_ns : int
            Candidate event timestamp (nanoseconds).
        previous_timestamp_ns : int | None, optional
            Last accepted timestamp; when provided, monotonic non-decreasing
            ordering is enforced.
        reference_timestamp_ns : int | None, optional
            Reference clock timestamp; when provided, future-time checks are
            enforced with ``max_future_drift_ns`` tolerance.

        Returns
        -------
        CausalityGuardResult
            Typed validation result with action and violation metadata.
        """
        timestamp_value = int(timestamp_ns)
        previous_value = int(previous_timestamp_ns) if previous_timestamp_ns is not None else None
        reference_value = (
            int(reference_timestamp_ns) if reference_timestamp_ns is not None else None
        )

        violation = self._detect_violation(
            timestamp_ns=timestamp_value,
            previous_timestamp_ns=previous_value,
            reference_timestamp_ns=reference_value,
        )
        if violation is None:
            return CausalityGuardResult(
                accepted=True,
                action=CausalityAction.ACCEPT,
                violation=None,
                message=None,
                timestamp_ns=timestamp_value,
                previous_timestamp_ns=previous_value,
                reference_timestamp_ns=reference_value,
            )

        action = self._action_for_violation()
        return CausalityGuardResult(
            accepted=action == CausalityAction.ACCEPT,
            action=action,
            violation=violation,
            message=self._build_violation_message(
                violation=violation,
                timestamp_ns=timestamp_value,
                previous_timestamp_ns=previous_value,
                reference_timestamp_ns=reference_value,
            ),
            timestamp_ns=timestamp_value,
            previous_timestamp_ns=previous_value,
            reference_timestamp_ns=reference_value,
        )

    def _detect_violation(
        self,
        *,
        timestamp_ns: int,
        previous_timestamp_ns: int | None,
        reference_timestamp_ns: int | None,
    ) -> CausalityViolation | None:
        if previous_timestamp_ns is not None and timestamp_ns < previous_timestamp_ns:
            return CausalityViolation.NON_MONOTONIC
        if reference_timestamp_ns is None:
            return None
        if timestamp_ns > reference_timestamp_ns + self._max_future_drift_ns:
            return CausalityViolation.FUTURE_TIMESTAMP
        return None

    def _action_for_violation(self) -> CausalityAction:
        if self._enforcement == CausalityMonotonicEnforcement.DROP:
            return CausalityAction.DROP
        if self._enforcement == CausalityMonotonicEnforcement.RESET:
            return CausalityAction.RESET
        return CausalityAction.ACCEPT

    def _build_violation_message(
        self,
        *,
        violation: CausalityViolation,
        timestamp_ns: int,
        previous_timestamp_ns: int | None,
        reference_timestamp_ns: int | None,
    ) -> str:
        if violation == CausalityViolation.NON_MONOTONIC:
            return (
                f"timestamp_ns={timestamp_ns} is older than previous_timestamp_ns="
                f"{previous_timestamp_ns}"
            )
        return (
            f"timestamp_ns={timestamp_ns} exceeds reference_timestamp_ns={reference_timestamp_ns} "
            f"with max_future_drift_ns={self._max_future_drift_ns}"
        )


__all__ = [
    "CausalityAction",
    "CausalityGuard",
    "CausalityGuardResult",
    "CausalityViolation",
]
