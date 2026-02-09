from __future__ import annotations

import pytest

from ml.common.causality_guard import CausalityAction
from ml.common.causality_guard import CausalityGuard
from ml.common.causality_guard import CausalityViolation
from ml.config.policy import CausalityMonotonicEnforcement


pytestmark = pytest.mark.unit


def test_causality_guard_when_timestamps_are_monotonic_accepts() -> None:
    guard = CausalityGuard()

    result = guard.validate_event_timestamp(
        timestamp_ns=200,
        previous_timestamp_ns=199,
        reference_timestamp_ns=200,
    )

    assert result.accepted is True
    assert result.has_violation is False
    assert result.action == CausalityAction.ACCEPT
    assert result.violation is None


def test_causality_guard_exposes_configuration_properties() -> None:
    guard = CausalityGuard(
        CausalityMonotonicEnforcement.DROP,
        max_future_drift_ns=9,
    )

    assert guard.enforcement == CausalityMonotonicEnforcement.DROP
    assert guard.max_future_drift_ns == 9


def test_causality_guard_when_warn_only_and_non_monotonic_keeps_accepting() -> None:
    guard = CausalityGuard(CausalityMonotonicEnforcement.WARN_ONLY)

    result = guard.validate_event_timestamp(timestamp_ns=99, previous_timestamp_ns=100)

    assert result.accepted is True
    assert result.has_violation is True
    assert result.action == CausalityAction.ACCEPT
    assert result.violation == CausalityViolation.NON_MONOTONIC


def test_causality_guard_when_no_reference_timestamp_only_monotonic_rule_is_applied() -> None:
    guard = CausalityGuard()

    result = guard.validate_event_timestamp(timestamp_ns=100, previous_timestamp_ns=None)

    assert result.accepted is True
    assert result.has_violation is False
    assert result.message is None


def test_causality_guard_when_drop_mode_and_non_monotonic_drops_event() -> None:
    guard = CausalityGuard(CausalityMonotonicEnforcement.DROP)

    result = guard.validate_event_timestamp(timestamp_ns=99, previous_timestamp_ns=100)

    assert result.accepted is False
    assert result.action == CausalityAction.DROP
    assert result.violation == CausalityViolation.NON_MONOTONIC
    assert result.requires_reset is False
    assert "previous_timestamp_ns=100" in (result.message or "")


def test_causality_guard_when_reset_mode_and_future_timestamp_requires_reset() -> None:
    guard = CausalityGuard(
        CausalityMonotonicEnforcement.RESET,
        max_future_drift_ns=5,
    )

    result = guard.validate_event_timestamp(timestamp_ns=111, reference_timestamp_ns=100)

    assert result.accepted is False
    assert result.action == CausalityAction.RESET
    assert result.violation == CausalityViolation.FUTURE_TIMESTAMP
    assert result.requires_reset is True
    assert "max_future_drift_ns=5" in (result.message or "")


def test_causality_guard_when_future_within_drift_is_accepted() -> None:
    guard = CausalityGuard(
        CausalityMonotonicEnforcement.DROP,
        max_future_drift_ns=5,
    )

    result = guard.validate_event_timestamp(timestamp_ns=105, reference_timestamp_ns=100)

    assert result.accepted is True
    assert result.action == CausalityAction.ACCEPT
    assert result.violation is None


def test_causality_guard_when_drift_negative_raises_value_error() -> None:
    with pytest.raises(ValueError, match="max_future_drift_ns"):
        CausalityGuard(max_future_drift_ns=-1)
