from __future__ import annotations

from ml.actors.common.remediation import evaluate_inference_deadline_guard
from ml.actors.common.remediation import evaluate_ml_failure_action
from ml.config.policy import InferenceTimeoutAction
from ml.config.policy import MLFailureAction


def test_evaluate_inference_deadline_guard_no_enforcement_when_disabled() -> None:
    decision = evaluate_inference_deadline_guard(
        elapsed_ms=9.0,
        deadline_ms=5.0,
        enabled=False,
        timeout_action=InferenceTimeoutAction.HALT,
    )

    assert decision.exceeded is True
    assert decision.drop_prediction is False
    assert decision.halt_inference is False


def test_evaluate_inference_deadline_guard_drop_action() -> None:
    decision = evaluate_inference_deadline_guard(
        elapsed_ms=9.0,
        deadline_ms=5.0,
        enabled=True,
        timeout_action=InferenceTimeoutAction.DROP,
    )

    assert decision.exceeded is True
    assert decision.drop_prediction is True
    assert decision.halt_inference is False


def test_evaluate_inference_deadline_guard_halt_action() -> None:
    decision = evaluate_inference_deadline_guard(
        elapsed_ms=9.0,
        deadline_ms=5.0,
        enabled=True,
        timeout_action=InferenceTimeoutAction.HALT,
    )

    assert decision.exceeded is True
    assert decision.drop_prediction is True
    assert decision.halt_inference is True


def test_evaluate_ml_failure_action_log_only() -> None:
    decision = evaluate_ml_failure_action(action=MLFailureAction.LOG_ONLY)

    assert decision.transition_degraded is False
    assert decision.halt_inference is False


def test_evaluate_ml_failure_action_degraded() -> None:
    decision = evaluate_ml_failure_action(action=MLFailureAction.DEGRADED)

    assert decision.transition_degraded is True
    assert decision.halt_inference is False


def test_evaluate_ml_failure_action_halt() -> None:
    decision = evaluate_ml_failure_action(action=MLFailureAction.HALT)

    assert decision.transition_degraded is False
    assert decision.halt_inference is True
