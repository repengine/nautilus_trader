"""
Policy decision helpers for actor-side remediation behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

from ml.config.policy import InferenceTimeoutAction
from ml.config.policy import MLFailureAction


@dataclass(frozen=True, slots=True)
class InferenceDeadlineDecision:
    """
    Decision result for inference deadline guard evaluation.

    Parameters
    ----------
    exceeded : bool
        Whether the measured inference latency exceeded the deadline.
    action : InferenceTimeoutAction
        Policy action configured for deadline breaches.
    drop_prediction : bool
        Whether the current prediction should be dropped.
    halt_inference : bool
        Whether inference should transition into a halted state.

    """

    exceeded: bool
    action: InferenceTimeoutAction
    drop_prediction: bool
    halt_inference: bool


@dataclass(frozen=True, slots=True)
class MLFailureDecision:
    """
    Decision result for ML failure policy evaluation.

    Parameters
    ----------
    action : MLFailureAction
        Policy action to apply for the failure.
    transition_degraded : bool
        Whether the actor should transition to degraded state.
    halt_inference : bool
        Whether the actor should halt inference.

    """

    action: MLFailureAction
    transition_degraded: bool
    halt_inference: bool


def evaluate_inference_deadline_guard(
    *,
    elapsed_ms: float,
    deadline_ms: float,
    enabled: bool,
    timeout_action: InferenceTimeoutAction,
) -> InferenceDeadlineDecision:
    """
    Evaluate deadline guard policy for one inference execution.

    Parameters
    ----------
    elapsed_ms : float
        Measured inference latency in milliseconds.
    deadline_ms : float
        Configured maximum inference latency in milliseconds.
    enabled : bool
        Whether deadline guard enforcement is enabled.
    timeout_action : InferenceTimeoutAction
        Action policy for deadline breaches.

    Returns
    -------
    InferenceDeadlineDecision
        Decision describing whether to drop or halt.

    """
    exceeded = float(elapsed_ms) > float(deadline_ms)
    if not exceeded or not enabled:
        return InferenceDeadlineDecision(
            exceeded=exceeded,
            action=timeout_action,
            drop_prediction=False,
            halt_inference=False,
        )

    if timeout_action == InferenceTimeoutAction.HALT:
        return InferenceDeadlineDecision(
            exceeded=True,
            action=timeout_action,
            drop_prediction=True,
            halt_inference=True,
        )

    return InferenceDeadlineDecision(
        exceeded=True,
        action=timeout_action,
        drop_prediction=True,
        halt_inference=False,
    )


def evaluate_ml_failure_action(
    *,
    action: MLFailureAction,
) -> MLFailureDecision:
    """
    Evaluate ML failure action policy for a prediction failure event.

    Parameters
    ----------
    action : MLFailureAction
        Configured action policy.

    Returns
    -------
    MLFailureDecision
        Decision describing degraded/halt transitions.

    """
    if action == MLFailureAction.HALT:
        return MLFailureDecision(
            action=action,
            transition_degraded=False,
            halt_inference=True,
        )
    if action == MLFailureAction.DEGRADED:
        return MLFailureDecision(
            action=action,
            transition_degraded=True,
            halt_inference=False,
        )
    return MLFailureDecision(
        action=action,
        transition_degraded=False,
        halt_inference=False,
    )


__all__ = [
    "InferenceDeadlineDecision",
    "MLFailureDecision",
    "evaluate_inference_deadline_guard",
    "evaluate_ml_failure_action",
]
