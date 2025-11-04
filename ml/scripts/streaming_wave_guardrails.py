"""Compatibility shims for streaming wave guardrail evaluation."""

from ml.training.event_driven.guardrails.evaluation import VALIDATION_FAILURE_COUNTER_NAME
from ml.training.event_driven.guardrails.evaluation import VALIDATION_FAILURE_DETAIL_GAUGE_NAME
from ml.training.event_driven.guardrails.evaluation import VALIDATION_FAILURE_LATEST_GAUGE_NAME
from ml.training.event_driven.guardrails.evaluation import GuardrailEvaluation
from ml.training.event_driven.guardrails.evaluation import ManifestGuardrailReport
from ml.training.event_driven.guardrails.evaluation import evaluate_manifests


__all__ = [
    "VALIDATION_FAILURE_COUNTER_NAME",
    "VALIDATION_FAILURE_DETAIL_GAUGE_NAME",
    "VALIDATION_FAILURE_LATEST_GAUGE_NAME",
    "GuardrailEvaluation",
    "ManifestGuardrailReport",
    "evaluate_manifests",
]
