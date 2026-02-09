from __future__ import annotations

import pytest

from ml.common import metrics as metrics_catalog
from ml.config import names as metric_names


pytestmark = pytest.mark.unit


def test_remediation_metric_names_are_unique_and_prefixed() -> None:
    metric_names_under_test = [
        metric_names.METRIC_CAUSALITY_MONOTONIC_VIOLATIONS_TOTAL,
        metric_names.METRIC_INFERENCE_DEADLINE_TIMEOUTS_TOTAL,
        metric_names.METRIC_DRIFT_POLICY_ACTIONS_TOTAL,
        metric_names.METRIC_ML_FAILURE_ACTIONS_TOTAL,
        metric_names.METRIC_REGISTRY_COMPATIBILITY_MIGRATION_BYPASS_TOTAL,
        metric_names.METRIC_REGISTRY_UNSIGNED_ARTIFACT_OVERRIDE_TOTAL,
    ]
    assert len(metric_names_under_test) == len(set(metric_names_under_test))
    assert all(name.startswith("nautilus_ml_") for name in metric_names_under_test)
    assert all(name.endswith("_total") for name in metric_names_under_test)


def test_remediation_metric_collectors_are_registered_in_catalog() -> None:
    assert hasattr(metrics_catalog, "causality_monotonic_violations_total")
    assert hasattr(metrics_catalog, "inference_deadline_timeouts_total")
    assert hasattr(metrics_catalog, "drift_policy_actions_total")
    assert hasattr(metrics_catalog, "ml_failure_actions_total")
    assert hasattr(metrics_catalog, "registry_compatibility_migration_bypass_total")
    assert hasattr(metrics_catalog, "registry_unsigned_artifact_override_total")
