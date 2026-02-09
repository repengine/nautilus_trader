from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ml.actors.common.drift_monitoring import DriftMonitoringComponent
from ml.actors.common.drift_monitoring import resolve_drift_policy_config
from ml.actors.common.drift_monitoring import resolve_replay_safe_drift_action
from ml.actors.common.drift_monitoring import threshold_for_policy_action
from ml.config.policy import DriftActionPolicy


pytestmark = [
    pytest.mark.unit,
    pytest.mark.usefixtures("isolated_prometheus_registry"),
]


def _config(
    *,
    action: DriftActionPolicy,
    log_only_threshold: float = 0.35,
    degraded_threshold: float = 0.75,
    fail_closed_threshold: float = 1.25,
    min_baseline_samples: int = 3,
    min_sample_window: int = 4,
) -> SimpleNamespace:
    return SimpleNamespace(
        remediation_policy=SimpleNamespace(drift_action_policy=action),
        drift_log_only_threshold=log_only_threshold,
        drift_degraded_threshold=degraded_threshold,
        drift_fail_closed_threshold=fail_closed_threshold,
        drift_min_baseline_samples=min_baseline_samples,
        drift_min_sample_window=min_sample_window,
    )


def test_resolve_drift_policy_config_reads_overrides() -> None:
    policy = resolve_drift_policy_config(
        _config(action=DriftActionPolicy.DEGRADED),
        env={
            "ML_DRIFT_LOG_ONLY_THRESHOLD": "0.2",
            "ML_DRIFT_DEGRADED_THRESHOLD": "0.6",
            "ML_DRIFT_FAIL_CLOSED_THRESHOLD": "1.4",
            "ML_DRIFT_MIN_BASELINE_SAMPLES": "7",
            "ML_DRIFT_MIN_SAMPLE_WINDOW": "9",
        },
    )

    assert policy.action_policy == DriftActionPolicy.DEGRADED
    assert policy.thresholds.log_only == pytest.approx(0.2)
    assert policy.thresholds.degraded == pytest.approx(0.6)
    assert policy.thresholds.fail_closed == pytest.approx(1.4)
    assert policy.min_baseline_samples == 7
    assert policy.min_observed_samples == 9


def test_threshold_for_policy_action_maps_expected_threshold() -> None:
    policy = resolve_drift_policy_config(
        _config(
            action=DriftActionPolicy.LOG_ONLY,
            log_only_threshold=0.1,
            degraded_threshold=0.5,
            fail_closed_threshold=0.9,
        ),
        env={},
    )

    assert threshold_for_policy_action(
        thresholds=policy.thresholds,
        action=DriftActionPolicy.LOG_ONLY,
    ) == pytest.approx(0.1)
    assert threshold_for_policy_action(
        thresholds=policy.thresholds,
        action=DriftActionPolicy.DEGRADED,
    ) == pytest.approx(0.5)
    assert threshold_for_policy_action(
        thresholds=policy.thresholds,
        action=DriftActionPolicy.FAIL_CLOSED,
    ) == pytest.approx(0.9)


def test_drift_monitoring_gates_until_baseline_and_sample_windows_ready() -> None:
    policy = resolve_drift_policy_config(
        _config(
            action=DriftActionPolicy.LOG_ONLY,
            log_only_threshold=0.1,
            min_baseline_samples=3,
            min_sample_window=4,
        ),
        env={},
    )
    component = DriftMonitoringComponent(
        n_features=2,
        policy_config=policy,
        actor_id="actor-1",
        feature_set_id="fs-1",
        log=None,
    )

    obs1 = component.record_inference(np.array([1.0, 1.0], dtype=np.float32))
    obs2 = component.record_inference(np.array([1.0, 1.0], dtype=np.float32))
    obs3 = component.record_inference(np.array([1.0, 1.0], dtype=np.float32))
    obs4 = component.record_inference(np.array([2.0, 2.0], dtype=np.float32))

    assert obs1.baseline_ready is False
    assert obs2.baseline_ready is False
    assert obs3.baseline_ready is True
    assert obs3.policy_ready is False
    assert obs4.policy_ready is True
    assert obs4.drift_score > 0.0

    assert component.evaluate_policy(observation=obs3, policy_ready=obs3.policy_ready) is None
    outcome = component.evaluate_policy(observation=obs4, policy_ready=obs4.policy_ready)
    assert outcome is not None
    assert outcome.action == DriftActionPolicy.LOG_ONLY


@pytest.mark.parametrize(
    ("action", "fail_closed_threshold", "expected"),
    [
        (DriftActionPolicy.DEGRADED, 1.2, DriftActionPolicy.DEGRADED),
        (DriftActionPolicy.FAIL_CLOSED, 0.8, DriftActionPolicy.FAIL_CLOSED),
    ],
)
def test_drift_monitoring_applies_configured_policy_action(
    action: DriftActionPolicy,
    fail_closed_threshold: float,
    expected: DriftActionPolicy,
) -> None:
    policy = resolve_drift_policy_config(
        _config(
            action=action,
            log_only_threshold=0.1,
            degraded_threshold=0.5,
            fail_closed_threshold=fail_closed_threshold,
            min_baseline_samples=2,
            min_sample_window=3,
        ),
        env={},
    )
    component = DriftMonitoringComponent(
        n_features=2,
        policy_config=policy,
        actor_id="actor-2",
        feature_set_id="fs-1",
        log=None,
    )

    component.record_inference(np.array([1.0, 1.0], dtype=np.float32))
    component.record_inference(np.array([1.0, 1.0], dtype=np.float32))
    obs = component.record_inference(np.array([2.0, 2.0], dtype=np.float32))

    outcome = component.evaluate_policy(observation=obs, policy_ready=obs.policy_ready)
    assert outcome is not None
    assert outcome.action == expected


def test_resolve_replay_safe_drift_action_forces_log_only_in_backtest() -> None:
    assert (
        resolve_replay_safe_drift_action(
            action=DriftActionPolicy.FAIL_CLOSED,
            is_backtesting=True,
        )
        == DriftActionPolicy.LOG_ONLY
    )
    assert (
        resolve_replay_safe_drift_action(
            action=DriftActionPolicy.DEGRADED,
            is_backtesting=False,
        )
        == DriftActionPolicy.DEGRADED
    )
