"""
Runtime drift monitoring helpers for actor-side inference remediation.

This module provides a single implementation for:
- Drift policy configuration access
- Threshold mapping by policy action
- Lightweight runtime drift scoring
- Policy outcome evaluation for actor remediation hooks

Defaults are intentionally permissive and additive; strict behavior requires
explicit policy/threshold overrides.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

from ml.common.metrics import feature_drift_score
from ml.config.policy import DriftActionPolicy


if TYPE_CHECKING:
    from logging import Logger


_DEFAULT_LOG_ONLY_THRESHOLD = 0.35
_DEFAULT_DEGRADED_THRESHOLD = 0.75
_DEFAULT_FAIL_CLOSED_THRESHOLD = 1.25
_DEFAULT_MIN_BASELINE_SAMPLES = 200
_DEFAULT_MIN_OBSERVED_SAMPLES = 240

_FEATURE_NAME_RUNTIME_AGGREGATE = "runtime_aggregate"


@dataclass(frozen=True, slots=True)
class DriftThresholds:
    """
    Drift threshold mapping by action policy.

    Parameters
    ----------
    log_only : float
        Threshold used when policy action is ``log_only``.
    degraded : float
        Threshold used when policy action is ``degraded``.
    fail_closed : float
        Threshold used when policy action is ``fail_closed``.

    """

    log_only: float
    degraded: float
    fail_closed: float


@dataclass(frozen=True, slots=True)
class DriftPolicyConfig:
    """
    Resolved drift policy configuration for runtime monitoring.

    Parameters
    ----------
    action_policy : DriftActionPolicy
        Configured policy action.
    thresholds : DriftThresholds
        Threshold mapping for each policy action.
    min_baseline_samples : int
        Minimum baseline sample count before drift actions can be evaluated.
    min_observed_samples : int
        Minimum total observed sample count before drift actions can be evaluated.

    """

    action_policy: DriftActionPolicy
    thresholds: DriftThresholds
    min_baseline_samples: int
    min_observed_samples: int


@dataclass(frozen=True, slots=True)
class DriftObservation:
    """
    Drift observation recorded from one inference-time feature vector.

    Parameters
    ----------
    drift_score : float
        Aggregated drift score for the latest feature vector.
    sample_count : int
        Total observed inference samples.
    baseline_samples : int
        Number of samples used in baseline estimation.
    baseline_ready : bool
        Whether minimum baseline sample requirement is satisfied.
    policy_ready : bool
        Whether both baseline and total sample windows satisfy policy gating.

    """

    drift_score: float
    sample_count: int
    baseline_samples: int
    baseline_ready: bool
    policy_ready: bool


@dataclass(frozen=True, slots=True)
class DriftPolicyOutcome:
    """
    Evaluated drift policy outcome.

    Parameters
    ----------
    action : DriftActionPolicy
        Effective action to apply.
    configured_action : DriftActionPolicy
        Original configured action before any runtime adjustments.
    reason : str
        Low-cardinality reason label for metrics and logs.
    drift_score : float
        Drift score that triggered the outcome.
    threshold : float
        Threshold used for evaluation.

    """

    action: DriftActionPolicy
    configured_action: DriftActionPolicy
    reason: str
    drift_score: float
    threshold: float


def _resolve_env(source: Mapping[str, str] | None) -> Mapping[str, str]:
    """
    Resolve environment mapping for policy overrides.
    """
    if source is not None:
        return source
    return cast(Mapping[str, str], os.environ)


def _coerce_non_negative_float(value: Any, *, default: float) -> float:
    """
    Coerce arbitrary value to a non-negative float.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if parsed < 0.0:
        return float(default)
    return float(parsed)


def _coerce_positive_int(value: Any, *, default: int) -> int:
    """
    Coerce arbitrary value to a positive integer.
    """
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    if parsed <= 0:
        return int(default)
    return int(parsed)


def _coerce_drift_action(value: object) -> DriftActionPolicy:
    """
    Coerce arbitrary action value into :class:`DriftActionPolicy`.
    """
    if isinstance(value, DriftActionPolicy):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        for candidate in DriftActionPolicy:
            if normalized == candidate.value or normalized == candidate.name.lower():
                return candidate
    return DriftActionPolicy.LOG_ONLY


def threshold_for_policy_action(
    *,
    thresholds: DriftThresholds,
    action: DriftActionPolicy,
) -> float:
    """
    Resolve threshold for a given policy action.

    Parameters
    ----------
    thresholds : DriftThresholds
        Threshold mapping.
    action : DriftActionPolicy
        Target policy action.

    Returns
    -------
    float
        Threshold value for ``action``.

    """
    if action == DriftActionPolicy.FAIL_CLOSED:
        return float(thresholds.fail_closed)
    if action == DriftActionPolicy.DEGRADED:
        return float(thresholds.degraded)
    return float(thresholds.log_only)


def resolve_replay_safe_drift_action(
    *,
    action: DriftActionPolicy,
    is_backtesting: bool,
) -> DriftActionPolicy:
    """
    Downgrade strict drift actions to ``log_only`` in replay/backtest runtimes.
    """
    if is_backtesting and action != DriftActionPolicy.LOG_ONLY:
        return DriftActionPolicy.LOG_ONLY
    return action


def resolve_drift_policy_config(
    config: object,
    *,
    env: Mapping[str, str] | None = None,
) -> DriftPolicyConfig:
    """
    Resolve drift policy config from actor config + environment overrides.

    Supported env overrides
    -----------------------
    ML_DRIFT_LOG_ONLY_THRESHOLD
        Drift threshold for ``log_only`` action.
    ML_DRIFT_DEGRADED_THRESHOLD
        Drift threshold for ``degraded`` action.
    ML_DRIFT_FAIL_CLOSED_THRESHOLD
        Drift threshold for ``fail_closed`` action.
    ML_DRIFT_MIN_BASELINE_SAMPLES
        Minimum baseline samples before policy actions can trigger.
    ML_DRIFT_MIN_SAMPLE_WINDOW
        Minimum observed samples before policy actions can trigger.

    """
    source = _resolve_env(env)
    remediation_policy = getattr(config, "remediation_policy", None)
    action_policy = _coerce_drift_action(
        getattr(remediation_policy, "drift_action_policy", DriftActionPolicy.LOG_ONLY),
    )

    default_baseline_samples = _coerce_positive_int(
        getattr(config, "drift_min_baseline_samples", _DEFAULT_MIN_BASELINE_SAMPLES),
        default=_DEFAULT_MIN_BASELINE_SAMPLES,
    )
    default_observed_samples = _coerce_positive_int(
        getattr(
            config,
            "drift_min_sample_window",
            max(default_baseline_samples, _DEFAULT_MIN_OBSERVED_SAMPLES),
        ),
        default=max(default_baseline_samples, _DEFAULT_MIN_OBSERVED_SAMPLES),
    )

    log_only_threshold = _coerce_non_negative_float(
        source.get(
            "ML_DRIFT_LOG_ONLY_THRESHOLD",
            getattr(config, "drift_log_only_threshold", _DEFAULT_LOG_ONLY_THRESHOLD),
        ),
        default=_DEFAULT_LOG_ONLY_THRESHOLD,
    )
    degraded_threshold = _coerce_non_negative_float(
        source.get(
            "ML_DRIFT_DEGRADED_THRESHOLD",
            getattr(config, "drift_degraded_threshold", _DEFAULT_DEGRADED_THRESHOLD),
        ),
        default=max(_DEFAULT_DEGRADED_THRESHOLD, log_only_threshold),
    )
    fail_closed_threshold = _coerce_non_negative_float(
        source.get(
            "ML_DRIFT_FAIL_CLOSED_THRESHOLD",
            getattr(config, "drift_fail_closed_threshold", _DEFAULT_FAIL_CLOSED_THRESHOLD),
        ),
        default=max(_DEFAULT_FAIL_CLOSED_THRESHOLD, degraded_threshold),
    )

    # Keep ordering monotonic regardless of external overrides.
    degraded_threshold = max(degraded_threshold, log_only_threshold)
    fail_closed_threshold = max(fail_closed_threshold, degraded_threshold)

    min_baseline_samples = _coerce_positive_int(
        source.get("ML_DRIFT_MIN_BASELINE_SAMPLES", default_baseline_samples),
        default=default_baseline_samples,
    )
    min_observed_samples = _coerce_positive_int(
        source.get("ML_DRIFT_MIN_SAMPLE_WINDOW", default_observed_samples),
        default=max(default_observed_samples, min_baseline_samples),
    )
    min_observed_samples = max(min_observed_samples, min_baseline_samples)

    thresholds = DriftThresholds(
        log_only=float(log_only_threshold),
        degraded=float(degraded_threshold),
        fail_closed=float(fail_closed_threshold),
    )
    return DriftPolicyConfig(
        action_policy=action_policy,
        thresholds=thresholds,
        min_baseline_samples=min_baseline_samples,
        min_observed_samples=min_observed_samples,
    )


def evaluate_drift_policy_outcome(
    *,
    observation: DriftObservation,
    policy_config: DriftPolicyConfig,
    policy_ready: bool,
    reason: str = "runtime_feature_drift",
) -> DriftPolicyOutcome | None:
    """
    Evaluate whether a drift observation should trigger policy action.

    Parameters
    ----------
    observation : DriftObservation
        Latest drift observation.
    policy_config : DriftPolicyConfig
        Drift policy configuration.
    policy_ready : bool
        Final gating result for policy application.
    reason : str, default "runtime_feature_drift"
        Reason label for downstream observability.

    Returns
    -------
    DriftPolicyOutcome | None
        Outcome when threshold is exceeded; otherwise ``None``.

    """
    if not policy_ready:
        return None

    configured_action = policy_config.action_policy
    threshold = threshold_for_policy_action(
        thresholds=policy_config.thresholds,
        action=configured_action,
    )
    if float(observation.drift_score) < float(threshold):
        return None

    return DriftPolicyOutcome(
        action=configured_action,
        configured_action=configured_action,
        reason=reason,
        drift_score=float(observation.drift_score),
        threshold=float(threshold),
    )


class DriftMonitoringComponent:
    """
    Runtime drift monitor for inference-time feature vectors.

    The monitor keeps a running baseline mean during warm-up and computes an
    aggregate normalized absolute deviation score once the baseline is ready.
    """

    def __init__(
        self,
        *,
        n_features: int,
        policy_config: DriftPolicyConfig,
        actor_id: str,
        feature_set_id: str = "default",
        log: Logger | None = None,
    ) -> None:
        """
        Initialize drift monitoring component.
        """
        if n_features <= 0:
            raise ValueError(f"n_features ({n_features}) must be > 0")

        self._n_features = int(n_features)
        self._policy_config = policy_config
        self._actor_id = actor_id or "unknown"
        self._feature_set_id = feature_set_id or "default"
        self._log = log

        self._sample_count = 0
        self._baseline_count = 0
        self._shape_mismatch_logged = False

        self._baseline_floor = np.float32(1e-6)
        self._baseline_mean: npt.NDArray[np.float32] = np.zeros(self._n_features, dtype=np.float32)
        self._aligned_features: npt.NDArray[np.float32] = np.zeros(
            self._n_features,
            dtype=np.float32,
        )
        self._abs_dev: npt.NDArray[np.float32] = np.zeros(self._n_features, dtype=np.float32)
        self._scale: npt.NDArray[np.float32] = np.zeros(self._n_features, dtype=np.float32)

    @property
    def policy_config(self) -> DriftPolicyConfig:
        """
        Return resolved drift policy config.
        """
        return self._policy_config

    def record_inference(self, features: npt.NDArray[np.float32]) -> DriftObservation:
        """
        Record inference-time feature vector and compute drift observation.
        """
        vector = self._align_features(features)
        self._sample_count += 1

        if self._baseline_count < self._policy_config.min_baseline_samples:
            self._baseline_count += 1
            weight = np.float32(1.0 / float(self._baseline_count))
            np.subtract(vector, self._baseline_mean, out=self._abs_dev)
            self._baseline_mean += self._abs_dev * weight
            drift_score = 0.0
        else:
            np.subtract(vector, self._baseline_mean, out=self._abs_dev)
            np.abs(self._abs_dev, out=self._abs_dev)
            np.abs(self._baseline_mean, out=self._scale)
            np.maximum(self._scale, self._baseline_floor, out=self._scale)
            np.divide(self._abs_dev, self._scale, out=self._abs_dev)
            drift_score = float(np.mean(self._abs_dev))

            # Slow baseline adaptation on the hot path keeps long runs stable.
            alpha = np.float32(1.0 / float(max(self._policy_config.min_baseline_samples, 1)))
            np.subtract(vector, self._baseline_mean, out=self._scale)
            self._baseline_mean += self._scale * alpha

        baseline_ready = self._baseline_count >= self._policy_config.min_baseline_samples
        policy_ready = baseline_ready and (
            self._sample_count >= self._policy_config.min_observed_samples
        )
        self._record_drift_score_metric(score=drift_score)

        return DriftObservation(
            drift_score=float(drift_score),
            sample_count=int(self._sample_count),
            baseline_samples=int(self._baseline_count),
            baseline_ready=bool(baseline_ready),
            policy_ready=bool(policy_ready),
        )

    def evaluate_policy(
        self,
        *,
        observation: DriftObservation,
        policy_ready: bool,
    ) -> DriftPolicyOutcome | None:
        """
        Evaluate policy outcome for a drift observation.
        """
        return evaluate_drift_policy_outcome(
            observation=observation,
            policy_config=self._policy_config,
            policy_ready=policy_ready,
        )

    def reset_runtime_state(self) -> None:
        """
        Reset drift baseline and counters for replay/rewind invalidation.
        """
        self._sample_count = 0
        self._baseline_count = 0
        self._shape_mismatch_logged = False
        self._baseline_mean.fill(np.float32(0.0))
        self._aligned_features.fill(np.float32(0.0))
        self._abs_dev.fill(np.float32(0.0))
        self._scale.fill(np.float32(0.0))

    def _align_features(self, features: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Align arbitrary feature vector shape to configured model feature width.
        """
        values = np.asarray(features, dtype=np.float32).reshape(-1)
        if values.size == self._n_features:
            return values

        self._aligned_features.fill(np.float32(0.0))
        count = min(values.size, self._n_features)
        if count > 0:
            self._aligned_features[:count] = values[:count]
        if self._log is not None and not self._shape_mismatch_logged:
            self._log.debug(
                "ml_actor.drift_feature_width_mismatch",
                extra={
                    "actor_id": self._actor_id,
                    "expected": self._n_features,
                    "actual": int(values.size),
                },
            )
            self._shape_mismatch_logged = True
        return self._aligned_features

    def _record_drift_score_metric(self, *, score: float) -> None:
        """
        Record aggregate drift score for observability.
        """
        try:
            feature_drift_score.labels(
                feature_set=self._feature_set_id,
                feature_name=_FEATURE_NAME_RUNTIME_AGGREGATE,
            ).set(max(0.0, float(score)))
        except Exception as exc:
            if self._log is not None:
                self._log.debug(
                    "ml_actor.runtime_drift_metric_failed",
                    exc_info=True,
                    extra={
                        "actor_id": self._actor_id,
                        "error": str(exc),
                    },
                )


__all__ = [
    "DriftMonitoringComponent",
    "DriftObservation",
    "DriftPolicyConfig",
    "DriftPolicyOutcome",
    "DriftThresholds",
    "evaluate_drift_policy_outcome",
    "resolve_drift_policy_config",
    "resolve_replay_safe_drift_action",
    "threshold_for_policy_action",
]
