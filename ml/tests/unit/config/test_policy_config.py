from __future__ import annotations

import pytest

from ml.config.policy import ActorRemediationPolicyConfig
from ml.config.policy import CausalityMonotonicEnforcement
from ml.config.policy import DriftActionPolicy
from ml.config.policy import InferenceTimeoutAction
from ml.config.policy import MLFailureAction
from ml.config.policy import RegistryCompatibilityPolicyConfig


pytestmark = pytest.mark.unit


def test_actor_remediation_policy_from_env_defaults_on_invalid_enum_values() -> None:
    policy = ActorRemediationPolicyConfig.from_env(
        env={
            "ML_INFERENCE_TIMEOUT_ACTION": "invalid",
            "ML_DRIFT_ACTION_POLICY": "invalid",
            "ML_CAUSALITY_MONOTONIC_ENFORCEMENT": "invalid",
            "ML_FAILURE_ACTION": "invalid",
            "ML_DETERMINISTIC_MODE": "true",
        },
    )
    assert policy.inference_timeout_action == InferenceTimeoutAction.DROP
    assert policy.drift_action_policy == DriftActionPolicy.LOG_ONLY
    assert policy.causality_monotonic_enforcement == CausalityMonotonicEnforcement.WARN_ONLY
    assert policy.ml_failure_action == MLFailureAction.LOG_ONLY
    assert policy.deterministic_mode is True


def test_actor_remediation_policy_from_env_defaults_on_blank_enum_values() -> None:
    policy = ActorRemediationPolicyConfig.from_env(
        env={
            "ML_INFERENCE_TIMEOUT_ACTION": "   ",
            "ML_DRIFT_ACTION_POLICY": "",
            "ML_CAUSALITY_MONOTONIC_ENFORCEMENT": " ",
            "ML_FAILURE_ACTION": "\t",
        },
    )
    assert policy.inference_timeout_action == InferenceTimeoutAction.DROP
    assert policy.drift_action_policy == DriftActionPolicy.LOG_ONLY
    assert policy.causality_monotonic_enforcement == CausalityMonotonicEnforcement.WARN_ONLY
    assert policy.ml_failure_action == MLFailureAction.LOG_ONLY


def test_registry_compatibility_policy_from_env_supports_legacy_strict_flag() -> None:
    policy = RegistryCompatibilityPolicyConfig.from_env(
        env={"ML_STRICT_FEATURE_PARITY": "true"},
    )
    assert policy.strict_model_compatibility is True
    assert policy.allow_compatibility_migration_override is False


def test_registry_compatibility_policy_legacy_strict_can_enable_migration_override() -> None:
    policy = RegistryCompatibilityPolicyConfig.from_env(
        env={
            "ML_STRICT_FEATURE_PARITY": "true",
            "ML_ALLOW_COMPATIBILITY_MIGRATION_OVERRIDE": "true",
        },
    )
    assert policy.strict_model_compatibility is True
    assert policy.allow_compatibility_migration_override is True
