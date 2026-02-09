"""
Static-audit remediation policy scaffolding.

This module defines additive, configuration-only policy controls used by the
remediation program. Defaults are intentionally permissive to avoid behavior
changes until later slices wire enforcement.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from enum import Enum
from typing import TypeVar

from ml.config._env_utils import ensure_env as _ensure_env
from ml.config._env_utils import env_truthy as _env_truthy
from nautilus_trader.common.config import NautilusConfig


LOGGER = logging.getLogger(__name__)

_EnumT = TypeVar("_EnumT", bound=Enum)


def _enum_from_env(
    source: Mapping[str, str],
    key: str,
    enum_type: type[_EnumT],
    default: _EnumT,
) -> _EnumT:
    """
    Parse enum value from environment with safe fallback.

    Parameters
    ----------
    source : Mapping[str, str]
        Environment value source.
    key : str
        Environment key name.
    enum_type : type[_EnumT]
        Target enum class.
    default : _EnumT
        Default enum value when parsing fails.

    Returns
    -------
    _EnumT
        Parsed enum value or ``default`` when invalid/unset.

    """
    raw = source.get(key)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if not normalized:
        return default
    for candidate in enum_type:
        if normalized == str(candidate.value).lower() or normalized == candidate.name.lower():
            return candidate
    LOGGER.debug(
        "invalid_enum_env_override",
        extra={"key": key, "value": raw, "default": str(default.value)},
    )
    return default


class InferenceTimeoutAction(str, Enum):
    """
    Timeout action for bounded inference.
    """

    DROP = "drop"
    HALT = "halt"


class DriftActionPolicy(str, Enum):
    """
    Drift action policy for runtime monitoring.
    """

    LOG_ONLY = "log_only"
    DEGRADED = "degraded"
    FAIL_CLOSED = "fail_closed"


class CausalityMonotonicEnforcement(str, Enum):
    """
    Monotonic timestamp enforcement mode.
    """

    WARN_ONLY = "warn_only"
    DROP = "drop"
    RESET = "reset"


class MLFailureAction(str, Enum):
    """
    Policy action when ML runtime failures occur.
    """

    LOG_ONLY = "log_only"
    DEGRADED = "degraded"
    HALT = "halt"


class ActorRemediationPolicyConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Additive policy controls for actor-side remediation work.

    Parameters
    ----------
    enable_inference_deadline_guard : bool, default False
        Toggle bounded inference deadline guard behavior.
    inference_timeout_action : InferenceTimeoutAction, default DROP
        Timeout action when deadline guard is enabled.
    drift_action_policy : DriftActionPolicy, default LOG_ONLY
        Runtime drift action policy.
    causality_monotonic_enforcement : CausalityMonotonicEnforcement, default WARN_ONLY
        Enforcement mode for monotonic ingress timestamp checks.
    ml_failure_action : MLFailureAction, default LOG_ONLY
        Action policy for ML runtime failures.
    deterministic_mode : bool, default False
        Toggle deterministic execution mode for reproducibility-sensitive paths.

    """

    enable_inference_deadline_guard: bool = False
    inference_timeout_action: InferenceTimeoutAction = InferenceTimeoutAction.DROP
    drift_action_policy: DriftActionPolicy = DriftActionPolicy.LOG_ONLY
    causality_monotonic_enforcement: CausalityMonotonicEnforcement = (
        CausalityMonotonicEnforcement.WARN_ONLY
    )
    ml_failure_action: MLFailureAction = MLFailureAction.LOG_ONLY
    deterministic_mode: bool = False

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
    ) -> ActorRemediationPolicyConfig:
        """
        Build policy config from environment variables.

        Environment overrides
        ---------------------
        ML_ENABLE_INFERENCE_DEADLINE_GUARD
            Toggle deadline guard policy.
        ML_INFERENCE_TIMEOUT_ACTION
            Timeout action (drop|halt).
        ML_DRIFT_ACTION_POLICY
            Drift action policy (log_only|degraded|fail_closed).
        ML_CAUSALITY_MONOTONIC_ENFORCEMENT
            Causality mode (warn_only|drop|reset).
        ML_FAILURE_ACTION
            ML failure action (log_only|degraded|halt).
        ML_DETERMINISTIC_MODE
            Toggle deterministic execution mode.

        """
        source = _ensure_env(env)
        return cls(
            enable_inference_deadline_guard=_env_truthy(
                source,
                "ML_ENABLE_INFERENCE_DEADLINE_GUARD",
                False,
            ),
            inference_timeout_action=_enum_from_env(
                source,
                "ML_INFERENCE_TIMEOUT_ACTION",
                InferenceTimeoutAction,
                InferenceTimeoutAction.DROP,
            ),
            drift_action_policy=_enum_from_env(
                source,
                "ML_DRIFT_ACTION_POLICY",
                DriftActionPolicy,
                DriftActionPolicy.LOG_ONLY,
            ),
            causality_monotonic_enforcement=_enum_from_env(
                source,
                "ML_CAUSALITY_MONOTONIC_ENFORCEMENT",
                CausalityMonotonicEnforcement,
                CausalityMonotonicEnforcement.WARN_ONLY,
            ),
            ml_failure_action=_enum_from_env(
                source,
                "ML_FAILURE_ACTION",
                MLFailureAction,
                MLFailureAction.LOG_ONLY,
            ),
            deterministic_mode=_env_truthy(source, "ML_DETERMINISTIC_MODE", False),
        )


class RegistryCompatibilityPolicyConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Additive compatibility/integrity policy controls for registry work.

    Parameters
    ----------
    strict_model_compatibility : bool, default False
        Enable strict compatibility checks for model loading.
    allow_compatibility_migration_override : bool, default True
        Allow temporary migration override for compatibility strictness.
    allow_unsigned_artifacts : bool, default False
        Allow loading artifacts without digest/signature metadata.
    require_output_semantics : bool, default False
        Require output semantics metadata for serveable models.

    """

    strict_model_compatibility: bool = False
    allow_compatibility_migration_override: bool = True
    allow_unsigned_artifacts: bool = False
    require_output_semantics: bool = False

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
    ) -> RegistryCompatibilityPolicyConfig:
        """
        Build registry compatibility policy from environment variables.

        Environment overrides
        ---------------------
        ML_STRICT_MODEL_COMPATIBILITY / ML_STRICT_FEATURE_PARITY
            Enable strict compatibility checks.
        ML_ALLOW_COMPATIBILITY_MIGRATION_OVERRIDE
            Allow migration override for compatibility strictness.
            Defaults to ``False`` when only legacy strict parity flag is used.
        ML_ALLOW_UNSIGNED_ARTIFACTS
            Allow unsigned or digest-missing artifacts.
        ML_REQUIRE_OUTPUT_SEMANTICS
            Require output semantics metadata for serveable models.

        """
        source = _ensure_env(env)
        strict_explicit = "ML_STRICT_MODEL_COMPATIBILITY" in source
        strict_from_legacy = _env_truthy(source, "ML_STRICT_FEATURE_PARITY", False)
        migration_override_default = not (strict_from_legacy and not strict_explicit)
        return cls(
            strict_model_compatibility=_env_truthy(
                source,
                "ML_STRICT_MODEL_COMPATIBILITY",
                strict_from_legacy,
            ),
            allow_compatibility_migration_override=_env_truthy(
                source,
                "ML_ALLOW_COMPATIBILITY_MIGRATION_OVERRIDE",
                migration_override_default,
            ),
            allow_unsigned_artifacts=_env_truthy(source, "ML_ALLOW_UNSIGNED_ARTIFACTS", False),
            require_output_semantics=_env_truthy(source, "ML_REQUIRE_OUTPUT_SEMANTICS", False),
        )


__all__ = [
    "ActorRemediationPolicyConfig",
    "CausalityMonotonicEnforcement",
    "DriftActionPolicy",
    "InferenceTimeoutAction",
    "MLFailureAction",
    "RegistryCompatibilityPolicyConfig",
]
