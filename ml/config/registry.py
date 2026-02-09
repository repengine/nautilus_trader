"""
Registry-related configuration classes.
"""

from __future__ import annotations

from collections.abc import Mapping

import msgspec

from ml.config._env_utils import ensure_env as _ensure_env
from ml.config._env_utils import env_truthy as _env_truthy
from ml.config.policy import RegistryCompatibilityPolicyConfig
from nautilus_trader.common.config import NautilusConfig
from nautilus_trader.common.config import PositiveFloat
from nautilus_trader.common.config import PositiveInt


def _compatibility_policy_from_env_strict_defaults(
    *,
    env: Mapping[str, str] | None = None,
) -> RegistryCompatibilityPolicyConfig:
    """
    Build registry compatibility policy with strict-by-default env behavior.

    This keeps constructor defaults backward-compatible while tightening
    ``RegistryPolicyConfig.from_env`` for production safety.
    """
    source = _ensure_env(env)
    strict_from_legacy = _env_truthy(source, "ML_STRICT_FEATURE_PARITY", True)
    strict_model_compatibility = _env_truthy(
        source,
        "ML_STRICT_MODEL_COMPATIBILITY",
        strict_from_legacy,
    )
    return RegistryCompatibilityPolicyConfig(
        strict_model_compatibility=strict_model_compatibility,
        allow_compatibility_migration_override=_env_truthy(
            source,
            "ML_ALLOW_COMPATIBILITY_MIGRATION_OVERRIDE",
            False,
        ),
        allow_unsigned_artifacts=_env_truthy(source, "ML_ALLOW_UNSIGNED_ARTIFACTS", False),
        require_output_semantics=_env_truthy(
            source,
            "ML_REQUIRE_OUTPUT_SEMANTICS",
            strict_model_compatibility,
        ),
    )


class ModelRegistryConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration for ML model registry (paths and retention).
    """

    registry_path: str = "ml/registry"
    enable_mlflow: bool = False
    mlflow_tracking_uri: str | None = None
    auto_versioning: bool = True
    max_versions_per_model: PositiveInt = 10


class RegistryPolicyConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Policy settings for the model registry (SLOs, A/B defaults).

    Parameters
    ----------
    max_inference_latency_ms : PositiveFloat, default 5.0
        Maximum inference latency budget used by registry quality gates.
    ab_models_required : PositiveInt, default 2
        Minimum number of models required for A/B workflows.
    compatibility_policy : RegistryCompatibilityPolicyConfig, optional
        Additive compatibility/integrity rollout policy controls.

    """

    max_inference_latency_ms: PositiveFloat = 5.0
    ab_models_required: PositiveInt = 2
    compatibility_policy: RegistryCompatibilityPolicyConfig = msgspec.field(
        default_factory=RegistryCompatibilityPolicyConfig,
    )

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
    ) -> RegistryPolicyConfig:
        """
        Build :class:`RegistryPolicyConfig` from environment variables.

        Compatibility policy loading is strict-by-default in env-derived paths:
        strict compatibility and output semantics are enabled unless explicitly
        disabled via environment overrides.
        """
        return cls(
            compatibility_policy=_compatibility_policy_from_env_strict_defaults(env=env),
        )


__all__ = ["ModelRegistryConfig", "RegistryPolicyConfig"]
