"""
Registry-related configuration classes.
"""

from __future__ import annotations

from nautilus_trader.common.config import NautilusConfig
from nautilus_trader.common.config import PositiveFloat
from nautilus_trader.common.config import PositiveInt


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
    """

    max_inference_latency_ms: PositiveFloat = 5.0
    ab_models_required: PositiveInt = 2


__all__ = ["ModelRegistryConfig", "RegistryPolicyConfig"]
