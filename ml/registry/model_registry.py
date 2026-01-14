"""
Compatibility shim for ModelRegistry canonical import paths.
"""

from __future__ import annotations

from ml.registry.model_registry_facade import ModelRegistry
from ml.registry.model_registry_facade import ModelRegistryFacade


__all__ = [
    "ModelRegistry",
    "ModelRegistryFacade",
]
