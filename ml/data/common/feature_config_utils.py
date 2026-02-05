"""
Feature config normalization helpers for data-domain components.

This module centralizes conversion from MLFeatureConfig to the canonical
FeatureConfig used by the feature pipeline.
"""

from __future__ import annotations

from typing import Any, cast

from ml.config.base import MLFeatureConfig
from ml.features.config import FeatureConfig
from ml.registry.base import DataRequirements


def normalize_feature_config(
    feature_config: FeatureConfig | MLFeatureConfig | None,
) -> FeatureConfig:
    """
    Normalize feature config to a FeatureConfig instance.

    Args:
        feature_config: Config to normalize (FeatureConfig or MLFeatureConfig).

    Returns:
        Normalized FeatureConfig instance.
    """
    if isinstance(feature_config, FeatureConfig):
        return feature_config
    if isinstance(feature_config, MLFeatureConfig):
        raw: dict[str, object]
        try:
            import msgspec

            raw = cast(dict[str, object], msgspec.to_builtins(feature_config))
        except Exception:
            raw = cast(dict[str, object], getattr(feature_config, "__dict__", {}) or {})

        allowed_fields = set(FeatureConfig.__annotations__.keys())
        filtered: dict[str, object] = {k: v for k, v in raw.items() if k in allowed_fields}

        data_req = filtered.get("data_requirements")
        if isinstance(data_req, str):
            try:
                filtered["data_requirements"] = DataRequirements(data_req)
            except Exception:
                ...

        return FeatureConfig(**cast(dict[str, Any], filtered)) if filtered else FeatureConfig()

    return FeatureConfig()


__all__ = ["normalize_feature_config"]
