"""
Legacy feature engineering compatibility layer.
"""

from __future__ import annotations

from ml.features import facade
from ml.features.config import FeatureConfigLike


FeatureConfig = facade.FeatureConfig
FeatureEngineer = facade.FeatureEngineer
IndicatorManager = facade.IndicatorManager
LegacyFeatureEngineer = FeatureEngineer

__all__ = [
    "FeatureConfig",
    "FeatureConfigLike",
    "FeatureEngineer",
    "IndicatorManager",
    "LegacyFeatureEngineer",
]
