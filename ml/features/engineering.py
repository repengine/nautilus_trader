"""
Legacy feature engineering compatibility layer.
"""

from __future__ import annotations

from ml.features import facade


FeatureConfig = facade.FeatureConfig
FeatureEngineer = facade.FeatureEngineer
IndicatorManager = facade.IndicatorManager

__all__ = ["FeatureConfig", "FeatureEngineer", "IndicatorManager"]
