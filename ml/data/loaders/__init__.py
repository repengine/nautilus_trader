"""
Data loaders for Nautilus Trader ML.

This module provides various data loaders for different data sources.

"""

from ml.data.loaders.fred_loader import FREDConfig
from ml.data.loaders.fred_loader import FREDDataLoader
from ml.data.loaders.fred_loader import FREDIndicator


__all__ = [
    "FREDConfig",
    "FREDDataLoader",
    "FREDIndicator",
]
