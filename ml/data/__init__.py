"""
ML data loading utilities for Nautilus Trader.

This package provides high-level data loading utilities specifically designed for ML
workflows in the cold path (training and research). All loaders integrate seamlessly
with Nautilus Trader's data infrastructure and return Polars DataFrames for efficient ML
processing.

"""

from ml.data.loader import MLDataLoader
from ml.data.loader import load_ml_data


__all__ = [
    "MLDataLoader",
    "load_ml_data",
]
