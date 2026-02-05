"""
Compatibility wrapper for TFT dataset builder.

This module preserves the legacy import path while routing calls to the
component-based facade implementation.
"""

from __future__ import annotations

from ml.data.catalog_utils import bars_to_dataframe
from ml.data.tft_dataset_builder_facade import SchemaValidationError
from ml.data.tft_dataset_builder_facade import TFTDatasetBuilder
from ml.data.tft_dataset_builder_facade import TFTDatasetBuilderFacade


__all__ = [
    "SchemaValidationError",
    "TFTDatasetBuilder",
    "TFTDatasetBuilderFacade",
    "bars_to_dataframe",
]
