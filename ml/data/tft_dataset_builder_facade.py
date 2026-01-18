"""
Compatibility shim for TFT dataset builder facade imports.

This module preserves the legacy facade import path while routing all calls
through the canonical TFTDatasetBuilder implementation.
"""

from __future__ import annotations

from ml.data.common import SchemaValidationError
from ml.data.tft_dataset_builder import TFTDatasetBuilder


TFTDatasetBuilderFacade = TFTDatasetBuilder

__all__ = [
    "SchemaValidationError",
    "TFTDatasetBuilderFacade",
]
