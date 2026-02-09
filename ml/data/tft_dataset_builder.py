"""
Compatibility wrapper for TFT dataset builder.

This module preserves the legacy import path while routing calls to the
component-based facade implementation.
"""

from __future__ import annotations

from ml.data.catalog_utils import bars_to_dataframe
from ml.data.tft_dataset_builder_facade import SchemaValidationError
from ml.data.tft_dataset_builder_facade import TFTDatasetBuilderFacade


class TFTDatasetBuilder(TFTDatasetBuilderFacade):
    """
    Canonical TFT dataset builder wrapper.

    This class intentionally remains a thin subclass so the canonical
    ``ml.data.tft_dataset_builder.TFTDatasetBuilder`` type stays distinct from
    the facade type while preserving identical behavior.
    """


__all__ = [
    "SchemaValidationError",
    "TFTDatasetBuilder",
    "TFTDatasetBuilderFacade",
    "bars_to_dataframe",
]
