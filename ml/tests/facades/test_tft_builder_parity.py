"""
Contract tests for TFTDatasetBuilder facade alias.
"""

from __future__ import annotations


def test_facade_wraps_builder() -> None:
    """Facade should wrap (not alias) the canonical TFTDatasetBuilder."""
    from ml.data.tft_dataset_builder import TFTDatasetBuilder
    from ml.data.tft_dataset_builder_facade import TFTDatasetBuilderFacade

    assert TFTDatasetBuilderFacade is not TFTDatasetBuilder
    assert hasattr(TFTDatasetBuilderFacade, "build_training_dataset")
