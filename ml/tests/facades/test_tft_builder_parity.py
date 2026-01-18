"""
Contract tests for TFTDatasetBuilder facade alias.
"""

from __future__ import annotations


def test_facade_aliases_builder() -> None:
    """Facade should alias the canonical TFTDatasetBuilder."""
    from ml.data.tft_dataset_builder import TFTDatasetBuilder
    from ml.data.tft_dataset_builder_facade import TFTDatasetBuilderFacade

    assert TFTDatasetBuilderFacade is TFTDatasetBuilder
