"""
Compatibility shim for EarningsAugmenter.

The augmenter now lives in ``ml.training.datasets.augmenters.earnings_augmenter``.
"""

from __future__ import annotations

from ml.training.datasets.augmenters.earnings_augmenter import EarningsAugmenter


__all__ = ["EarningsAugmenter"]
