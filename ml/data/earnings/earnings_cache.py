"""Compatibility shim for earnings cache."""

from __future__ import annotations

import warnings

from ml.features.earnings.cache import EarningsCache


warnings.warn(
    "ml.data.earnings.earnings_cache is deprecated; "
    "import from ml.features.earnings.cache instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["EarningsCache"]
