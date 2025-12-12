"""
Backwards-compatibility shim for earnings store.

DEPRECATED: Import from ml.features.earnings instead:
    from ml.features.earnings import EarningsStore, DummyEarningsStore
"""

import warnings

from ml.features.earnings.store import DummyEarningsStore
from ml.features.earnings.store import EarningsStore


warnings.warn(
    "ml.stores.earnings_store is deprecated. "
    "Import from ml.features.earnings instead.",
    DeprecationWarning,
    stacklevel=2,
)


__all__ = ["DummyEarningsStore", "EarningsStore"]
