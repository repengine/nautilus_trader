"""
Backwards-compatibility shim for earnings raw writer.

DEPRECATED: Import from ml.features.earnings instead:
    from ml.features.earnings import EarningsParquetRawWriter
"""

import warnings

from ml.features.earnings.raw_writer import EarningsParquetRawWriter


warnings.warn(
    "ml.stores.earnings_raw_writer is deprecated. "
    "Import from ml.features.earnings instead.",
    DeprecationWarning,
    stacklevel=2,
)


__all__ = ["EarningsParquetRawWriter"]
