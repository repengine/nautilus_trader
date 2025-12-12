"""Compatibility shim for XBRL parser utilities."""

from __future__ import annotations

import warnings

from ml.features.earnings.ingestion.xbrl_parser import XBRL_TAGS
from ml.features.earnings.ingestion.xbrl_parser import XBRLParser


warnings.warn(
    "ml.data.earnings.xbrl_parser is deprecated; "
    "import from ml.features.earnings.ingestion.xbrl_parser instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["XBRL_TAGS", "XBRLParser"]
