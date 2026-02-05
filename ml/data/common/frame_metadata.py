"""
Frame metadata extraction helpers for dataset building.

Centralizes extraction logic so facade and legacy builders share one
implementation for source metadata handling.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from ml._imports import pl as pl_runtime


if TYPE_CHECKING:
    import polars as _pl
else:  # pragma: no cover - typing fallback
    _pl = Any


pl: Any = cast(Any, pl_runtime)

logger = logging.getLogger(__name__)


def extract_frame_metadata(frame: _pl.DataFrame) -> tuple[str | None, None, None]:
    """
    Extract source metadata from a frame when available.

    Args:
        frame: Polars DataFrame that may include a ``source_dataset`` column.

    Returns:
        Tuple of (source_dataset, None, None) for legacy compatibility.

    Example:
        >>> pl = __import__("polars")
        >>> df = pl.DataFrame({"source_dataset": ["foo"]})
        >>> extract_frame_metadata(df)[0]
        'foo'
    """
    if pl is None:
        return None, None, None
    source_dataset: str | None = None

    if "source_dataset" in frame.columns:
        try:
            values = (
                frame.get_column("source_dataset")
                .drop_nulls()
                .unique()
                .to_list()
            )
            if len(values) == 1 and values[0]:
                source_dataset = str(values[0])
        except Exception:
            logger.debug("Failed to extract frame metadata", exc_info=True)
            source_dataset = None
    return source_dataset, None, None
