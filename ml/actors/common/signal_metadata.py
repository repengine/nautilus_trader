"""
Signal metadata helpers for ML actors.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nautilus_trader.model.data import Bar


def build_signal_metadata(
    bar: Bar,
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build signal metadata with bar context.

    Parameters
    ----------
    bar : Bar
        Source bar for the signal.
    extra : Mapping[str, Any] | None, optional
        Extra metadata fields to merge.

    Returns
    -------
    dict[str, Any]
        Metadata dict including bar close and bar spec.

    """
    metadata: dict[str, Any] = {
        "bar_close": float(bar.close.as_double()),
        "bar_spec": str(bar.bar_type.spec),
    }
    if extra:
        metadata.update(extra)
    return metadata
