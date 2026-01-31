"""
Signal metadata helpers for ML actors.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nautilus_trader.model.data import Bar

from ml.schema import PREDICTION_SURFACE_V1


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


def build_prediction_surface_metadata(
    *,
    neutral_band: float,
    decision_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build canonical prediction surface metadata for signal payloads.

    Parameters
    ----------
    neutral_band : float
        Neutral band value applied around the canonical threshold.
    decision_metadata : Mapping[str, Any] | None, optional
        Normalized decision metadata payload to include.

    Returns
    -------
    dict[str, Any]
        Metadata fields describing the canonical prediction surface.

    """
    metadata: dict[str, Any] = {
        "prediction_surface": PREDICTION_SURFACE_V1.surface,
        "prediction_surface_version": PREDICTION_SURFACE_V1.version,
        "neutral_band": float(neutral_band),
        "confidence_semantics": PREDICTION_SURFACE_V1.confidence_semantics,
    }
    if decision_metadata is not None:
        metadata["decision_metadata"] = dict(decision_metadata)
    return metadata
