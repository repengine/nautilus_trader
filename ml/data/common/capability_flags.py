"""
Capability flag helpers for dataset metadata.

Centralizes capability flag extraction from dataset builders.
"""

from __future__ import annotations

from typing import Protocol


class CapabilityFlagSource(Protocol):
    """
    Minimal interface for capability flag extraction.
    """

    include_macro: bool
    include_macro_revisions: bool
    include_macro_deltas: bool
    include_calendar: bool
    include_calendar_lags: bool
    include_clustering_tags: bool
    include_context_features: bool
    include_events: bool
    include_earnings: bool
    include_l2: bool
    include_micro: bool


def capability_flags_from_builder(builder: CapabilityFlagSource) -> dict[str, bool]:
    """
    Extract capability flags from a dataset builder.

    Args:
        builder: Dataset builder exposing capability booleans.

    Returns:
        Mapping of capability flags to booleans.
    """
    include_l2 = bool(getattr(builder, "include_l2", False))
    include_micro = bool(getattr(builder, "include_micro", False)) or include_l2
    return {
        "include_macro": bool(getattr(builder, "include_macro", False)),
        "include_macro_revisions": bool(getattr(builder, "include_macro_revisions", False)),
        "include_macro_deltas": bool(getattr(builder, "include_macro_deltas", False)),
        "include_calendar": bool(getattr(builder, "include_calendar", False)),
        "include_calendar_lags": bool(getattr(builder, "include_calendar_lags", False)),
        "include_clustering_tags": bool(getattr(builder, "include_clustering_tags", False)),
        "include_context_features": bool(getattr(builder, "include_context_features", False)),
        "include_events": bool(getattr(builder, "include_events", False)),
        "include_earnings": bool(getattr(builder, "include_earnings", False)),
        "include_l2": include_l2,
        "include_micro": include_micro,
    }
