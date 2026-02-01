#!/usr/bin/env python3
"""
Target semantics helpers for tests.
"""

from __future__ import annotations

from ml.config.targets import TargetSemanticsConfig


def build_default_target_semantics(
    *,
    horizon_minutes: int = 15,
    threshold: float = 0.001,
    legacy_aliases: bool = False,
    primary_target: str | None = None,
) -> TargetSemanticsConfig:
    """
    Build a minimal target semantics configuration for tests.

    Args:
        horizon_minutes: Horizon in minutes.
        threshold: Binary threshold in decimal return units.
        legacy_aliases: Whether to emit legacy alias columns.
        primary_target: Optional primary target column name.

    Returns:
        TargetSemanticsConfig instance.
    """
    return TargetSemanticsConfig.from_legacy(
        horizon_minutes=horizon_minutes,
        threshold=threshold,
        legacy_aliases=legacy_aliases,
        primary_target=primary_target,
    )
