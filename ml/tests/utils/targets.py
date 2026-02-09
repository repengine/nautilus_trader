#!/usr/bin/env python3
"""
Target semantics helpers for tests.
"""

from __future__ import annotations

from dataclasses import asdict

from ml.config.targets import BinaryTargetConfig
from ml.config.targets import HORIZON_RESOLUTION_BAR_INDEX
from ml.config.targets import HorizonResolutionMode
from ml.config.targets import MulticlassTargetConfig
from ml.config.targets import RegressionTargetConfig
from ml.config.targets import TargetHorizonSpec
from ml.config.targets import TargetSemanticsConfig
from ml.config.targets import decimal_to_bps


def build_default_target_semantics(
    *,
    horizon_minutes: int = 15,
    threshold: float = 0.001,
    legacy_aliases: bool = False,
    primary_target: str | None = None,
    horizon_resolution_mode: HorizonResolutionMode = HORIZON_RESOLUTION_BAR_INDEX,
    wall_clock_timestamp_column: str = "timestamp",
) -> TargetSemanticsConfig:
    """
    Build a minimal target semantics configuration for tests.

    Args:
        horizon_minutes: Horizon in minutes.
        threshold: Binary threshold in decimal return units.
        legacy_aliases: Whether to emit legacy alias columns.
        primary_target: Optional primary target column name.
        horizon_resolution_mode: Horizon resolution mode (`bar_index`/`wall_clock`).
        wall_clock_timestamp_column: Timestamp column for `wall_clock` mode.

    Returns:
        TargetSemanticsConfig instance.
    """
    return TargetSemanticsConfig(
        horizons=(TargetHorizonSpec(minutes=horizon_minutes),),
        binary=BinaryTargetConfig(
            enabled=True,
            threshold_bps=decimal_to_bps(threshold),
            return_basis="raw",
        ),
        multiclass=MulticlassTargetConfig(enabled=False),
        regression=RegressionTargetConfig(enabled=False),
        primary_target=primary_target,
        legacy_aliases=legacy_aliases,
        horizon_resolution_mode=horizon_resolution_mode,
        wall_clock_timestamp_column=wall_clock_timestamp_column,
    )


def build_default_target_semantics_payload(
    *,
    horizon_minutes: int = 15,
    threshold: float = 0.001,
    legacy_aliases: bool = False,
    primary_target: str | None = None,
    horizon_resolution_mode: HorizonResolutionMode = HORIZON_RESOLUTION_BAR_INDEX,
    wall_clock_timestamp_column: str = "timestamp",
) -> dict[str, object]:
    """
    Build a JSON-ready target semantics payload for orchestration configs.

    Args:
        horizon_minutes: Horizon in minutes.
        threshold: Binary threshold in decimal return units.
        legacy_aliases: Whether to emit legacy alias columns.
        primary_target: Optional primary target column name.
        horizon_resolution_mode: Horizon resolution mode (`bar_index`/`wall_clock`).
        wall_clock_timestamp_column: Timestamp column for `wall_clock` mode.

    Returns:
        Target semantics payload dictionary.
    """
    semantics = build_default_target_semantics(
        horizon_minutes=horizon_minutes,
        threshold=threshold,
        legacy_aliases=legacy_aliases,
        primary_target=primary_target,
        horizon_resolution_mode=horizon_resolution_mode,
        wall_clock_timestamp_column=wall_clock_timestamp_column,
    )
    return asdict(semantics)
